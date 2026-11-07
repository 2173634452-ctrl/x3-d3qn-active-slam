#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <System.h>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <iterator>

using Image = sensor_msgs::msg::Image;
using Steady = std::chrono::steady_clock;
struct Pair { Image::ConstSharedPtr rgb, depth; Steady::time_point arrived; };
int main(int argc,char **argv) {
 rclcpp::init(argc,argv);
 auto node=std::make_shared<rclcpp::Node>("x3_rgbd");
 auto config=node->declare_parameter<std::string>("config","");
 auto vocab=node->declare_parameter<std::string>("vocab","/workspace/x3-slam/Vocabulary/ORBvoc.txt");
 auto output=node->declare_parameter<std::string>("output","tracking.csv");
 const bool viewer=node->declare_parameter<bool>("viewer",false);
 const bool diagnostics=node->declare_parameter<bool>("diagnostics",false);
 ORB_SLAM3::System slam(vocab,config,ORB_SLAM3::System::RGBD,viewer);
 const bool load_localization=node->declare_parameter<bool>("load_localization",false);
 if(load_localization && !slam.SelectLoadedMapForLocalization())throw std::runtime_error("No loaded map available for localization");
 const int requested_map=node->declare_parameter<int>("localization_map_id",-1);
 if(requested_map>=0){
  if(!load_localization)throw std::runtime_error("localization_map_id requires load_localization");
  ORB_SLAM3::Map* selected=nullptr;
  for(auto m:slam.GetAtlasForAudit()->GetAllMaps())if(m->GetId()==static_cast<unsigned long>(requested_map)&&!m->IsBad()&&m->KeyFramesInMap()>0)selected=m;
  if(!selected)throw std::runtime_error("Requested loaded map is missing or empty");
  slam.GetAtlasForAudit()->ChangeMap(selected);
  RCLCPP_INFO(node->get_logger(),"AUDIT_LOCALIZATION_MAP_OVERRIDE map=%d keyframes=%lu",requested_map,static_cast<unsigned long>(selected->KeyFramesInMap()));
 }
 if(load_localization){
  std::ofstream loaded("atlas_loaded.csv");loaded<<"map_id,keyframes,map_points,selected\n";
  auto atlas=slam.GetAtlasForAudit();
  for(auto m:atlas->GetAllMaps())loaded<<m->GetId()<<','<<m->KeyFramesInMap()<<','<<m->MapPointsInMap()<<','<<(m==atlas->GetCurrentMap())<<'\n';
 }
 std::atomic<bool> blackout{false},localization{load_localization};
 std::ofstream log(output);
 log<<"stamp,process_ms,queue_ms,sim_age_s,state,extracted_keypoints,overwritten_pairs,unpaired_discarded,tx,ty,tz,qx,qy,qz,qw,map_id,maps,keyframes,map_points,blackout,localization_only\n";
 std::ofstream arrivals,quality;
 if(diagnostics){
  arrivals.open(output+".arrivals.csv");quality.open(output+".quality.csv");
  if(!arrivals||!quality)throw std::runtime_error("Cannot open diagnostic CSV files");
  arrivals<<"stream,stamp_ns,arrival_steady_ns,received_rgb,received_depth,paired,overwritten,unpaired,rgb_buffer,depth_buffer\n";
  quality<<"stamp,associated_map_points,associated_grid_cells_4x4,keypoint_grid_cells_4x4,gray_mean,gray_std,laplacian_variance,valid_depth_fraction,diagnostic_ms\n";
 }
 auto odom=node->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry",10);
 auto status=node->create_publisher<std_msgs::msg::Int32>("/slam/tracking_state",10);
 auto tracked_image=node->create_publisher<Image>("/slam/tracked_image",rclcpp::QoS(2));
 auto cloudpub=node->create_publisher<sensor_msgs::msg::PointCloud2>("/slam/map_points",rclcpp::QoS(1).transient_local());
 auto pathpub=node->create_publisher<nav_msgs::msg::Path>("/slam/path",rclcpp::QoS(1).transient_local());
 auto keypathpub=node->create_publisher<nav_msgs::msg::Path>("/slam/keyframes",rclcpp::QoS(1).transient_local());
 auto tf=std::make_shared<tf2_ros::TransformBroadcaster>(node);
 auto static_tf=std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
 Eigen::Matrix3f S;S<<0,0,1,-1,0,0,0,-1,0;
 Eigen::Vector3f mount(0.057105f,0.000017948f,node->declare_parameter<double>("camera_height",0.11905));
 geometry_msgs::msg::TransformStamped fixed;fixed.header.frame_id="slam_base_footprint";fixed.child_frame_id="slam_camera";fixed.transform.translation.x=mount.x();fixed.transform.translation.y=mount.y();fixed.transform.translation.z=mount.z();fixed.transform.rotation.w=1;static_tf->sendTransform(fixed);
 fixed.header.frame_id="slam_camera";fixed.child_frame_id="slam_camera_optical";fixed.transform.translation.x=0;fixed.transform.translation.y=0;fixed.transform.translation.z=0;Eigen::Quaternionf oq(S);fixed.transform.rotation.x=oq.x();fixed.transform.rotation.y=oq.y();fixed.transform.rotation.z=oq.z();fixed.transform.rotation.w=oq.w();static_tf->sendTransform(fixed);
 auto blacksub=node->create_subscription<std_msgs::msg::Bool>("/slam/blackout",10,[&](std_msgs::msg::Bool::ConstSharedPtr msg){blackout=msg->data;});
 auto modesub=node->create_subscription<std_msgs::msg::Bool>("/slam/localization_only",10,[&](std_msgs::msg::Bool::ConstSharedPtr msg){localization=msg->data;});
 std::mutex mutex;std::condition_variable cv;bool stop=false;
 std::map<int64_t,Image::ConstSharedPtr> rgbs,depths;std::optional<Pair> pending;
 size_t overwritten=0,unpaired=0,received_rgb=0,received_depth=0,paired=0;
 auto incoming=[&](Image::ConstSharedPtr image,bool depth) {
  const auto arrival=Steady::now();
  std::lock_guard<std::mutex> lock(mutex);
  ++(depth?received_depth:received_rgb);
  const int64_t key=int64_t(image->header.stamp.sec)*1000000000LL+image->header.stamp.nanosec;
  (depth?depths:rgbs)[key]=image;
  auto rgb=rgbs.find(key),dep=depths.find(key);
  if(rgb!=rgbs.end()&&dep!=depths.end()) {
   ++paired;
   if(pending)++overwritten;
   pending=Pair{rgb->second,dep->second,Steady::now()};
   rgbs.erase(rgb);depths.erase(dep);cv.notify_one();
  }
  for(auto *buf:{&rgbs,&depths})while(buf->size()>4){buf->erase(buf->begin());++unpaired;}
  if(diagnostics){arrivals<< (depth?"depth":"rgb") <<','<<key<<','<<std::chrono::duration_cast<std::chrono::nanoseconds>(arrival.time_since_epoch()).count()<<','<<received_rgb<<','<<received_depth<<','<<paired<<','<<overwritten<<','<<unpaired<<','<<rgbs.size()<<','<<depths.size()<<'\n';}

 };
 auto rgbsub=node->create_subscription<Image>("/camera/color",rclcpp::QoS(4).reliable(),[&](Image::ConstSharedPtr m){incoming(m,false);});
 auto depsub=node->create_subscription<Image>("/camera/depth",rclcpp::QoS(4).reliable(),[&](Image::ConstSharedPtr m){incoming(m,true);});
 std::thread worker([&]{
  bool last_mode=load_localization;double last_cloud=-100;long last_map=-1;nav_msgs::msg::Path path;path.header.frame_id="slam_map";
  while(true) {
   Pair pair;size_t dropped,missed;
   {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return stop||pending.has_value();});if(stop)break;pair=*pending;pending.reset();dropped=overwritten;missed=unpaired;}
   try {
    auto rgb=cv_bridge::toCvShare(pair.rgb,"bgr8")->image;
    cv::Mat depth;
    if(pair.depth->encoding=="32FC1")depth=cv_bridge::toCvShare(pair.depth,"32FC1")->image;
    else if(pair.depth->encoding=="16UC1")cv_bridge::toCvShare(pair.depth,"16UC1")->image.convertTo(depth,CV_32F,0.001);
    else throw std::runtime_error("Unsupported depth encoding: "+pair.depth->encoding);
    bool mode=localization.load();if(mode!=last_mode){if(mode)slam.ActivateLocalizationMode();else slam.DeactivateLocalizationMode();last_mode=mode;}
    bool dark=blackout.load();if(dark){rgb=cv::Mat::zeros(rgb.size(),rgb.type());depth=cv::Mat::zeros(depth.size(),depth.type());}
    double stamp=rclcpp::Time(pair.rgb->header.stamp).seconds();auto start=Steady::now();
    double queue=std::chrono::duration<double,std::milli>(start-pair.arrived).count();
    auto Tcw=slam.TrackRGBD(rgb,depth,stamp);
    double elapsed=std::chrono::duration<double,std::milli>(Steady::now()-start).count();
    int state=slam.GetTrackingState();auto Twc=Tcw.inverse();auto p=Twc.translation();auto q=Twc.unit_quaternion();
    cv::Mat visual=rgb.clone();auto feature_keys=slam.GetTrackedKeyPointsUn();auto feature_points=slam.GetTrackedMapPoints();
    if(diagnostics){
     const auto diagnostic_start=Steady::now();
     size_t associated=0;bool keycells[16]={},mapcells[16]={};
     for(size_t i=0;i<feature_keys.size();++i){
      const auto pt=feature_keys[i].pt;
      if(pt.x<0||pt.y<0||pt.x>=rgb.cols||pt.y>=rgb.rows)continue;
      const int cell=std::min(3,int(pt.y*4/rgb.rows))*4+std::min(3,int(pt.x*4/rgb.cols));keycells[cell]=true;
      if(i<feature_points.size()&&feature_points[i]&&!feature_points[i]->isBad()){++associated;mapcells[cell]=true;}
     }
     // Associations are not the optimizer's exact inlier count; keep the label explicit.
     cv::Mat gray,lap;cv::cvtColor(rgb,gray,cv::COLOR_BGR2GRAY);cv::Laplacian(gray,lap,CV_32F);
     cv::Scalar gm,gs,lm,ls;cv::meanStdDev(gray,gm,gs);cv::meanStdDev(lap,lm,ls);
     size_t valid=0;for(int y=0;y<depth.rows;++y){const float* row=depth.ptr<float>(y);for(int x=0;x<depth.cols;++x)if(std::isfinite(row[x])&&row[x]>0)++valid;}
     const double diagnostic_ms=std::chrono::duration<double,std::milli>(Steady::now()-diagnostic_start).count();
     quality.precision(12);quality<<stamp<<','<<associated<<','<<std::count(std::begin(mapcells),std::end(mapcells),true)<<','<<std::count(std::begin(keycells),std::end(keycells),true)<<','<<gm[0]<<','<<gs[0]<<','<<ls[0]*ls[0]<<','<<double(valid)/depth.total()<<','<<diagnostic_ms<<'\n';
    }
    for(size_t i=0;i<std::min(feature_keys.size(),feature_points.size());++i)if(feature_points[i]&&!feature_points[i]->isBad())cv::circle(visual,feature_keys[i].pt,2,cv::Scalar(70,230,90),1);
    tracked_image->publish(*cv_bridge::CvImage(pair.rgb->header,"bgr8",visual).toImageMsg());
    auto atlas=slam.GetAtlasForAudit();auto map=atlas->GetCurrentMap();
    std_msgs::msg::Int32 s;s.data=state;status->publish(s);
    log.precision(12);log<<stamp<<','<<elapsed<<','<<queue<<','<<node->now().seconds()-stamp<<','<<state<<','<<slam.GetTrackedKeyPointsUn().size()<<','<<dropped<<','<<missed<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<','<<map->GetId()<<','<<atlas->CountMaps()<<','<<map->KeyFramesInMap()<<','<<map->MapPointsInMap()<<','<<dark<<','<<mode<<'\n';log.flush();
    if(state==2){
     Eigen::Vector3f pp=S*p;Eigen::Matrix3f rot=S*Twc.rotationMatrix()*S.transpose();Eigen::Quaternionf qq(rot);
     nav_msgs::msg::Odometry out;out.header.stamp=pair.rgb->header.stamp;out.header.frame_id="slam_map";out.child_frame_id="slam_camera";out.pose.pose.position.x=pp.x();out.pose.pose.position.y=pp.y();out.pose.pose.position.z=pp.z();out.pose.pose.orientation.x=qq.x();out.pose.pose.orientation.y=qq.y();out.pose.pose.orientation.z=qq.z();out.pose.pose.orientation.w=qq.w();odom->publish(out);
     geometry_msgs::msg::TransformStamped pose;pose.header=out.header;pose.child_frame_id="slam_base_footprint";Eigen::Vector3f pb=pp-rot*mount;pose.transform.translation.x=pb.x();pose.transform.translation.y=pb.y();pose.transform.translation.z=pb.z();pose.transform.rotation=out.pose.pose.orientation;tf->sendTransform(pose);
     if(last_map!=long(map->GetId())){path.poses.clear();last_map=long(map->GetId());}
     geometry_msgs::msg::PoseStamped ps;ps.header=out.header;ps.pose=out.pose.pose;path.header=out.header;path.poses.push_back(ps);if(path.poses.size()>10000)path.poses.erase(path.poses.begin());
     if(stamp-last_cloud>=1){
      last_cloud=stamp;pathpub->publish(path);
      nav_msgs::msg::Path keypath;keypath.header=out.header;
      auto keys=atlas->GetAllKeyFrames();std::sort(keys.begin(),keys.end(),ORB_SLAM3::KeyFrame::lId);
      for(auto kf:keys)if(kf&&!kf->isBad()){
       auto pose=kf->GetPoseInverse();auto kp=S*pose.translation();Eigen::Quaternionf kq(S*pose.rotationMatrix()*S.transpose());geometry_msgs::msg::PoseStamped ks;ks.header=out.header;ks.pose.position.x=kp.x();ks.pose.position.y=kp.y();ks.pose.position.z=kp.z();ks.pose.orientation.x=kq.x();ks.pose.orientation.y=kq.y();ks.pose.orientation.z=kq.z();ks.pose.orientation.w=kq.w();keypath.poses.push_back(ks);
      }keypathpub->publish(keypath);
      std::vector<Eigen::Vector3f> points;
      for(auto mp:atlas->GetAllMapPoints())if(mp&&!mp->isBad())points.push_back(S*mp->GetWorldPos());
      sensor_msgs::msg::PointCloud2 cloud;cloud.header=out.header;cloud.height=1;cloud.is_dense=false;sensor_msgs::PointCloud2Modifier modifier(cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(points.size());sensor_msgs::PointCloud2Iterator<float> ix(cloud,"x"),iy(cloud,"y"),iz(cloud,"z");for(auto point:points){*ix=point.x();*iy=point.y();*iz=point.z();++ix;++iy;++iz;}cloudpub->publish(cloud);
     }
    }
   }catch(const std::exception &e){RCLCPP_ERROR(node->get_logger(),"Frame error: %s",e.what());}
  }
 });
 rclcpp::spin(node);
 {std::lock_guard<std::mutex> lock(mutex);stop=true;}cv.notify_one();worker.join();
 slam.Shutdown();
 auto atlas=slam.GetAtlasForAudit();auto map=atlas->GetCurrentMap();
 slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
 std::ofstream audit("atlas_final.csv");audit<<"map_id,keyframes,map_points,big_changes\n";
 for(auto m:atlas->GetAllMaps())audit<<m->GetId()<<','<<m->KeyFramesInMap()<<','<<m->MapPointsInMap()<<','<<m->GetLastBigChangeIdx()<<'\n';
 // Export every retained map independently: their coordinate frames may differ.
 std::ofstream all_edges("atlas_edges.csv");all_edges<<"map_id,kind,keyframe_a,keyframe_b\n";
 for(auto m:atlas->GetAllMaps()){
  const auto prefix="map_"+std::to_string(m->GetId());
  std::ofstream trajectory(prefix+"_keyframes.csv");trajectory<<"stamp,keyframe_id,tx,ty,tz,qx,qy,qz,qw\n";trajectory.precision(12);
  auto keys=m->GetAllKeyFrames();std::sort(keys.begin(),keys.end(),ORB_SLAM3::KeyFrame::lId);
  for(auto kf:keys)if(kf&&!kf->isBad()){
   auto pose=kf->GetPoseInverse();auto t=pose.translation();auto q=pose.unit_quaternion();
   trajectory<<kf->mTimeStamp<<','<<kf->mnId<<','<<t.x()<<','<<t.y()<<','<<t.z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<'\n';
   for(auto other:kf->GetLoopEdges())if(other&&!other->isBad()&&kf->mnId<other->mnId)all_edges<<m->GetId()<<",loop,"<<kf->mnId<<','<<other->mnId<<'\n';
   for(auto other:kf->GetMergeEdges())if(other&&!other->isBad()&&kf->mnId<other->mnId)all_edges<<m->GetId()<<",merge,"<<kf->mnId<<','<<other->mnId<<'\n';
  }
 }
 std::ofstream edges("loop_edges.csv");edges<<"keyframe_a,keyframe_b\n";
 for(auto kf:atlas->GetAllKeyFrames())if(kf&&!kf->isBad())for(auto other:kf->GetLoopEdges())if(kf->mnId<other->mnId)edges<<kf->mnId<<','<<other->mnId<<'\n';edges.close();
 std::vector<Eigen::Vector3f> points;for(auto mp:atlas->GetAllMapPoints())if(mp&&!mp->isBad())points.push_back(S*mp->GetWorldPos());
 std::ofstream ply("map.ply");ply<<"ply\nformat ascii 1.0\nelement vertex "<<points.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";for(auto p:points)ply<<p.x()<<' '<<p.y()<<' '<<p.z()<<'\n';ply.close();audit.close();
 std::ofstream done("shutdown_complete.json");done<<"{\"workers_stopped\":true,\"active_map\":"<<map->GetId()<<",\"keyframes\":"<<map->KeyFramesInMap()<<",\"map_points\":"<<points.size()<<"}\n";done.close();
 log.close();rclcpp::shutdown();return 0;
}
