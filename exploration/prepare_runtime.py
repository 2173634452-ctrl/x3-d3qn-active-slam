"""Derive an instrumented ORB wrapper without changing the original project."""
from pathlib import Path
import hashlib, json, os

root = Path('/workspace/x3-slam')
out = Path(os.environ.get('X3_RUNTIME_DIR', '/results/exploration/runtime'))
out.mkdir(parents=True, exist_ok=True)
source = (root/'runtime/rgbd_node.cpp').read_text()
text = source.replace('#include <std_msgs/msg/int32.hpp>',
    '#include <std_msgs/msg/int32.hpp>\n#include <std_msgs/msg/string.hpp>')
old = ' auto odom=node->create_publisher<nav_msgs::msg::Odometry>'
assert text.count(old) == 1
text = text.replace(old, ' auto epochpub=node->create_publisher<std_msgs::msg::String>("/slam/map_epoch",rclcpp::QoS(1).transient_local());\n'+old)
old = 'auto map=atlas->GetCurrentMap();\n    std_msgs::msg::Int32 s;'
assert text.count(old) == 1
text = text.replace(old, '''auto map=atlas->GetCurrentMap();
    std_msgs::msg::String epoch;
    epoch.data="{\\"map_id\\":"+std::to_string(map->GetId())+",\\"big_change\\":"+std::to_string(map->GetLastBigChangeIdx())+"}";
    epochpub->publish(epoch);
    std_msgs::msg::Int32 s;''')
text = text.replace(' auto keypathpub=', ' auto anchorspub=node->create_publisher<nav_msgs::msg::Path>("/slam/map_anchors",rclcpp::QoS(1).transient_local());\n auto keypathpub=')
old = '      }keypathpub->publish(keypath);'
assert text.count(old) == 1
text = text.replace(old, old + '''
      const auto revision = map->GetLastBigChangeIdx();
      nav_msgs::msg::Path anchors;
      anchors.header = out.header;
      anchors.header.frame_id = std::to_string(map->GetId()) + ":" + std::to_string(revision);
      for(auto kf : map->GetAllKeyFrames()) {
        if(!kf || kf->isBad()) continue;
        auto pose = kf->GetPoseInverse();
        auto position = S * pose.translation();
        Eigen::Quaternionf orientation(S * pose.rotationMatrix() * S.transpose());
        geometry_msgs::msg::PoseStamped anchor;
        anchor.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(kf->mTimeStamp * 1e9)));
        anchor.header.frame_id = std::to_string(kf->mnId);
        anchor.pose.position.x = position.x();
        anchor.pose.position.y = position.y();
        anchor.pose.position.z = position.z();
        anchor.pose.orientation.x = orientation.x();
        anchor.pose.orientation.y = orientation.y();
        anchor.pose.orientation.z = orientation.z();
        anchor.pose.orientation.w = orientation.w();
        anchors.poses.push_back(anchor);
      }
      if(atlas->GetCurrentMap() == map && revision == map->GetLastBigChangeIdx())
        anchorspub->publish(anchors);
''')
(out/'rgbd_node.cpp').write_text(text)
cmake = (root/'runtime/CMakeLists.txt').read_text()
(out/'CMakeLists.txt').write_text(cmake)
(out/'provenance.json').write_text(json.dumps(dict(
    original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    instrumented_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
    change='Publish active map ID and Map::GetLastBigChangeIdx on /slam/map_epoch'), indent=2))
