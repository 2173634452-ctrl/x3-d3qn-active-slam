# 来源与许可范围

- `third_party/ORB_SLAM3/`：从现有服务器工程导出的修改版源码；保留其 GNU GPL v3 许可证、Dependencies.md 及文件头作者声明。上游：https://github.com/UZ-SLAMLab/ORB_SLAM3 。其中 DBoW2 / g2o / Sophus 的单独许可证保留在对应目录或文件头中。
- ORB 核心含现有工程为审计和定位恢复增加的接口；无法从无 Git 元数据的服务器目录恢复精确上游提交号。请勿把此快照标记为某个已验证官方 tag。
- `learning/`、`exploration/`、`runtime/` 等来自本项目当前服务器/本地工程与此次整理。参考 ExplORB-SLAM-Dueling-DDQN 的思路，未将原作者全部仓库整体复制到本仓库。本次未给项目作者代码新增统一的开源授权；仓库默认为私有。公开发布前应由权利人确定项目整体授权范围。
- X3 网格、房屋纹理、词典、模型权重、数据集、日志和录像没有随源码上传。外部素材需分别取得并遵循其许可。
- Dockerfile 构建时从官方源安装 ROS / Gazebo / Nav2 / Pangolin / PyTorch / Gymnasium 等依赖；这些依赖各自保留原许可证，本仓库不重新授予其权利。

来源清单 `SOURCE_MANIFEST.json` 记录首次提交中源码文件的 SHA-256（不含清单自身）；后续修改以 Git 历史为准。
