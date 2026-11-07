#!/usr/bin/env bash
set -euo pipefail
root=/workspace/x3-slam
orb="$root/third_party/ORB_SLAM3"
for dep in DBoW2 g2o; do
  cmake -S "$orb/Thirdparty/$dep" -B "$orb/Thirdparty/$dep/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$orb/Thirdparty/$dep/build" -j2
done
cmake -S "$orb" -B "$orb/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$orb/build" -j2
python3 "$root/exploration/prepare_runtime.py"
cmake -S /results/exploration/runtime -B /results/exploration/runtime/build -DCMAKE_BUILD_TYPE=Release -DORB_ROOT="$orb"
cmake --build /results/exploration/runtime/build -j2
