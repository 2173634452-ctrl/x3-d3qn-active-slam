#!/usr/bin/env bash
# Linux host entry point. Run from the repository or use its absolute path.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo/.env" ]]; then set -a; source "$repo/.env"; set +a; fi
: "${ASLAM_RESULTS_ROOT:?Set ASLAM_RESULTS_ROOT in .env}"
: "${ASLAM_ASSETS_ROOT:?Set ASLAM_ASSETS_ROOT in .env}"
mkdir -p "$ASLAM_RESULTS_ROOT"
for p in worlds/assets third_party/x3-robodog2 Vocabulary; do
  [[ -d "$ASLAM_ASSETS_ROOT/$p" ]] || { echo "Missing simulation dependency: $ASLAM_ASSETS_ROOT/$p" >&2; exit 1; }
done
exec docker run --rm --init --network none --ipc private --shm-size=1g \
  --cpus="${ASLAM_CPUS:-8}" --memory="${ASLAM_MEMORY:-16g}" --pids-limit=2048 \
  --security-opt=no-new-privileges:true --cap-drop ALL \
  --device="${ASLAM_GPU_DEVICE:-nvidia.com/gpu=0}" \
  --mount "type=bind,src=$repo,dst=/workspace/x3-slam" \
  --mount "type=bind,src=$ASLAM_RESULTS_ROOT,dst=/results" \
  --mount "type=bind,src=$ASLAM_ASSETS_ROOT/worlds/assets,dst=/workspace/x3-slam/worlds/assets,readonly" \
  --mount "type=bind,src=$ASLAM_ASSETS_ROOT/third_party/x3-robodog2,dst=/workspace/x3-slam/third_party/x3-robodog2,readonly" \
  --mount "type=bind,src=$ASLAM_ASSETS_ROOT/Vocabulary,dst=/workspace/x3-slam/Vocabulary,readonly" \
  --workdir /workspace/x3-slam -e ROS_LOCALHOST_ONLY=1 -e OMP_NUM_THREADS=2 \
  -e X3_SAVE_VIDEO=0 -e X3_RECORD_VIDEO=0 \
  "${ASLAM_IMAGE:-x3-aslam:local}" "$@"
