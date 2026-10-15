#!/usr/bin/env bash
# Display only. Never starts/stops robot controllers or CAN services.
set -e
source /home/liberai/Desktop/colcon_ws/install/setup.bash
export ROS_DOMAIN_ID=103
HERE="${RVIZ_OVERLAY_DIR:-/home/liberai/camera_pose/rviz_overlay}"
cd /home/liberai/camera_pose
exec 9>"$HERE/.overlay.lock"
if ! flock -n 9; then echo "Overlay 已在运行，请使用已有窗口。" >&2; exit 1; fi
LOG="$HERE/overlay_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "启动日志：$LOG"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "请在 Leo3 图形桌面的终端中运行此脚本。" >&2; exit 1
fi
RUNDIR="$(mktemp -d "$HERE/.run.XXXXXX")"
CLOUD_PID=""; HAND_PID=""; RVIZ_PID=""; PROJECTION_PID=""
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$PROJECTION_PID" ]] || kill -TERM "$PROJECTION_PID" 2>/dev/null || true
  [[ -z "$RVIZ_PID" ]] || kill -INT "$RVIZ_PID" 2>/dev/null || true
  [[ -z "$HAND_PID" ]] || kill -TERM "$HAND_PID" 2>/dev/null || true
  [[ -z "$CLOUD_PID" ]] || kill -TERM "$CLOUD_PID" 2>/dev/null || true
  [[ -z "$RVIZ_PID" ]] || wait "$RVIZ_PID" 2>/dev/null || true
  [[ -z "$CLOUD_PID" ]] || wait "$CLOUD_PID" 2>/dev/null || true
  [[ -z "$HAND_PID" ]] || wait "$HAND_PID" 2>/dev/null || true
  [[ -z "$PROJECTION_PID" ]] || wait "$PROJECTION_PID" 2>/dev/null || true
  rm -f "$RUNDIR/ready" "$RUNDIR/hand_ready"; rmdir "$RUNDIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/usr/bin/python3 -u "$HERE/publish_cloud.py" --ready-file "$RUNDIR/ready" &
CLOUD_PID=$!
/usr/bin/python3 -u "$HERE/publish_hand_model_cloud.py" --ready-file "$RUNDIR/hand_ready" &
HAND_PID=$!
for ((i=0;i<200;i++)); do
  if [[ -f "$RUNDIR/ready" && -f "$RUNDIR/hand_ready" ]]; then break; fi
  if ! kill -0 "$HAND_PID" 2>/dev/null; then
    code=0; wait "$HAND_PID" || code=$?
    echo "手部模型点云启动失败，请检查上方错误。" >&2
    [[ "$code" != 0 ]] || code=1
    exit "$code"
  fi
  if ! kill -0 "$CLOUD_PID" 2>/dev/null; then
    code=0; wait "$CLOUD_PID" || code=$?
    echo "点云启动失败，未打开 RViz。请查看上方错误及 $LOG" >&2
    [[ "$code" != 0 ]] || code=1
    exit "$code"
  fi
  sleep 0.2
done
if [[ ! -f "$RUNDIR/ready" || ! -f "$RUNDIR/hand_ready" ]]; then echo "等待首帧超时，退出并释放相机。" >&2; exit 1; fi
PROJECTION_SCRIPT="${RVIZ_PROJECTION_SCRIPT:-/home/liberai/Liberai_Deploy_astra/astra/live_projections.py}"
if [[ -f "$PROJECTION_SCRIPT" ]]; then
  /usr/bin/python3 -u "$PROJECTION_SCRIPT" &
  PROJECTION_PID=$!
  RVIZ_CONFIG="${RVIZ_CONFIG:-/home/liberai/Liberai_Deploy_astra/config/astra/version6_pointcloud.rviz}"
fi
/opt/ros/jazzy/lib/rviz2/rviz2 -d "${RVIZ_CONFIG:-$HERE/cloud_and_robot.rviz}" --ros-args -r __node:=leo_cloud_robot_overlay &
RVIZ_PID=$!
PIDS=("$CLOUD_PID" "$HAND_PID" "$RVIZ_PID")
[[ -z "$PROJECTION_PID" ]] || PIDS+=("$PROJECTION_PID")
code=0; wait -n "${PIDS[@]}" || code=$?
exit "$code"
