# Direct Leo ImageWAM modes

This path runs one of the four existing Leo-700h unified-v2.2 ImageWAM
checkpoints directly. It does not start a high-level planner or perform an
automatic subtask transition.

The maintained deploy client converts the checkpoint's canonical 326-D action
to Leo arm pose6 and Wuji q20 commands. In particular, the supervised MANO90
fields are converted to q20 through the native MANO decoder; the inactive q20
slots in the canonical tensor are not used.

Numeric modes:

1. L2-only
2. L3+L2
3. L3+L2+Progress (detach)
4. L3+L2+Progress (backprop)

Use the same mode number for the GPU server and the robot client. The client
also checks the server checkpoint and adapter before publishing its action.

## Stream the four checkpoints to Baidu Cloud

Run this command on a local machine that can SSH to both `gpu03` and the Baidu
Cloud server:

```bash
bash High_level/RealRobot/wam_direct/transfer_wam_700h_to_baidu.sh
```

The local machine only relays the byte stream and does not store checkpoint
data. Interrupted uploads remain as `.part` files on Baidu Cloud and resume on
the next invocation. Every completed file is checked with SHA256 before its
`.part` suffix is removed.

## 1. Start one policy server on gpu03

```bash
bash /share-2/code/wzn/vlm_pretrain/High_level/RealRobot/wam_direct/serve_wam_mode.sh \
  --mode 2
```

Only one mode needs to be resident on the GPU. Stop that process before serving
another mode on the same port.

## 2. Start the tunnel on Leo

```bash
cd /home/liberai/wzn/vla_inference/wam_direct
bash tunnel_wam.sh start
bash tunnel_wam.sh status
```

## 3. Run direct WAM

Mode 1 takes an L2 prompt:

```bash
bash run_wam.sh \
  --mode 1 \
  --prompt "pick up the banana" \
  --chunk_size 32
```

Modes 2, 3, and 4 take the exact combined training format:

```bash
bash run_wam.sh \
  --mode 2 \
  --prompt "Task: Pick up the banana and place the banana in the bowl. Subtask: pick up the banana" \
  --chunk_size 32
```

Add `--record` to save a rollout under
`/home/liberai/wzn/rollout_records`. Use `--video-name` to choose the file
name; `.mp4` is appended when omitted:

```bash
bash run_wam.sh \
  --mode 2 \
  --prompt "Task: Pick up the banana and place the banana in the bowl. Subtask: pick up the banana" \
  --chunk_size 32 \
  --record \
  --video-name banana_to_bowl_01
```

This creates `banana_to_bowl_01.mp4` and
`banana_to_bowl_01.events.jsonl`. The event file records rollout start/end and
every manual subtask switch with wall-clock and rollout-relative timestamps.

Press Enter once after the robot reaches its configured startup pose. During
rollout, press `P`; after the current action chunk finishes, paste only the next
subtask and press Enter. The L3 task remains fixed and inference resumes without
resetting or reconnecting. Press `Ctrl+C` to exit the rollout.
