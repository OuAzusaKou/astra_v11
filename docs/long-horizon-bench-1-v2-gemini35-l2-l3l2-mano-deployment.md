# Long Horizon Bench 1 v2 L3 / Gemini 3.5 L2 / L3+L2 ImageWAM MANO 真机评测

模型：

- L3：`job-a468p4c3uw5c` / `imagewam-longhorizon1-v2-mano-joint-h32-368x320-bs512-60k-1n-b30z-20260901-031831` / `step_060000.pt`
- L2：`job-8u5j5wjnapdt` / `imagewam-longhorizon1-v2-mano-joint-gemini35-l2-h32-368x320-bs512-60k-1n-b30z-20260904-184514` / `step_060000.pt`
- L3+L2：`job-0sa1xsk2xjnz` / `imagewam-longhorizon1-v2-mano-joint-gemini35-l3l2-h32-368x320-bs512-60k-1n-b30z-20260904-184523` / `step_060000.pt`

共 4 个任务布局 × 6 种操作顺序 = 24 个 setting。每个 setting 测 3 个 trial；每个 trial 严格依次评测 L3、L2、L3+L2，共 216 次。

## 1. 同步代码

```bash
SSH_CONFIG_PATH="${HOME}/.ssh/config"
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
touch "${SSH_CONFIG_PATH}"
chmod 600 "${SSH_CONFIG_PATH}"

if ! grep -Eq '^[[:space:]]*Host[[:space:]]+bce_dev01([[:space:]]|$)' "${SSH_CONFIG_PATH}"; then
  SSH_CONFIG_TMP="$(mktemp)"
  {
    printf '%s\n' \
      'Host bce_dev01' \
      '  HostName 180.76.149.31' \
      '  User root' \
      '  Port 5338' \
      ''
    sed -n '1,$p' "${SSH_CONFIG_PATH}"
  } > "${SSH_CONFIG_TMP}"
  install -m 600 "${SSH_CONFIG_TMP}" "${SSH_CONFIG_PATH}"
  rm -f "${SSH_CONFIG_TMP}"
fi

ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=accept-new \
  bce_dev01 'echo SERVER_SSH_OK'

DEPLOY_ROOT="${HOME}/bangguo/Liberai_Deploy"
mkdir -p "${DEPLOY_ROOT}"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='records/' \
  bce_dev01:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${DEPLOY_ROOT}/"

cd "${DEPLOY_ROOT}"
source "${DEPLOY_ROOT}/ops/long_horizon_bench_1_v2_gemini35_mano/alias.sh"
python3 scripts/build_native_mano_ik.py --force
python3 - <<'PY'
from mano.native.runtime import native_extension_path

path = native_extension_path()
if path is None:
    raise SystemExit("ERROR: MANO native extension is unavailable")
print(f"MANO native extension: {path}")
PY
```

## 2. 评测指令

按键规则：

- Client 显示 `按下回车以继续` 时，检查现场后按一次 `Enter` 开始推理；这一次不会切换 L2。
- L3 only 全程固定使用当前 combination 的 L3，不切换 instruction；任务完成后直接按 `Ctrl-C`。
- 第 1 条 L2 会自动生效。从第 2 条起，只有观察到上一条 L2 描述的动作已经完成后，才按一次 `Enter`。
- 每次按键后，等待终端显示 `[imagewam] instruction i/N` 再继续；不要提前按或连续按。
- 最后一条 L2 完成后按 `Ctrl-C` 停止评测，不再按 `Enter`。

评测与记录规则：

- 每个 Combination 的顺序固定为：Trial 1 的 L3 → L2 → L3+L2；Trial 2 的 L3 → L2 → L3+L2；Trial 3 的 L3 → L2 → L3+L2。不要先连续测完某一个模型。
- 每条命令最后的 `1`、`2`、`3` 是 Trial 编号，不可省略。
- Client 会在开始推理时保存首帧图片，并打印 `[start-image] saved -> ...`。路径中同时包含 setting、trial 和模型；评测结束后按相同三项把图片手动复制到结果表的“首帧图片”列。
- “完成子任务数”按 `1/1/1/...` 填写，每一位依次代表一个子任务是否完成；写到最后一个未完成的子任务为止。例如前 6 项完成、第 7 项未完成，填写 `1/1/1/1/1/1/0`。

### 清理桌面 1

#### Combination 1

L3：餐巾纸摆到餐桌左侧，把所有餐具叠放在右侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000028.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000028.mp4)

L2 / L3+L2 切分与切换时机：

1. 把纸巾盒放到桌面左侧（启动推理后自动生效，不按切换键）
2. 拿起杯子（观察到上一条“把纸巾盒放到桌面左侧”完成后，按一次 Enter；看到 instruction 2/13 后进入本条）
3. 把杯子放到碗里（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 3/13 后进入本条）
4. 拿起盘子（观察到上一条“把杯子放到碗里”完成后，按一次 Enter；看到 instruction 4/13 后进入本条）
5. 把盘子放到桌面右侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 5/13 后进入本条）
6. 拿起碗（观察到上一条“把盘子放到桌面右侧”完成后，按一次 Enter；看到 instruction 6/13 后进入本条）
7. 把碗放到盘子上（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 7/13 后进入本条）
8. 拿起橘子（观察到上一条“把碗放到盘子上”完成后，按一次 Enter；看到 instruction 8/13 后进入本条）
9. 把橘子放到盘子里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 9/13 后进入本条）
10. 拿起揉皱的纸巾（观察到上一条“把橘子放到盘子里”完成后，按一次 Enter；看到 instruction 10/13 后进入本条）
11. 把揉皱的纸巾放到垃圾桶里（观察到上一条“拿起揉皱的纸巾”完成后，按一次 Enter；看到 instruction 11/13 后进入本条）
12. 拿起面包（观察到上一条“把揉皱的纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 12/13 后进入本条）
13. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 13/13 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c1 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c1 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c1 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c1 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c1 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c1 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c1 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c1 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c1 3
```

#### Combination 2

L3：把所有餐具叠放在右侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧，餐巾纸摆到餐桌左侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000053.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000053.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起碗（启动推理后自动生效，不按切换键）
2. 把碗放到桌面右侧（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 2/14 后进入本条）
3. 拿起盘子（观察到上一条“把碗放到桌面右侧”完成后，按一次 Enter；看到 instruction 3/14 后进入本条）
4. 把盘子放到碗上（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 4/14 后进入本条）
5. 拿起杯子（观察到上一条“把盘子放到碗上”完成后，按一次 Enter；看到 instruction 5/14 后进入本条）
6. 把杯子放到桌面右侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 6/14 后进入本条）
7. 拿起橘子（观察到上一条“把杯子放到桌面右侧”完成后，按一次 Enter；看到 instruction 7/14 后进入本条）
8. 把橘子放到盘子里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 8/14 后进入本条）
9. 拿起面包（观察到上一条“把橘子放到盘子里”完成后，按一次 Enter；看到 instruction 9/14 后进入本条）
10. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 10/14 后进入本条）
11. 拿起废纸（观察到上一条“把面包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 11/14 后进入本条）
12. 把废纸放到垃圾桶里（观察到上一条“拿起废纸”完成后，按一次 Enter；看到 instruction 12/14 后进入本条）
13. 拿起纸巾盒（观察到上一条“把废纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 13/14 后进入本条）
14. 把纸巾盒放到桌面左侧（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 14/14 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c2 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c2 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c2 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c2 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c2 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c2 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c2 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c2 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c2 3
```

#### Combination 3

L3：把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，餐巾纸摆到餐桌左侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶。

训练示范视频（主视角，评测前先完整观看）：[episode_000123.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000123.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起盘子（启动推理后自动生效，不按切换键）
2. 把盘子放到桌面右侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起小碟子（观察到上一条“把盘子放到桌面右侧”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把小碟子放到盘子上（观察到上一条“拿起小碟子”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起杯子（观察到上一条“把小碟子放到盘子上”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把杯子放到小碟子上（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起垃圾桶（观察到上一条“把杯子放到小碟子上”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起纸巾盒（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把纸巾盒放到桌面左侧（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起橘子（观察到上一条“把纸巾盒放到桌面左侧”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把橘子放到水果盘里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起面包（观察到上一条“把橘子放到水果盘里”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起揉皱的纸（观察到上一条“把面包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把揉皱的纸放到垃圾桶里（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c3 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c3 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c3 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c3 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c3 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c3 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c3 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c3 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c3 3
```

#### Combination 4

L3：剩余食物放在水果盘中，把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，把餐余垃圾扔进垃圾桶，餐巾纸摆到餐桌左侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000156.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000156.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起碗（启动推理后自动生效，不按切换键）
2. 把碗放到盘子上（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 2/14 后进入本条）
3. 拿起面包（观察到上一条“把碗放到盘子上”完成后，按一次 Enter；看到 instruction 3/14 后进入本条）
4. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 4/14 后进入本条）
5. 拿起红杯子（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 5/14 后进入本条）
6. 把红杯子放到盘子右侧（观察到上一条“拿起红杯子”完成后，按一次 Enter；看到 instruction 6/14 后进入本条）
7. 拿起垃圾桶（观察到上一条“把红杯子放到盘子右侧”完成后，按一次 Enter；看到 instruction 7/14 后进入本条）
8. 把垃圾桶放到盘子后方（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 8/14 后进入本条）
9. 拿起废纸（观察到上一条“把垃圾桶放到盘子后方”完成后，按一次 Enter；看到 instruction 9/14 后进入本条）
10. 把废纸放到垃圾桶里（观察到上一条“拿起废纸”完成后，按一次 Enter；看到 instruction 10/14 后进入本条）
11. 拿起橘子（观察到上一条“把废纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 11/14 后进入本条）
12. 把橘子放到垃圾桶里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 12/14 后进入本条）
13. 拿起纸巾盒（观察到上一条“把橘子放到垃圾桶里”完成后，按一次 Enter；看到 instruction 13/14 后进入本条）
14. 把纸巾盒放到餐垫左侧（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 14/14 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c4 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c4 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c4 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c4 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c4 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c4 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c4 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c4 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c4 3
```

#### Combination 5

L3：把餐余垃圾扔进垃圾桶，餐巾纸摆到餐桌左侧，把所有餐具叠放在右侧，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000207.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000207.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起纸屑（启动推理后自动生效，不按切换键）
2. 把纸屑放到垃圾桶里（观察到上一条“拿起纸屑”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起橘子皮（观察到上一条“把纸屑放到垃圾桶里”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起纸巾盒（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把纸巾盒放到桌面左侧（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起白盘子（观察到上一条“把纸巾盒放到桌面左侧”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把白盘子放到桌面右侧（观察到上一条“拿起白盘子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起小盘子（观察到上一条“把白盘子放到桌面右侧”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把小盘子放到白盘子上（观察到上一条“拿起小盘子”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起杯子（观察到上一条“把小盘子放到白盘子上”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把杯子放到小盘子上（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起面包（观察到上一条“把杯子放到小盘子上”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起垃圾桶（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c5 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c5 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c5 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c5 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c5 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c5 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c5 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c5 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c5 3
```

#### Combination 6

L3：把餐余垃圾扔进垃圾桶，把所有餐具叠放在右侧，垃圾桶摆到餐桌右侧，剩余食物放在水果盘中，餐巾纸摆到餐桌左侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000229.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000229.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起橘子皮，然后把橘子皮放到垃圾桶里（启动推理后自动生效，不按切换键）
2. 拿起揉皱的纸巾（观察到上一条“拿起橘子皮，然后把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 2/15 后进入本条）
3. 把揉皱的纸巾放到垃圾桶里（观察到上一条“拿起揉皱的纸巾”完成后，按一次 Enter；看到 instruction 3/15 后进入本条）
4. 拿起盘子（观察到上一条“把揉皱的纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/15 后进入本条）
5. 把盘子放到餐垫右侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 5/15 后进入本条）
6. 拿起碗（观察到上一条“把盘子放到餐垫右侧”完成后，按一次 Enter；看到 instruction 6/15 后进入本条）
7. 把碗放到盘子上（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 7/15 后进入本条）
8. 拿起杯子（观察到上一条“把碗放到盘子上”完成后，按一次 Enter；看到 instruction 8/15 后进入本条）
9. 把杯子放到盘子右侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 9/15 后进入本条）
10. 拿起垃圾桶（观察到上一条“把杯子放到盘子右侧”完成后，按一次 Enter；看到 instruction 10/15 后进入本条）
11. 把垃圾桶放到桌面上（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 11/15 后进入本条）
12. 拿起面包（观察到上一条“把垃圾桶放到桌面上”完成后，按一次 Enter；看到 instruction 12/15 后进入本条）
13. 把面包放到碗里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 13/15 后进入本条）
14. 拿起纸巾包（观察到上一条“把面包放到碗里”完成后，按一次 Enter；看到 instruction 14/15 后进入本条）
15. 把纸巾包放到桌面上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 15/15 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c6 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c6 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c6 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c6 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c6 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c6 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable1_c6 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable1_c6 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable1_c6 3
```

### 清理桌面 3

#### Combination 1

L3：餐巾纸摆到餐垫上边，把所有餐具叠放在左侧，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶，垃圾桶摆到餐桌右侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000619.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000619.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起纸巾包（启动推理后自动生效，不按切换键）
2. 把纸巾包放到餐垫上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起盘子（观察到上一条“把纸巾包放到餐垫上”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把盘子放到桌面左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起小碗（观察到上一条“把盘子放到桌面左侧”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把小碗放到盘子上（观察到上一条“拿起小碗”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起杯子（观察到上一条“把小碗放到盘子上”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把杯子放到碗里（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起面包（观察到上一条“把杯子放到碗里”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把面包放到水果盘里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起揉皱的纸巾（观察到上一条“把面包放到水果盘里”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把揉皱的纸巾扔进垃圾桶（观察到上一条“拿起揉皱的纸巾”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起橘子（观察到上一条“把揉皱的纸巾扔进垃圾桶”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把橘子扔进垃圾桶（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起垃圾桶（观察到上一条“把橘子扔进垃圾桶”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c1 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c1 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c1 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c1 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c1 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c1 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c1 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c1 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c1 3
```

#### Combination 2

L3：把所有餐具叠放在左侧，把餐余垃圾扔进垃圾桶，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧，餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000949.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000949.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起盘子，然后把盘子放到桌面左侧（启动推理后自动生效，不按切换键）
2. 拿起小碗（观察到上一条“拿起盘子，然后把盘子放到桌面左侧”完成后，按一次 Enter；看到 instruction 2/15 后进入本条）
3. 把小碗放到盘子上（观察到上一条“拿起小碗”完成后，按一次 Enter；看到 instruction 3/15 后进入本条）
4. 拿起杯子（观察到上一条“把小碗放到盘子上”完成后，按一次 Enter；看到 instruction 4/15 后进入本条）
5. 把杯子放到桌面左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 5/15 后进入本条）
6. 拿起橘子皮（观察到上一条“把杯子放到桌面左侧”完成后，按一次 Enter；看到 instruction 6/15 后进入本条）
7. 把橘子皮扔进垃圾桶（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 7/15 后进入本条）
8. 拿起揉皱的纸（观察到上一条“把橘子皮扔进垃圾桶”完成后，按一次 Enter；看到 instruction 8/15 后进入本条）
9. 把揉皱的纸扔进垃圾桶（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 9/15 后进入本条）
10. 拿起面包（观察到上一条“把揉皱的纸扔进垃圾桶”完成后，按一次 Enter；看到 instruction 10/15 后进入本条）
11. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 11/15 后进入本条）
12. 拿起垃圾桶（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 12/15 后进入本条）
13. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 13/15 后进入本条）
14. 拿起纸巾包（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 14/15 后进入本条）
15. 把纸巾包放到餐垫上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 15/15 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c2 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c2 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c2 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c2 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c2 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c2 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c2 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c2 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c2 3
```

#### Combination 3

L3：把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，餐巾纸摆到餐垫上边，剩余食物放在水果盘中，把餐余垃圾扔进垃圾桶。

训练示范视频（主视角，评测前先完整观看）：[episode_000989.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000989.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起盘子（启动推理后自动生效，不按切换键）
2. 把盘子放到桌面左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起碗（观察到上一条“把盘子放到桌面左侧”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把碗放到盘子上（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起杯子（观察到上一条“把碗放到盘子上”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把杯子放到碗里（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起垃圾桶（观察到上一条“把杯子放到碗里”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起纸巾包（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把纸巾包放到餐垫上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起橘子（观察到上一条“把纸巾包放到餐垫上”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把橘子放到碗里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起揉皱的纸（观察到上一条“把橘子放到碗里”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把揉皱的纸放到垃圾桶里（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起面包（观察到上一条“把揉皱的纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c3 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c3 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c3 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c3 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c3 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c3 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c3 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c3 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c3 3
```

#### Combination 4

L3：剩余食物放在水果盘中，把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，把餐余垃圾扔进垃圾桶，餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_001054.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-001/observation.images.image_ego_0/episode_001054.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起橘子（启动推理后自动生效，不按切换键）
2. 把橘子放到水果盘里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起盘子（观察到上一条“把橘子放到水果盘里”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把盘子放到水果盘上（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起碗（观察到上一条“把盘子放到水果盘上”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把碗放到盘子上（观察到上一条“拿起碗”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起杯子（观察到上一条“把碗放到盘子上”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把杯子放到碗上（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起垃圾桶（观察到上一条“把杯子放到碗上”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起揉皱的纸巾（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把揉皱的纸巾放到垃圾桶里（观察到上一条“拿起揉皱的纸巾”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起面包（观察到上一条“把揉皱的纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起纸巾包（观察到上一条“把面包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把纸巾包放到餐垫上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c4 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c4 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c4 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c4 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c4 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c4 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c4 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c4 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c4 3
```

#### Combination 5

L3：把餐余垃圾扔进垃圾桶，餐巾纸摆到餐垫上边，把所有餐具叠放在左侧，剩余食物放在水果盘中，垃圾桶摆到餐桌右侧。

训练示范视频（主视角，评测前先完整观看）：[episode_001074.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-001/observation.images.image_ego_0/episode_001074.mp4)

L2 / L3+L2 切分与切换时机：

1. 把橘子皮扔进垃圾桶（启动推理后自动生效，不按切换键）
2. 拿起揉皱的纸巾（观察到上一条“把橘子皮扔进垃圾桶”完成后，按一次 Enter；看到 instruction 2/13 后进入本条）
3. 把揉皱的纸巾扔进垃圾桶（观察到上一条“拿起揉皱的纸巾”完成后，按一次 Enter；看到 instruction 3/13 后进入本条）
4. 拿起纸巾包（观察到上一条“把揉皱的纸巾扔进垃圾桶”完成后，按一次 Enter；看到 instruction 4/13 后进入本条）
5. 把纸巾包放到餐垫上（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 5/13 后进入本条）
6. 拿起白盘子（观察到上一条“把纸巾包放到餐垫上”完成后，按一次 Enter；看到 instruction 6/13 后进入本条）
7. 把白盘子叠放到碗上（观察到上一条“拿起白盘子”完成后，按一次 Enter；看到 instruction 7/13 后进入本条）
8. 拿起长方形纸碗（观察到上一条“把白盘子叠放到碗上”完成后，按一次 Enter；看到 instruction 8/13 后进入本条）
9. 把长方形纸碗叠放到白盘子上（观察到上一条“拿起长方形纸碗”完成后，按一次 Enter；看到 instruction 9/13 后进入本条）
10. 拿起面包（观察到上一条“把长方形纸碗叠放到白盘子上”完成后，按一次 Enter；看到 instruction 10/13 后进入本条）
11. 把面包放到水果盘里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 11/13 后进入本条）
12. 拿起垃圾桶（观察到上一条“把面包放到水果盘里”完成后，按一次 Enter；看到 instruction 12/13 后进入本条）
13. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 13/13 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c5 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c5 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c5 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c5 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c5 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c5 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c5 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c5 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c5 3
```

#### Combination 6

L3：把餐余垃圾扔进垃圾桶，把所有餐具叠放在左侧，垃圾桶摆到餐桌右侧，剩余食物放在水果盘中，餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_001134.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-001/observation.images.image_ego_0/episode_001134.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起橘子并扔进垃圾桶（启动推理后自动生效，不按切换键）
2. 拿起揉皱的纸巾并扔进垃圾桶（观察到上一条“拿起橘子并扔进垃圾桶”完成后，按一次 Enter；看到 instruction 2/7 后进入本条）
3. 把白盘子放到桌面左侧（观察到上一条“拿起揉皱的纸巾并扔进垃圾桶”完成后，按一次 Enter；看到 instruction 3/7 后进入本条）
4. 把碗放到白盘子上（观察到上一条“把白盘子放到桌面左侧”完成后，按一次 Enter；看到 instruction 4/7 后进入本条）
5. 把垃圾桶放到桌面右侧（观察到上一条“把碗放到白盘子上”完成后，按一次 Enter；看到 instruction 5/7 后进入本条）
6. 把面包放到碗里（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 6/7 后进入本条）
7. 把纸巾包放到餐垫前方（观察到上一条“把面包放到碗里”完成后，按一次 Enter；看到 instruction 7/7 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c6 1
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c6 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c6 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c6 2
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c6 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c6 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_cleartable3_c6 3
long_horizon_bench_1_v2_gemini35_eval_l2_cleartable3_c6 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_cleartable3_c6 3
```

### 准备早餐 1

#### Combination 1

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边，把水果盘放到餐垫右边。

训练示范视频（主视角，评测前先完整观看）：[episode_000308.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000308.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起垃圾桶（启动推理后自动生效，不按切换键）
2. 把垃圾桶放到餐垫左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起橘子皮（观察到上一条“把垃圾桶放到餐垫左侧”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起揉皱的纸（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把揉皱的纸放到垃圾桶里（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起盘子（观察到上一条“把揉皱的纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把盘子放到餐垫中央（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起面包（观察到上一条“把盘子放到餐垫中央”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起杯子（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起纸巾包（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把纸巾包放到餐垫前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起水果盘（观察到上一条“把纸巾包放到餐垫前方”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c1 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c1 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c1 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c1 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c1 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c1 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c1 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c1 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c1 3
```

#### Combination 2

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把水果盘放到餐垫右边，把餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000336.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000336.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起垃圾桶（启动推理后自动生效，不按切换键）
2. 把垃圾桶放到桌面左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起纸（观察到上一条“把垃圾桶放到桌面左侧”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把纸放到垃圾桶里（观察到上一条“拿起纸”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起橘子皮（观察到上一条“把纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起盘子（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把盘子放到餐垫中央（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起面包（观察到上一条“把盘子放到餐垫中央”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起杯子（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起水果盘（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起纸巾包（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把纸巾包放到餐垫前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c2 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c2 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c2 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c2 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c2 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c2 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c2 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c2 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c2 3
```

#### Combination 3

L3：把垃圾桶放在桌面左侧，清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000376.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000376.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起垃圾桶（启动推理后自动生效，不按切换键）
2. 把垃圾桶放到桌面左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起揉皱的纸（观察到上一条“把垃圾桶放到桌面左侧”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把揉皱的纸放到垃圾桶里（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起橘子皮（观察到上一条“把揉皱的纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起水果盘（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起盘子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把盘子放到餐垫上（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起面包（观察到上一条“把盘子放到餐垫上”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起杯子（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起纸巾包（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把纸巾包放到餐垫前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c3 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c3 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c3 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c3 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c3 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c3 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c3 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c3 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c3 3
```

#### Combination 4

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，将食物放到餐盘上，把水杯摆放到餐垫左边，把餐巾纸摆到餐垫上边，把垃圾桶放在桌面左侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000442.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000442.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起揉皱的纸巾，然后把揉皱的纸巾放到垃圾桶里（启动推理后自动生效，不按切换键）
2. 拿起橘子皮（观察到上一条“拿起揉皱的纸巾，然后把揉皱的纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 2/15 后进入本条）
3. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 3/15 后进入本条）
4. 拿起水果盘（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/15 后进入本条）
5. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 5/15 后进入本条）
6. 拿起餐盘（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 6/15 后进入本条）
7. 把餐盘放到餐垫上（观察到上一条“拿起餐盘”完成后，按一次 Enter；看到 instruction 7/15 后进入本条）
8. 拿起面包（观察到上一条“把餐盘放到餐垫上”完成后，按一次 Enter；看到 instruction 8/15 后进入本条）
9. 把面包放到餐盘上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 9/15 后进入本条）
10. 拿起杯子（观察到上一条“把面包放到餐盘上”完成后，按一次 Enter；看到 instruction 10/15 后进入本条）
11. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 11/15 后进入本条）
12. 拿起纸巾包（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 12/15 后进入本条）
13. 把纸巾包放到餐垫前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 13/15 后进入本条）
14. 拿起垃圾桶（观察到上一条“把纸巾包放到餐垫前方”完成后，按一次 Enter；看到 instruction 14/15 后进入本条）
15. 把垃圾桶放到餐垫左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 15/15 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c4 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c4 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c4 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c4 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c4 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c4 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c4 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c4 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c4 3
```

#### Combination 5

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫中间，把水杯摆放到餐垫左边，将食物放到餐盘上，把垃圾桶放在桌面左侧，把餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000515.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000515.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起揉皱的纸巾，然后把揉皱的纸巾放到垃圾桶里（启动推理后自动生效，不按切换键）
2. 拿起牛角包（观察到上一条“拿起揉皱的纸巾，然后把揉皱的纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 2/15 后进入本条）
3. 把牛角包放到垃圾桶里（观察到上一条“拿起牛角包”完成后，按一次 Enter；看到 instruction 3/15 后进入本条）
4. 拿起水果盘（观察到上一条“把牛角包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/15 后进入本条）
5. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 5/15 后进入本条）
6. 拿起盘子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 6/15 后进入本条）
7. 把盘子放到餐垫中央（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 7/15 后进入本条）
8. 拿起杯子（观察到上一条“把盘子放到餐垫中央”完成后，按一次 Enter；看到 instruction 8/15 后进入本条）
9. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 9/15 后进入本条）
10. 拿起橘子（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 10/15 后进入本条）
11. 把橘子放到盘子上（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 11/15 后进入本条）
12. 拿起垃圾桶（观察到上一条“把橘子放到盘子上”完成后，按一次 Enter；看到 instruction 12/15 后进入本条）
13. 把垃圾桶放到餐垫左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 13/15 后进入本条）
14. 拿起纸巾盒（观察到上一条“把垃圾桶放到餐垫左侧”完成后，按一次 Enter；看到 instruction 14/15 后进入本条）
15. 把纸巾盒放到餐垫前方（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 15/15 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c5 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c5 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c5 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c5 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c5 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c5 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c5 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c5 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c5 3
```

#### Combination 6

L3：清除餐桌上的垃圾，把垃圾桶放在桌面左侧，把餐盘摆放到餐垫中间，把水果盘放到餐垫右边，把水杯摆放到餐垫左边，将食物放到餐盘上，把餐巾纸摆到餐垫上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000588.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000588.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起纸屑（启动推理后自动生效，不按切换键）
2. 把纸屑放到垃圾桶里（观察到上一条“拿起纸屑”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起果皮（观察到上一条“把纸屑放到垃圾桶里”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把果皮放到垃圾桶里（观察到上一条“拿起果皮”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起垃圾桶（观察到上一条“把果皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把垃圾桶放到桌面左侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起盘子（观察到上一条“把垃圾桶放到桌面左侧”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把盘子放到餐垫中央（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起水果盘（观察到上一条“把盘子放到餐垫中央”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起杯子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把杯子放到餐垫左侧（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起面包（观察到上一条“把杯子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起纸巾包（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把纸巾包放到餐垫前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c6 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c6 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c6 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c6 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c6 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c6 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast1_c6 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast1_c6 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast1_c6 3
```

### 准备早餐 4

#### Combination 1

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边，把水果盘放到餐垫右边。

训练示范视频（主视角，评测前先完整观看）：[episode_000661.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000661.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起垃圾桶，然后把垃圾桶放到餐垫右侧（启动推理后自动生效，不按切换键）
2. 拿起揉皱的厨房纸（观察到上一条“拿起垃圾桶，然后把垃圾桶放到餐垫右侧”完成后，按一次 Enter；看到 instruction 2/14 后进入本条）
3. 把揉皱的厨房纸放到垃圾桶里（观察到上一条“拿起揉皱的厨房纸”完成后，按一次 Enter；看到 instruction 3/14 后进入本条）
4. 拿起牛角包（观察到上一条“把揉皱的厨房纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/14 后进入本条）
5. 把牛角包放到垃圾桶里（观察到上一条“拿起牛角包”完成后，按一次 Enter；看到 instruction 5/14 后进入本条）
6. 拿起盘子（观察到上一条“把牛角包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 6/14 后进入本条）
7. 把盘子放到餐垫左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 7/14 后进入本条）
8. 拿起小番茄（观察到上一条“把盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 8/14 后进入本条）
9. 把小番茄放到盘子上（观察到上一条“拿起小番茄”完成后，按一次 Enter；看到 instruction 9/14 后进入本条）
10. 拿起杯子（观察到上一条“把小番茄放到盘子上”完成后，按一次 Enter；看到 instruction 10/14 后进入本条）
11. 把杯子放到盘子后方（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 11/14 后进入本条）
12. 拿起纸巾包（观察到上一条“把杯子放到盘子后方”完成后，按一次 Enter；看到 instruction 12/14 后进入本条）
13. 把纸巾包放到水果盘后方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 13/14 后进入本条）
14. 拿起水果盘，然后把水果盘放到餐垫右侧（观察到上一条“把纸巾包放到水果盘后方”完成后，按一次 Enter；看到 instruction 14/14 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c1 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c1 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c1 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c1 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c1 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c1 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c1 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c1 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c1 3
```

#### Combination 2

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把水果盘放到餐垫右边，把餐巾纸摆到餐垫右上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000710.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000710.mp4)

L2 / L3+L2 切分与切换时机：

1. 把垃圾桶放到餐垫右侧（启动推理后自动生效，不按切换键）
2. 拿起橘子皮（观察到上一条“把垃圾桶放到餐垫右侧”完成后，按一次 Enter；看到 instruction 2/14 后进入本条）
3. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 3/14 后进入本条）
4. 拿起纸巾（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/14 后进入本条）
5. 把纸巾放到垃圾桶里（观察到上一条“拿起纸巾”完成后，按一次 Enter；看到 instruction 5/14 后进入本条）
6. 拿起盘子（观察到上一条“把纸巾放到垃圾桶里”完成后，按一次 Enter；看到 instruction 6/14 后进入本条）
7. 把盘子放到餐垫左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 7/14 后进入本条）
8. 拿起面包（观察到上一条“把盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 8/14 后进入本条）
9. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 9/14 后进入本条）
10. 拿起杯子，然后把杯子放到盘子前方（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 10/14 后进入本条）
11. 拿起水果盘（观察到上一条“拿起杯子，然后把杯子放到盘子前方”完成后，按一次 Enter；看到 instruction 11/14 后进入本条）
12. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 12/14 后进入本条）
13. 拿起纸巾包（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 13/14 后进入本条）
14. 把纸巾包放到水果盘后方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 14/14 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c2 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c2 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c2 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c2 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c2 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c2 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c2 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c2 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c2 3
```

#### Combination 3

L3：把垃圾桶放在桌面右侧，清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000762.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000762.mp4)

L2 / L3+L2 切分与切换时机：

1. 把垃圾桶放到桌面右侧（启动推理后自动生效，不按切换键）
2. 拿起橘子皮（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 2/11 后进入本条）
3. 把橘子皮放到垃圾桶里（观察到上一条“拿起橘子皮”完成后，按一次 Enter；看到 instruction 3/11 后进入本条）
4. 拿起塑料包装纸（观察到上一条“把橘子皮放到垃圾桶里”完成后，按一次 Enter；看到 instruction 4/11 后进入本条）
5. 把塑料包装纸放到垃圾桶里（观察到上一条“拿起塑料包装纸”完成后，按一次 Enter；看到 instruction 5/11 后进入本条）
6. 把水果盘放到餐垫右侧（观察到上一条“把塑料包装纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 6/11 后进入本条）
7. 拿起盘子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 7/11 后进入本条）
8. 把盘子放到餐垫左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/11 后进入本条）
9. 把面包放到盘子上（观察到上一条“把盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 9/11 后进入本条）
10. 把杯子放到盘子后方（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 10/11 后进入本条）
11. 把纸巾包放到水果盘后方（观察到上一条“把杯子放到盘子后方”完成后，按一次 Enter；看到 instruction 11/11 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c3 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c3 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c3 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c3 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c3 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c3 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c3 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c3 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c3 3
```

#### Combination 4

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，将食物放到餐盘上，把水杯摆放到餐垫左上边，把餐巾纸摆到餐垫右上边，把垃圾桶放在桌面右侧。

训练示范视频（主视角，评测前先完整观看）：[episode_000788.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000788.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起揉皱的餐巾纸（启动推理后自动生效，不按切换键）
2. 把揉皱的餐巾纸放到垃圾桶里（观察到上一条“拿起揉皱的餐巾纸”完成后，按一次 Enter；看到 instruction 2/12 后进入本条）
3. 拿起面包（观察到上一条“把揉皱的餐巾纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 3/12 后进入本条）
4. 把面包放到垃圾桶里（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 4/12 后进入本条）
5. 拿起白盘子（观察到上一条“把面包放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/12 后进入本条）
6. 把白盘子放到餐垫左侧（观察到上一条“拿起白盘子”完成后，按一次 Enter；看到 instruction 6/12 后进入本条）
7. 拿起左侧的橘子（观察到上一条“把白盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 7/12 后进入本条）
8. 把左侧的橘子放到白盘子上（观察到上一条“拿起左侧的橘子”完成后，按一次 Enter；看到 instruction 8/12 后进入本条）
9. 拿起纸杯（观察到上一条“把左侧的橘子放到白盘子上”完成后，按一次 Enter；看到 instruction 9/12 后进入本条）
10. 把纸杯放到餐垫前方（观察到上一条“拿起纸杯”完成后，按一次 Enter；看到 instruction 10/12 后进入本条）
11. 拿起纸巾包，然后把纸巾包放到椭圆盘前方（观察到上一条“把纸杯放到餐垫前方”完成后，按一次 Enter；看到 instruction 11/12 后进入本条）
12. 拿起垃圾桶，然后把垃圾桶放到桌面右侧（观察到上一条“拿起纸巾包，然后把纸巾包放到椭圆盘前方”完成后，按一次 Enter；看到 instruction 12/12 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c4 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c4 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c4 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c4 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c4 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c4 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c4 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c4 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c4 3
```

#### Combination 5

L3：清除餐桌上的垃圾，把水果盘放到餐垫右边，把餐盘摆放到餐垫左边，把水杯摆放到餐垫左上边，将食物放到餐盘上，把垃圾桶放在桌面右侧，把餐巾纸摆到餐垫右上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000816.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000816.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起揉成团的纸（启动推理后自动生效，不按切换键）
2. 把揉成团的纸放到垃圾桶里（观察到上一条“拿起揉成团的纸”完成后，按一次 Enter；看到 instruction 2/16 后进入本条）
3. 拿起橘子（观察到上一条“把揉成团的纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 3/16 后进入本条）
4. 把橘子放到垃圾桶里（观察到上一条“拿起橘子”完成后，按一次 Enter；看到 instruction 4/16 后进入本条）
5. 拿起水果盘（观察到上一条“把橘子放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/16 后进入本条）
6. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 6/16 后进入本条）
7. 拿起盘子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 7/16 后进入本条）
8. 把盘子放到餐垫左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/16 后进入本条）
9. 拿起杯子（观察到上一条“把盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 9/16 后进入本条）
10. 把杯子放到盘子前方（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 10/16 后进入本条）
11. 拿起面包（观察到上一条“把杯子放到盘子前方”完成后，按一次 Enter；看到 instruction 11/16 后进入本条）
12. 把面包放到盘子上（观察到上一条“拿起面包”完成后，按一次 Enter；看到 instruction 12/16 后进入本条）
13. 拿起垃圾桶（观察到上一条“把面包放到盘子上”完成后，按一次 Enter；看到 instruction 13/16 后进入本条）
14. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 14/16 后进入本条）
15. 拿起纸巾包（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 15/16 后进入本条）
16. 把纸巾包放到水果盘前方（观察到上一条“拿起纸巾包”完成后，按一次 Enter；看到 instruction 16/16 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c5 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c5 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c5 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c5 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c5 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c5 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c5 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c5 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c5 3
```

#### Combination 6

L3：清除餐桌上的垃圾，把垃圾桶放在桌面右侧，把餐盘摆放到餐垫左边，把水果盘放到餐垫右边，把水杯摆放到餐垫左上边，将食物放到餐盘上，把餐巾纸摆到餐垫右上边。

训练示范视频（主视角，评测前先完整观看）：[episode_000872.mp4](/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-teleop-data/posttrain/processed_lerobot/long_horizon_bench_1_v2/videos/chunk-000/observation.images.image_ego_0/episode_000872.mp4)

L2 / L3+L2 切分与切换时机：

1. 拿起揉皱的纸（启动推理后自动生效，不按切换键）
2. 把揉皱的纸放到垃圾桶里（观察到上一条“拿起揉皱的纸”完成后，按一次 Enter；看到 instruction 2/15 后进入本条）
3. 拿起塑料包装纸（观察到上一条“把揉皱的纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 3/15 后进入本条）
4. 把塑料包装纸放到垃圾桶里（观察到上一条“拿起塑料包装纸”完成后，按一次 Enter；看到 instruction 4/15 后进入本条）
5. 拿起垃圾桶（观察到上一条“把塑料包装纸放到垃圾桶里”完成后，按一次 Enter；看到 instruction 5/15 后进入本条）
6. 把垃圾桶放到桌面右侧（观察到上一条“拿起垃圾桶”完成后，按一次 Enter；看到 instruction 6/15 后进入本条）
7. 拿起盘子（观察到上一条“把垃圾桶放到桌面右侧”完成后，按一次 Enter；看到 instruction 7/15 后进入本条）
8. 把盘子放到餐垫左侧（观察到上一条“拿起盘子”完成后，按一次 Enter；看到 instruction 8/15 后进入本条）
9. 拿起水果盘（观察到上一条“把盘子放到餐垫左侧”完成后，按一次 Enter；看到 instruction 9/15 后进入本条）
10. 把水果盘放到餐垫右侧（观察到上一条“拿起水果盘”完成后，按一次 Enter；看到 instruction 10/15 后进入本条）
11. 拿起杯子（观察到上一条“把水果盘放到餐垫右侧”完成后，按一次 Enter；看到 instruction 11/15 后进入本条）
12. 把杯子放到盘子前方（观察到上一条“拿起杯子”完成后，按一次 Enter；看到 instruction 12/15 后进入本条）
13. 拿起面包，然后把面包放到盘子上（观察到上一条“把杯子放到盘子前方”完成后，按一次 Enter；看到 instruction 13/15 后进入本条）
14. 拿起纸巾盒（观察到上一条“拿起面包，然后把面包放到盘子上”完成后，按一次 Enter；看到 instruction 14/15 后进入本条）
15. 把纸巾盒放到水果盘右侧（观察到上一条“拿起纸巾盒”完成后，按一次 Enter；看到 instruction 15/15 后进入本条）

最后一条完成后按 `Ctrl-C` 停止评测。

本 Combination 的评测顺序（每条命令单独执行，严格从上到下）：

Trial 1：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c6 1
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c6 1
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c6 1
```

Trial 2：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c6 2
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c6 2
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c6 2
```

Trial 3：

```bash
long_horizon_bench_1_v2_eval_l3_preparebreakfast4_c6 3
long_horizon_bench_1_v2_gemini35_eval_l2_preparebreakfast4_c6 3
long_horizon_bench_1_v2_gemini35_eval_l3_l2_preparebreakfast4_c6 3
```

## 3. 结果记录

[打开独立飞书评测结果表](https://ucn81zkkano6.feishu.cn/sheets/IUnKsShjdhEwcNtcu7rcrv9Lnqc)
