# 0827 Bench1 WAM M4a / M4a baseline 真机评测

模型：

- M4a：WAM image10/action10
- M4a baseline：与 M4a 使用同一个模型

## 1. 同步代码

```bash
WAM_DEPLOY_REPO="${HOME}/bangguo/Liberai_Deploy"
mkdir -p "${WAM_DEPLOY_REPO}"
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='records/' \
  baiduyun32:/mnt/pfs/pfs-yc2F4O/bangguo/code/Liberai_Deploy/ \
  "${WAM_DEPLOY_REPO}/"
source "${WAM_DEPLOY_REPO}/ops/pickplace_v22_q20/wam_bench_eval.sh"
```

## 2. 评测指令

### 任意自然语言

```bash
wam_pickplace_eval_text m4a lr "任意自然语言指令"
wam_pickplace_eval_text m4a_baseline lr "任意自然语言指令"
```

将 `lr` 替换为 `lr`、`rl`、`fb` 或 `bf`。也可以省略最后的指令，在终端提示后输入：

```bash
wam_pickplace_eval_text m4a lr
wam_pickplace_eval_text m4a_baseline lr
```

### Task 1 · 拿起红色辣椒，放入碗里

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 1 lr` | `wam_pickplace_eval m4a_baseline 1 lr` |
| rl | `wam_pickplace_eval m4a 1 rl` | `wam_pickplace_eval m4a_baseline 1 rl` |
| fb | `wam_pickplace_eval m4a 1 fb` | `wam_pickplace_eval m4a_baseline 1 fb` |
| bf | `wam_pickplace_eval m4a 1 bf` | `wam_pickplace_eval m4a_baseline 1 bf` |

### Task 2 · 拿起香蕉，放在红色辣椒上

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 2 lr` | `wam_pickplace_eval m4a_baseline 2 lr` |
| rl | `wam_pickplace_eval m4a 2 rl` | `wam_pickplace_eval m4a_baseline 2 rl` |
| fb | `wam_pickplace_eval m4a 2 fb` | `wam_pickplace_eval m4a_baseline 2 fb` |
| bf | `wam_pickplace_eval m4a 2 bf` | `wam_pickplace_eval m4a_baseline 2 bf` |

### Task 3 · 拿起杯子，放在盘子里

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 3 lr` | `wam_pickplace_eval m4a_baseline 3 lr` |
| rl | `wam_pickplace_eval m4a 3 rl` | `wam_pickplace_eval m4a_baseline 3 rl` |
| fb | `wam_pickplace_eval m4a 3 fb` | `wam_pickplace_eval m4a_baseline 3 fb` |
| bf | `wam_pickplace_eval m4a 3 bf` | `wam_pickplace_eval m4a_baseline 3 bf` |

### Task 4 · 拿起绿色苹果，放入盘子里

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 4 lr` | `wam_pickplace_eval m4a_baseline 4 lr` |
| rl | `wam_pickplace_eval m4a 4 rl` | `wam_pickplace_eval m4a_baseline 4 rl` |
| fb | `wam_pickplace_eval m4a 4 fb` | `wam_pickplace_eval m4a_baseline 4 fb` |
| bf | `wam_pickplace_eval m4a 4 bf` | `wam_pickplace_eval m4a_baseline 4 bf` |

### Task 9 · 拿起红色辣椒，放在绿色苹果上

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 9 lr` | `wam_pickplace_eval m4a_baseline 9 lr` |
| rl | `wam_pickplace_eval m4a 9 rl` | `wam_pickplace_eval m4a_baseline 9 rl` |
| fb | `wam_pickplace_eval m4a 9 fb` | `wam_pickplace_eval m4a_baseline 9 fb` |
| bf | `wam_pickplace_eval m4a 9 bf` | `wam_pickplace_eval m4a_baseline 9 bf` |

### Task 10 · 拿起绿色苹果，放在香蕉上

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 10 lr` | `wam_pickplace_eval m4a_baseline 10 lr` |
| rl | `wam_pickplace_eval m4a 10 rl` | `wam_pickplace_eval m4a_baseline 10 rl` |
| fb | `wam_pickplace_eval m4a 10 fb` | `wam_pickplace_eval m4a_baseline 10 fb` |
| bf | `wam_pickplace_eval m4a 10 bf` | `wam_pickplace_eval m4a_baseline 10 bf` |

### Task 11 · 拿起杯子，放到碗里

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 11 lr` | `wam_pickplace_eval m4a_baseline 11 lr` |
| rl | `wam_pickplace_eval m4a 11 rl` | `wam_pickplace_eval m4a_baseline 11 rl` |
| fb | `wam_pickplace_eval m4a 11 fb` | `wam_pickplace_eval m4a_baseline 11 fb` |
| bf | `wam_pickplace_eval m4a 11 bf` | `wam_pickplace_eval m4a_baseline 11 bf` |

### Task 12 · 拿起红辣椒，放在盘子里

| 方位 | M4a | M4a baseline |
| --- | --- | --- |
| lr | `wam_pickplace_eval m4a 12 lr` | `wam_pickplace_eval m4a_baseline 12 lr` |
| rl | `wam_pickplace_eval m4a 12 rl` | `wam_pickplace_eval m4a_baseline 12 rl` |
| fb | `wam_pickplace_eval m4a 12 fb` | `wam_pickplace_eval m4a_baseline 12 fb` |
| bf | `wam_pickplace_eval m4a 12 bf` | `wam_pickplace_eval m4a_baseline 12 bf` |
