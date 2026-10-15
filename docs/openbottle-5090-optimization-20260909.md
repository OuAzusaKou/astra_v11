# 拧瓶盖 MANO100：推理和图像传输优化结果

后续已完成 [leo1 图片处理第二轮优化](openbottle-5090-camera-optimization-20260909.md)：独立图片处理约 42 → 21 ms，最新现场同连接对照的完整链路为 239 → 197 ms。下文保留第一轮测量及部署说明。

2026-09-09，已部署在现场 RTX 5090 / leo1。继续使用原 OpenBottle MANO 100 条 `step_010000.pt`、BF16、10 步推理、32 action、30 Hz、default mask、头部相机延迟修正 100 ms。未复位或发布机器人动作。

## 最终结果和默认配置

正式对照在**同一个 leo1 进程、同一条 TCP 连接**内，以 server/client/client/server 顺序交替请求。总共预热 8 次，测量 40 次，每种方式各 20 次。每次读取新的现场观测；串行请求，无并发模型请求。

表中总耗时从获取观测后的请求构建开始，到客户端 action 解码完成，**包含 leo1 本地图片处理**，不只是发送后的 RTT。

| 方案 | 每次传输 | 总耗时平均 | P50 | P95 |
| --- | ---: | ---: | ---: | ---: |
| 优化前：原图上传，5090 启动 FFmpeg 子进程 | 8.30 MB | 1290.8 ms | 1341.6 ms | 1432.5 ms |
| 优化后：原图上传，5090 进程内预处理 | 8.30 MB | 248.6 ms | 236.2 ms | 326.7 ms |
| **优化后：leo1 预处理后上传模型输入** | **1.16 MB** | **234.2 ms** | **233.6 ms** | **255.8 ms** |

优化前为上一轮相同口径的 20 次实测；优化后两行来自上述同连接交替对照。不同时间的旧基线只用于衡量大幅改善。

本轮 leo1 方案比优化后的 server 方案平均快 **5.8%**，P95 低 **21.7%**，带宽减少 **86.0%**；相对优化前总延迟减少约 **81.9%**。已把现场 `client.yaml` 默认设为 leo1 预处理，同时保留两种显式配置。样本数量有限，现场 CPU/相机负载变化可能影响差距。

分项均值：

| 指标 | 5090 处理图像 | leo1 处理图像 |
| --- | ---: | ---: |
| leo1 请求构建（含所选本地预处理） | 7.9 ms | 62.2 ms |
| 发出请求到接收响应 RTT | 236.9 ms | 168.4 ms |
| 客户端 action 解码 | 3.8 ms | 3.6 ms |
| 5090 图像转换/预处理 | 60.8 ms | 约 0 ms |
| 模型整体调用（包含在 RTT 中） | 71.7 ms | 137.5 ms |
| action 就绪时观测年龄 | 389.2 ms | 370.6 ms |
| 剩余有效 action 数 | 18–23 | 20–23 |

两种路径的模型段时间确实存在差异，这是完整运行链路下的整体调用计时，不能单独当作相同运行状态的裸模型基准。对比同观测时，两条路径的模型输入及 action 均一致；默认方案按完整客户端耗时和长尾选择。没有进一步 profile 模型内部。

## 实现

### 模型

`ops/openbottle_5090_leo1/optimized_server.py` 继承原 standardized 服务，仅用于本次拧瓶盖 MANO 部署：

- 用容量 16 的 LRU 缓存确定性的 Qwen 文本编码，固定指令预热后命中；不缓存观测、图像或 action。
- 编译 `MoT.prefill_flux2_video_cache` 和 `MoT.forward_action_with_video_cache`，模式 `default`。
- 编译 `VAE.encode`，模式 `max-autotune-no-cudagraphs`。
- 模型输入统一使用 channels-last；CPU Torch 线程设置为 1。
- 返回模型整体耗时和服务端图片处理耗时。

首次编译会明显慢于稳态，本次首次运行约 89 s；后续布局试验另有首次编译约 59 s，均未纳入延迟统计。服务目前已预热。重启后，先运行只读 benchmark 完成预热，再进行机器人评测。

对照期间试验过固定 CPU 性能核，最终已恢复原 CPU affinity，启动脚本未增加绑核要求。

### 图像

`utils/native/imagewam_camera.cpp` 直接链接两台机器已有的系统 FFmpeg 6.1.1，替代每个 chunk 的 6 次 `ffmpeg` 子进程启动。每张图仍然独立执行：

1. 按原参数居中裁剪、cover resize 到 640×480、YUV420P 转换。
2. libx264、CRF 20、GOP 2 编码单帧。
3. H.264 解码回 RGB。

每帧重新创建 codec 上下文，保留原单帧语义；三张图用三个线程并行处理。没有使用跨请求的视频编码历史，也没有删除训练时的 H.264 roundtrip。

`utils/imagewam_camera.py` 提供 Python 调用，并用原训练相同的 float32 / torchvision antialias resize 拼成 `[3,336,288]` 模型输入。

- **server 模式**：保持旧版 CHW BGR 原图协议，5090 完成全部预处理。
- **client 模式**：leo1 完成预处理和拼图，传输 float32 模型输入，`image_source=v21_model_input`；服务端跳过重复图片预处理。MANO 状态转换和 IK 仍在 5090。

float32 拼图为 1,161,216 bytes，加状态和协议元数据后消息体为 **1,162,240 bytes**。原图模式消息体为 **8,295,410 bytes**。没有引入额外图片量化。

## 一致性和验证

- 三张现场图像分别测试 crop=100 和 crop=90，共 6 项：进程内 FFmpeg 与原 CLI 输出逐像素一致。
- float32 拼图与原 server 的训练拼图实现逐元素一致。
- 5090（torch 2.12.1）和 leo1（torch 2.13.0）生成的同一观测拼图 SHA256 均为 `9ff1dac05f95787140fc006bf1e5875429079701c2820c87f1b79f128e5a1a67`。
- 同一冻结观测通过优化后的两种传输路径：模型 action198 和返回 action58 的最大差异均为 **0**。
- 编译版与原 eager BF16 的输出不是逐位一致。验证样本中解码后的最大末端位置差异：左 0.768 mm、右 0.995 mm；旋转差异：左 0.419°、右 0.364°；手关节最大差异 0.01070 rad。尚未评估真机任务成功率。
- 14 项相关客户端、推理循环、action 时间过滤测试通过；新测试确认客户端预处理保留状态协议且不重复上传三张原图。

原始结果在云端本仓库 `records/openbottle_5090_leo1/`：

- `openbottle_same_connection_ab.json`：正式同连接 A/B。
- `camera_native_validation.json`：像素与拼图验证。
- `optimized_validation.json`：模型和最终控制量比较。

## 现场命令

5090 服务地址保持 `192.168.1.18:18213`。部署根目录为 `/home/pc/bangguo/openbottle_5090`，正式启动脚本 `serve.sh`，PID 文件 `server.pid`，日志 `logs/server.log`。原启动脚本保存为 `serve_legacy.sh`。

leo1 默认评测入口现在使用本地预处理：

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle
bash ops/openbottle_5090_leo1/evaluate.sh
```

这条命令会进入原复位及按 Enter 开始的流程，机器人评测由用户自行执行。

切换到 5090 处理原图：

```bash
bash ops/openbottle_5090_leo1/evaluate.sh \
  --config ops/openbottle_5090_leo1/client_server_images.yaml
```

只读复测两种方案：

```bash
bash ops/openbottle_5090_leo1/benchmark.sh \
  --profile --compare-image-preprocess --warmup 8 --requests 40
```

只测某一种方案可通过 `--config` 指定 `client_server_images.yaml` 或 `client_client_images.yaml`。默认客户端预处理依赖优化服务提供的新输入路径；旧服务继续使用 server 配置。

## 原生扩展构建

新增编译入口 `scripts/build_native_camera_preprocess.py`，需要 C++ 编译器以及系统 FFmpeg 开发头文件。现场缺少的 `libavfilter-dev` 通过 `apt-get download` 下载并解压到任务目录，没有升级系统 FFmpeg。

5090 已构建：

```bash
python3 /home/pc/bangguo/openbottle_5090/Liberai_Deploy/scripts/build_native_camera_preprocess.py \
  --include /home/pc/bangguo/openbottle_5090/ffmpeg-dev/usr/include/x86_64-linux-gnu
```

leo1 已构建：

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle
python3 scripts/build_native_camera_preprocess.py \
  --include ffmpeg-dev/usr/include/x86_64-linux-gnu
```

扩展是 ctypes C ABI `.so`，不绑定某个 Python minor 版本。请在目标机器构建；这份实现服务于当前 FFmpeg 6.1.1 的现场部署。
