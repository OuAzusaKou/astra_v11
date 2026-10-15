# OpenBottle MANO 100 条：RTX 5090 / leo1 延迟分析

此文记录优化前的定位结果。当前已部署优化版本，默认 leo1 预处理，完整延迟平均 234 ms；见 [后续优化结果](openbottle-5090-optimization-20260909.md)。

2026-09-09，现场只读实时观测测试；未复位或发布机器人动作。完成后恢复原服务，地址 `192.168.1.18:18213`。本次仅诊断，没有修改正式服务的预处理、推理步数、缓存或编译配置。

## 结论

当前请求往返平均 **1281.1 ms**，其中 **931.7 ms（72.7%）用于三路图像 FFmpeg 预处理**，模型整体 `infer_action()` 平均 **213.9 ms（16.7%）**。此前报告的 1.244 s 是端到端请求延迟，不能理解成纯模型推理耗时。

现场之前 BF16 95.17 ms / FP8 104.82 ms 的日志真实存在，但使用了另一个 checkpoint、另一份服务端代码、prompt cache、编译后的 MoT / VAE，以及进程内合成观测测试。当前部署未迁移那套缓存与编译优化，因此也不能把本次 213.9 ms 直接视作与其同配置的退化。

## 计时口径

- RTX 5090，torch 2.12.1+cu130；模型 `openbottle_ep100_mano6d_ee1_hand1_20260818_0129/step_010000.pt`。
- BF16，10 步，`default` mask，32 action，30 Hz，模型内部 198D，服务端转为 58D。
- leo1 每次读取现场三路 1280×720 图像和机器人状态；头部相机延迟修正 100 ms，腕部 94 ms。
- 预热 3 次，正式测量 20 次。模型只在 `infer_action()` 入口/出口计时，不再做模型内部插桩，也没有采算子 trace。
- 模型返回 CPU action，墙钟包含该次调用完成的等待；另用一对 CUDA event 交叉检查，平均 213.95 ms，与墙钟 213.91 ms 一致。这个模型整体数值包含文本/图像编码和 action 生成，不代表纯 kernel 执行时间。
- socket 保留原收包实现；服务端接收时间从消息体读取开始，排除等待下一条请求头的空闲时间。
- 客户端往返计时包括序列化、发送、服务端处理、返回和反序列化；客户端构建请求、最终 52D action 解码在它之外。
- 第一轮内部插桩的临时结果不用于下表。下表来自去掉内部插桩后的复测，与此前无插桩的 1244.0 ms 相差约 3%，存在现场运行波动。

## 延迟分布

以下平均值可相加，避免重复累加父子计时：

| 阶段 | 平均耗时 | 往返占比 |
| --- | ---: | ---: |
| 三路 FFmpeg 裁剪/缩放、H.264 编码和解码 | 931.7 ms | 72.7% |
| 模型整体 `infer_action()` | 213.9 ms | 16.7% |
| 服务端接收原始图像消息体 | 75.9 ms | 5.9% |
| 三路 CHW/BGR 转 HWC/RGB 等拷贝 | 29.5 ms | 2.3% |
| Q20 → MANO 输入转换 + MANO → Q20 输出转换 | 7.5 ms | 0.6% |
| 其余序列化、模型输入整理、归一化、日志、返回等 | 22.7 ms | 1.8% |
| **请求往返总计** | **1281.1 ms** | **100%** |

单独的分位数：

| 指标 | 平均 | P50 | P95 |
| --- | ---: | ---: | ---: |
| FFmpeg 预处理 | 931.7 ms | 996.8 ms | 1022.6 ms |
| 模型整体 | 213.9 ms | 217.5 ms | 288.5 ms |
| 请求往返 | 1281.1 ms | 1329.6 ms | 1425.3 ms |
| 返回时观测年龄 | 1430.5 ms | 1480.9 ms | 1579.4 ms |

此外，请求构建平均 6.8 ms，客户端 action 解码平均 2.9 ms。相机的 100 ms 延迟修正在观测年龄中，不包含在请求往返中，不能重复相加。此次 20 次测量中 18 次整段 action 过期，另两次分别剩 6、3 行。

## 为什么 FFmpeg 慢

旧版 standardized 服务在 `_convert_request()` 中串行处理三张相机图像。每张都调用 `_ffmpeg_resize_rgb(codec_roundtrip=True)`：

1. 启动 FFmpeg，读取原始 RGB，裁剪/缩放到 640×480，再用 libx264 编码为 H.264。
2. 等待其结束，再启动一个 FFmpeg，把 H.264 解码回 RGB。
3. 完成后才处理下一张相机图像。

所以每个 chunk 都创建 **6 个 FFmpeg 子进程**。正式请求测量中，3 次编码子进程合计 **502.8 ms**，3 次解码子进程合计 **425.3 ms**，其余约 3.5 ms 是外围拷贝等。这些子进程耗时包含启动、管道传输、编解码和退出，并非纯编码器计算量。

另用同一份现场图像独立测试，每组预热 1 次、测量 5 次：

| 独立实验 | 三张图像平均耗时 |
| --- | ---: |
| 原参数 | 810.1 ms |
| FFmpeg 输入/输出及 filter 限制为单线程 | 668.3 ms |

该组图像的两种输出逐像素一致。单线程约节省 17.5%，但仍有显著进程开销，不能据此承诺所有场景相同收益或像素一致。

只启动 `ffmpeg -version` 的平均耗时就有 **63.5 ms**。原参数实验中，单个编码进程总耗时平均 147.7 ms，FFmpeg 自报运行段约 52.7 ms；单个解码进程总耗时 120.6 ms，自报运行段约 16.8 ms。两者差额包含初始化、管道和退出等，不能全部归为 codec 计算。

因此优化优先级应是复用编解码进程/上下文，减少反复启动，随后再考虑三路并行和线程配置。H.264 roundtrip 是当前训练预处理协议的一部分；直接删除它会改变模型输入，需要单独验证。

## 网络与 MANO

每次请求消息体 **8,295,410 bytes**，主体为三张 1280×720×3 原始图像。千兆链路的理论传输下界约 66.4 ms，实测接收 75.9 ms。此前 0.347 ms ping 只反映小包 RTT，不反映这 8.3 MB 的上传耗时。

MANO 输入转换平均 3.7 ms，输出原生 IK 平均 3.8 ms，合计约 7.5 ms，对当前延迟贡献很小。

## 与现场 95 / 105 ms 测试对比

已读取原始日志：

- `/home/pc/test-cairuwei/Liberai-Model/logs/bench/remote-actiononly-bf16-cmpdefault-0909/bench.json`
- `/home/pc/test-cairuwei/Liberai-Model/logs/bench/remote-actiononly-fp8-cmpdefault-0909/bench.json`

| 项目 | 原现场 BF16 / FP8 测试 | 当前 OpenBottle 部署 |
| --- | --- | --- |
| checkpoint | `iwam-para-unified-teleop-700h-0823/step_100000.pt` | `openbottle_ep100_mano6d_ee1_hand1_20260818_0129/step_010000.pt` |
| 服务端 | `test-cairuwei/Liberai-Model` 的 v2.2 服务 | 本次复制的旧版 standardized 服务 |
| 测试模式 | inproc，合成观测，预热 5 次测 40 次 | leo1 TCP 直连，实时观测，预热 3 次测 20 次 |
| prompt cache | 开启，容量 64，固定指令预热后命中 | 无 prompt cache，每次编码指令 |
| MoT 编译 | `torch.compile(mode='default')`，prefill 和 action 两条路径 | 未编译 |
| VAE 编译 | `max-autotune-no-cudagraphs` | 未编译 |
| mask / 推理步数 | default / 10 | default / 10 |
| 模型输入 H×W | 368×320 | 336×288 |
| action horizon / 维数 | 30 / 326 | 32 / 198 |
| 原始相机输入 | 480×640，合成图像 | 720×1280，实时图像 |
| 网络和本次 6 个 FFmpeg 子进程 | 不包含 | 包含在现场往返中 |

原日志的平均值为：

| 原现场模式 | 进程内完整请求 | 日志 `inference` |
| --- | ---: | ---: |
| BF16 | 95.17 ms | 93.90 ms |
| FP8 | 104.82 ms | 103.52 ms |

`bench_latency.py` 对完整 `_handle()` 前后做 CUDA synchronize，95/105 ms 不是只测到了异步提交。该版本先查 prompt cache，再开始 `inference` 计时；预热后的固定指令不会重复跑文本编码，并且主要推理路径已经编译。FP8 在这组日志里比 BF16 慢，不能把开启 FP8 当成必然提速。

当前部署复用了同一个 torch 环境，但未复用这一版服务端代码及其推理优化。要对齐比较，需要在同一个 OpenBottle checkpoint、相同预处理、相同推理步数下，再对 prompt cache / compile 做整体延迟 A/B；不能仅凭两个不同 checkpoint 的日志把差额准确分配给每项优化。

## 文件与复现

本仓库：

- `ops/openbottle_5090_leo1/profile_server.py`：独立 profiling 入口，只计模型整体及外部阶段；不改原模型源码。
- `scripts/benchmark_imagewam_mix_latency.py --profile`：保存客户端 wire 分段和服务端计时。
- `ops/openbottle_5090_leo1/profile_ffmpeg.py`：固定图像 FFmpeg 独立对照。
- `records/openbottle_5090_leo1/openbottle_5090_profile_coarse.json`：20 次客户端完整测量。
- `records/openbottle_5090_leo1/server_profile_coarse.jsonl`：服务端计时，按 request_id 对应。
- `records/openbottle_5090_leo1/ffmpeg_isolated.json`：FFmpeg 命令、逐次计时及像素校验。
- `records/openbottle_5090_leo1/previous_{bf16,fp8}_bench.json`：现场原始日志副本。

临时 profiling 服务在 5090 的 `/home/pc/bangguo/openbottle_5090/serve_profile.sh`，输出目录通过 `IMAGEWAM_PROFILE_DIR` 指定。profiling 服务启动后可在 leo1 运行：

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle
bash ops/openbottle_5090_leo1/benchmark.sh --profile --warmup 3 --requests 20
```

结束时已恢复 `serve.sh` 的原服务，不再运行 profiling 服务。
