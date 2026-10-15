# leo1 图片处理第二轮优化

2026-09-09，已部署到 leo1 当前拧瓶盖评测目录。模型服务和评测命令沿用上一轮配置。

## 实测结果

独立图片测试：将现场观测加载到内存，交替执行新旧图片处理，各 50 次。包含三张图的颜色转换、原 FFmpeg 处理、float32 antialias 缩放和拼图，不包含模型或 ROS 订阅。

| 图片处理 | 平均 | P95 |
| --- | ---: | ---: |
| 上一轮实现 | 41.8 ms | 43.9 ms |
| 本轮实现 | **20.7 ms** | **21.7 ms** |

同一条 TCP 连接、同一进程订阅现场 ROS 观测，以 original/optimized/optimized/original 顺序交替请求；预热 8 次，测量 60 次，各方案 30 次：

| 现场链路指标 | 上一轮实现 | 本轮实现 |
| --- | ---: | ---: |
| 请求构建平均（含图片处理、状态转换） | 70.7 ms | **34.8 ms** |
| 请求构建 P95 | 92.2 ms | **48.4 ms** |
| 请求构建至 action 解码平均 | 239.2 ms | **196.8 ms** |
| 请求构建至 action 解码 P95 | 280.5 ms | **240.6 ms** |
| action 就绪时观测年龄平均 | 378.0 ms | **337.6 ms** |
| 剩余有效 action 数 | 19–24 | **20–25** |

独立图片处理约快 50%，现场请求构建节省约 36 ms。完整链路本轮平均快 17.7%；模型整体计时也有波动（两组平均 135.0 / 129.1 ms），不能把完整链路减少的 42 ms 全部归因于图片代码。现场 ROS 订阅、CPU 调度及状态转换使请求构建比独立图片测试更慢。不同时间的样本不能直接视为同负载对照。

## 改动及原因

原实现先在主线程用 NumPy 反转通道并复制三张 1280×720 图片，然后并行 FFmpeg，最后又串行缩放三张图。分项测试中，NumPy BGR→RGB 约 11 ms，串行缩放拼图约 9 ms，三路并行 native FFmpeg 约 16 ms。

1. **颜色转换并入 C++ 已有的输入拷贝。** 直接读取 BGR，在写入 FFmpeg RGB buffer 时交换通道，省掉 Python 侧三张 RGB 中间图的分配和拷贝。
2. **每个相机线程处理到最终 tile。** 三路颜色转换、FFmpeg 和 float32 antialias 缩放一起并行，主线程只等待并拼接结果。

实现位于 `utils/native/imagewam_camera.cpp`、`utils/imagewam_camera.py`；客户端通过 `prepare_model_input()` 使用新路径。没有新增运行时配置。RGB 原生入口继续支持 server 模式。

H.264 仍保留训练相同的 libx264 / CRF 20 / GOP 2 单帧 roundtrip。分项样本中，单相机编码约 8 ms、解码约 2 ms，是后续继续优化的主要成本；移除 roundtrip 或更换编码器会改变模型输入，需要另做效果对照。

## 验证与部署

- crop=100、90 × 三张相机图，BGR 新入口与原 RGB 入口逐像素一致。
- 两组冻结现场观测分别进行各 100 次交替测试，新旧模型输入逐元素一致。
- 14 项客户端、推理循环和 action 时间过滤测试通过；Python 语法及两台机器的 native 编译通过。
- 新 native 扩展通过临时文件编译后原子替换，避免重编译破坏进程已映射的 `.so`。
- leo1 已使用新客户端；5090 的 native 源码和库同步完成，服务无需重启。没有发布机器人动作。

## 复测

在 leo1：

```bash
cd /home/liberai/bangguo/Liberai_Deploy_5090_openbottle

# 同一连接交替比较新旧图片处理，读取现场观测，不发布动作。
bash ops/openbottle_5090_leo1/benchmark.sh \
  --profile --compare-camera-implementation --warmup 8 --requests 60

# 独立图片处理测试，使用已保存的现场观测。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m scripts.benchmark_imagewam_camera \
  --observation-npz records/openbottle_camera_fused_ab.observation.npz \
  --requests 100 --output records/imagewam_camera_recheck.json
```

`--compare-camera-implementation` 使用 `image_preprocess: client` 配置；原来的 `--compare-image-preprocess` 继续用于比较 server/client 处理位置。两种对照选项分别使用。

云端仓库 `records/openbottle_5090_leo1/` 保存 `openbottle_camera_fused_ab.json`（正式现场对照）、`imagewam_camera_fused_live_sample.json`（第二组图片测试）、`imagewam_camera_fused_comparison.json`（第一组图片测试）以及 `imagewam_camera_profile.json`（分项计时）。
