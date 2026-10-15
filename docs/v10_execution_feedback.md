# V10：上一批动作的执行过程图

V10 不再向子会话发送 WAM 的历史动作、状态或观测图。WAM 的推理、执行和原有日志保留，其他控制版本不变。

每个实际发布的 V10 takeover chunk，由只读采样线程在执行期间读取已解码的相机缓冲；不调用 get_obs、不消费控制观测、不改发布频率或轨迹时序。图片在原有 drain 结束后编码保存。

一批 30 行动作均匀选择第 1、8、15、23、30 行的计划时刻，每个时刻采胸前、左腕、右腕，共最多 15 张图。少于 5 行时每行采样一次，不重复凑数。第一行时间来自真实 publish receipt 的 start_stamp + time_from_start[0]，不能简单以模型返回时间为基准。

下一轮的 execution_feedback 包含：

- source_request_id：生成这批动作的子会话请求 ID；chunk_id：此次实际发布批次 ID。
- actions：本批实际提交发布的完整 52D 绝对目标。
- action_index（从 0 开始）、action_row（从 1 开始）、action_target、scheduled_timestamp。
- 每路图的 image_label、timestamp、raw_recv_timestamp、delta_to_scheduled_s，以及 nearest_action_index/nearest_action_row。
- publication_status、drain_completed、capture_status、missing_cameras 和必要的 capture_error。

相机 timestamp 沿用环境的延迟修正 ROS 时间；raw_recv_timestamp 沿用原相机 header 时间。相机与轨迹不是硬同步，最多允许 75 ms 时间偏差，并展示实际对应的最近动作行。图像用于视觉反馈，不能证明指定动作已达到目标，也不构成完整关节、力或碰撞验证。hardware_execution_confirmed/physical_execution_confirmed 仍不能因发布完成或等待结束而设为 true。

相机帧缺失、陈旧、时间不匹配时标为缺失，不用动作结束后的截图补齐。首次接管无上一批记录时 execution_feedback=null。B 复位清空该记录。跨 L2 的记录保留原 L2 和 conditioning_version，并标出是否匹配当前条件。

只发送上一批的执行过程图。最近两轮问答仍保留，但历史消息中的示范图与更早的 execution_feedback 图不重复发送；本次完整 30 帧示范及当前三视角保留。完整本地输入、输出、图像日志不裁剪。

Leo3 图片与完整记录位于 session/execution_feedback/<chunk_id>/，通过现有 SHA-256 校验的 SSH 邮箱传到 Windows。子会话只能看到白名单数值和图片标签，控制代码、机器人接口和文件路径不注入。

离线验证：test_v10_execution_feedback.py 覆盖滚动相机缓冲中的实时采样、时间/动作行映射、取消、缺帧、控制器绑定、SSH 图片导出与 Windows 重建；不连接真机。修改需重启两端新测试后生效，不能用旧进程验收新采样功能。
