"""Schedule decoded arm/hand chunks against the camera observation clock."""

import numpy as np


def publish_observation_aligned_chunk(env, arm, hand, offsets, observation_timestamp):
    """Drop expired rows and preserve their original deadlines on both planners.

    ``offsets`` are seconds from the latency-corrected observation timestamp,
    NOT from the inference response. The ROS clock and camera use the same
    clock domain. No second camera-latency subtraction or time-axis rebasing.
    An entirely expired chunk must not cancel the previously queued trajectory.
    Returns (published row count, observation age in seconds).
    """
    arm, hand = np.asarray(arm), np.asarray(hand)
    offsets = np.asarray(offsets, dtype=np.float64)
    stamp = env._node.get_clock().now().to_msg()
    age = stamp.sec + stamp.nanosec * 1e-9 - float(observation_timestamp)
    remaining = offsets - age
    # Use one mask and one header stamp so arms and hands keep the same rows
    # and absolute deadlines, even though they are published sequentially.
    keep = remaining > max(env.arm_exec_latency, env.hand_exec_latency)
    count = int(np.count_nonzero(keep))
    if count:
        env.publish_arm_chunk(
            arm[keep], remaining[keep], cancel_previous=True, start_stamp=stamp
        )
        env.publish_hand_chunk(
            hand[keep], remaining[keep], cancel_previous=True, start_stamp=stamp
        )
    return count, age
