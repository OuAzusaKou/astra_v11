import numpy as np
import scipy.spatial.transform as st

# Old quest mount
# T_glove_base_to_quest_right = np.array([
#     [-0.666999, 0.379404, 0.641221, -3.56165 * 1e-3],
#     [0.715758, 0.0873173, 0.692868, -15.0754 * 1e-3],
#     [0.206887, 0.921102, -0.329802, 60.7160 * 1e-3],
#     [0, 0, 0, 1]
# ])

# T_glove_base_to_quest_left = np.array([
#     [-0.676350, -0.406551, -0.614221, 5.41431 * 1e-3],
#     [-0.702095, 0.103666, 0.704497, -15.2634 * 1e-3],
#     [-0.222740, 0.907728, -0.355552, 60.9500 * 1e-3],
#     [0, 0, 0, 1]
# ])

# New quest mount
T_glove_base_to_quest_right = np.array([
    [0.127790, 0.225870, 0.965739, -19.6764 * 1e-3],
    [-0.171569, -0.954009, 0.245829, -19.0322 * 1e-3],
    [0.976849, -0.197105, -0.0831605, 31.1954 * 1e-3],
    [0, 0, 0, 1]
])

T_glove_base_to_quest_left = np.array([
    [0.127790, -0.225870, -0.965739, 19.6764 * 1e-3],
    [0.171569, -0.954009, 0.245829, -19.0322 * 1e-3],
    [-0.976849, -0.197105, -0.0831605, 31.1954 * 1e-3],
    [0, 0, 0, 1]
])

# T_palm_to_glove = np.array([
#     [0.0, 0.0, -1, -0.02],
#     [-1.0, 0.0, 0.0, 0],
#     [0.0, 1.0, 0.0, 0.045],
#     [0.0, 0.0, 0.0, 1.0]
# ])

T_palm_to_glove = np.array([[0.09409017, 0.33275017, -0.93830931, -0.0368],
                            [-0.98099843,  -0.12964038, -0.1443449, -0.00444],
                            [-0.16967357, 0.9340614 ,  0.31422951, 0.0764],
                            [0, 0, 0, 1]])

T_palm_to_glove_hanbao = np.array([
    [0.0, 0.0, -1, -0.02],
    [1.0, 0.0, 0.0, 0],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
])

T_palm_to_glove_new_right = np.array([
    [0.0, 0.0, -1, -0.02],
    [-1.0, 0.0, 0.0, -0.0005],
    [0.0, 1.0, 0.0, 0.049],
    [0.0, 0.0, 0.0, 1.0]
])

T_palm_to_glove_new_left = np.array([
    [0.0, 0.0, -1, -0.02],
    [-1.0, 0.0, 0.0, 0.0005],
    [0.0, 1.0, 0.0, 0.049],
    [0.0, 0.0, 0.0, 1.0]
])
 
# palm corridante in flange
T_leo_flange_to_palm_right = np.array([
    [0.0, 1.0, 0.0, -0.001],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.029],
    [0.0, 0.0, 0.0, 1.0]
])

T_leo_flange_to_palm_left = np.array([
    [0.0, -1.0, 0.0, -0.001],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.029],
    [0.0, 0.0, 0.0, 1.0]
])

T_flange_to_palm_right = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -0.563 * 1e-3],
    [0.0, 0.0, 1.0, 30.0 * 1e-3],
    [0.0, 0.0, 0.0, 1.0]
])

T_flange_to_palm_left = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.563 * 1e-3],
    [0.0, 0.0, 1.0, 30.0 * 1e-3],
    [0.0, 0.0, 0.0, 1.0]
])

T_robot_tip_to_glove_tip_right = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

def cal_leo_glove_base_relative_to_robot_relative(glove_base_relative_pose, hand_side):
    glove_base_relative_pose_mat = pose10d_to_mat(glove_base_relative_pose)
    this_T_palm_to_glove = T_palm_to_glove_new_right if hand_side == 'right' else T_palm_to_glove_new_left
    T_glove_to_palm = np.linalg.inv(this_T_palm_to_glove)
    T_palm_to_flange = np.linalg.inv(T_leo_flange_to_palm_right if hand_side == 'right' else T_leo_flange_to_palm_left)
    transform_matrix = T_glove_to_palm @ T_palm_to_flange
    robot_relative_pose_mat =  np.linalg.inv(transform_matrix) @ glove_base_relative_pose_mat @ transform_matrix
    return mat_to_pose10d(robot_relative_pose_mat)

def cal_glove_base_relative_to_robot_relative(glove_base_relative_pose, hand_side):
    glove_base_relative_pose_mat = pose10d_to_mat(glove_base_relative_pose)
    this_T_palm_to_glove = T_palm_to_glove_new_right if hand_side == 'right' else T_palm_to_glove_new_left
    T_glove_to_palm = np.linalg.inv(this_T_palm_to_glove)
    T_palm_to_flange = np.linalg.inv(T_flange_to_palm_right if hand_side == 'right' else T_flange_to_palm_left)
    transform_matrix = T_glove_to_palm @ T_palm_to_flange
    robot_relative_pose_mat =  np.linalg.inv(transform_matrix) @ glove_base_relative_pose_mat @ transform_matrix
    return mat_to_pose10d(robot_relative_pose_mat)

def cal_quest_relative_to_robot_relative(quest_relative_pose, hand_side):
    quest_relative_pose_mat = pose10d_to_mat(quest_relative_pose)
    T_quest_to_glove_base = np.linalg.inv(T_glove_base_to_quest_right if hand_side == "right" else T_glove_base_to_quest_left)
    T_glove_to_palm = np.linalg.inv(T_palm_to_glove)
    T_palm_to_flange = np.linalg.inv(T_flange_to_palm_right if hand_side == "right" else T_flange_to_palm_left)
    transform_matrix = T_quest_to_glove_base @ T_glove_to_palm @ T_palm_to_flange
    robot_relative_pose_mat =  np.linalg.inv(transform_matrix) @ quest_relative_pose_mat @ transform_matrix
    return mat_to_pose10d(robot_relative_pose_mat)

def cal_abs_hand_ee_to_glove_ee(hand_ee_pose, hand_side):
    """
    Docstring for cal_abs_hand_ee_to_glove_ee
    
    :param hand_ee_pose: (5, 6)
    :param glove_ee_pose: (5, 9)
    """
    hand_ee_pose_mat = pose_to_mat(hand_ee_pose)
    glove_ee_pose_mat =  np.linalg.inv(T_palm_to_glove_hanbao) @ hand_ee_pose_mat
    if hand_side == "right":
        glove_ee_pose_mat = glove_ee_pose_mat @ T_robot_tip_to_glove_tip_right
    return mat_to_pose10d(glove_ee_pose_mat)


def cal_quest_to_flange(robot_initial_pose, quest_initial_pose, quest_current_pose, hand_side):
    T_quest_to_glove_base = np.linalg.inv(T_glove_base_to_quest_right if hand_side == "right" else T_glove_base_to_quest_left)
    this_T_palm_to_glove = T_palm_to_glove_new_right if hand_side == 'right' else T_palm_to_glove_new_left
    this_T_flange_to_plam = T_flange_to_palm_right if hand_side == 'right' else T_flange_to_palm_left
    T_glove_to_palm = np.linalg.inv(this_T_palm_to_glove)
    T_palm_to_flange = np.linalg.inv(this_T_flange_to_plam)
    transform_matrix =  np.linalg.inv(pose_to_mat(quest_initial_pose) @ T_quest_to_glove_base @ T_glove_to_palm @ T_palm_to_flange) @ pose_to_mat(quest_current_pose) @ T_quest_to_glove_base @ T_glove_to_palm @ T_palm_to_flange
    return mat_to_pose(pose_to_mat(robot_initial_pose) @ transform_matrix)


def cal_tianji_quest_to_initial(robot_initial_pose, quest_initial_pose, quest_current_pose, calibrate_T_quest_F):
    quest_initial_pose_mat = pose_to_mat(quest_initial_pose)
    quest_current_pose_mat = pose_to_mat(quest_current_pose)
    transform_matrix = np.linalg.inv(quest_initial_pose_mat @ calibrate_T_quest_F) @ (quest_current_pose_mat @ calibrate_T_quest_F)
    target_pose = pose_to_mat(robot_initial_pose) @ transform_matrix
    return mat_to_pose(target_pose)


# Transform from right_hand to flange
T_R_F_tianji = np.array([[0, 0, -1, 0],
                  [-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])

# Transform from right_hand to visionpro_base
T_L_F_tianji = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])

T_R_visionpro = np.array([[0, 0, -1, 0],
                  [-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])
 
T_L_visionpro = np.array([[0, 0, -1, 0],
                  [1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, 0, 1]])


# Transform from right_hand to flange
# T_R_F_tianji = np.array([[0, 0, -1, 0],
#                   [1, 0, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1]])

# Transform from right_hand to visionpro_base
# T_L_F_tianji = np.array([[0, 0, 1, 0],
#                   [-1, 0, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1]])

def cal_tianji_relative_pose_to_initial(robot_initial_pose, visionpro_initial_pose, visionpro_current_pose, hand_side):
    # calculate the relative pose between the previous and current teleoperation pose, apply to current pose
    # concat eef_pos and eef_rot_axis_angle to a pose
    T_hand_to_flange = T_R_F_tianji if hand_side == "right" else T_L_F_tianji
    T_visionpro = T_R_visionpro if hand_side == "right" else T_L_visionpro

    visionpro_initial_pose_wo_trans = visionpro_initial_pose.copy()
    visionpro_initial_pose_wo_trans[:-1, 3] = 0.0
    correct_transform = np.linalg.inv(visionpro_initial_pose_wo_trans) @ T_visionpro

    transform_matrix = np.linalg.inv(visionpro_initial_pose @ correct_transform @ T_hand_to_flange) @ (visionpro_current_pose @ correct_transform @ T_hand_to_flange)
    target_pose = pose_to_mat(robot_initial_pose) @ transform_matrix
    return mat_to_pose(target_pose)


# Franka Transform Matrices

x = 2 ** 0.5 / 2

# Transform from robot_base to visionpro_base
T_X_B = np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

# Transform from right_hand to flange
T_R_F = np.array([[-x, x, 0, 0],
                  [0, 0, -1, 0],
                  [-x, -x, 0, 0],
                  [0, 0, 0, 1]])

# Transform from right_hand to visionpro_base
T_L_F = np.array([[x, -x, 0, 0],
                  [0, 0, 1, 0],
                  [-x, -x, 0, 0],
                  [0, 0, 0, 1]])

def cal_relative_pose_to_initial(robot_initial_pose, visionpro_initial_pose, visionpro_current_pose, hand_side):
    # eef_pos: (3,)
    # eef_rot_axis_angle: (3,)
    # prev_pose: (4, 4)
    # curr_pose: (4, 4)
    # calculate the relative pose between the previous and current teleoperation pose, apply to current pose
    
    # concat eef_pos and eef_rot_axis_angle to a pose
    T_hand_to_flange = T_R_F if hand_side == "right" else T_L_F
    transform_matrix = np.linalg.inv(T_X_B @ visionpro_initial_pose @ T_hand_to_flange) @ (T_X_B @ visionpro_current_pose @ T_hand_to_flange)
    transform_matrix[:-1, 3] = transform_matrix[:-1, 3] * 1.0
    target_pose = pose_to_mat(robot_initial_pose) @ transform_matrix
    return mat_to_pose(target_pose)

def cal_relative_pose(eef_pos, eef_rot_axis_angle, prev_pose, curr_pose, hand_side):
    # eef_pos: (3,)
    # eef_rot_axis_angle: (3,)
    # prev_pose: (4, 4)
    # curr_pose: (4, 4)
    # calculate the relative pose between the previous and current teleoperation pose, apply to current pose
    
    # concat eef_pos and eef_rot_axis_angle to a pose
    eef_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=-1)
    eef_pose_matrix = pose_to_mat(eef_pose)
    T_hand_to_flange = T_R_F if hand_side == "right" else T_L_F
    transform_matrix = np.linalg.inv(T_X_B @ prev_pose @ T_hand_to_flange) @ (T_X_B @ curr_pose @ T_hand_to_flange)
    transform_matrix[:-1, 3] = transform_matrix[:-1, 3] * 1.0
    target_pose = eef_pose_matrix @ transform_matrix
    return mat_to_pose(target_pose)

def cal_relative_pose_transform(eef_pos, eef_rot_axis_angle, prev_pose, curr_pose, hand_side):
    # eef_pos: (3,)
    # eef_rot_axis_angle: (3,)
    # prev_pose: (4, 4)
    # curr_pose: (4, 4)
    # calculate the relative pose between the previous and current teleoperation pose, apply to current pose
    
    # concat eef_pos and eef_rot_axis_angle to a pose
    eef_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=-1)
    eef_pose_matrix = pose_to_mat(eef_pose)
    T_hand_to_flange = T_R_F if hand_side == "right" else T_L_F
    transform_matrix = np.linalg.inv(T_X_B @ prev_pose @ T_hand_to_flange) @ (T_X_B @ curr_pose @ T_hand_to_flange)
    transform_matrix[:-1, 3] = transform_matrix[:-1, 3] * 1.0
    target_pose = eef_pose_matrix @ transform_matrix
    return mat_to_pose(target_pose), transform_matrix

def get_delta_action(hand_state, arm_state, hand_action, arm_action, robot_num):
    # hand_state: (40)
    # arm_state: (12,)
    # hand_action: (H, 40)
    # arm_action: (H, 12)
    # output: (H, 40 + 18)

    delta_hand_action = hand_action - hand_state[None, :]  # (H, 40)
    arm_action_dim = 6

    relative_arm_list = []
    for i in range(robot_num):
        this_arm_state = arm_state[i*arm_action_dim:(i+1)*arm_action_dim]
        this_arm_action = arm_action[:, i*arm_action_dim:(i+1)*arm_action_dim]

        relative_arm_action_mat = np.linalg.inv(pose_to_mat(this_arm_state)) @ pose_to_mat(this_arm_action)
        relative_pose10d = mat_to_pose10d(relative_arm_action_mat)  # (H, 9)
        relative_arm_list.append(relative_pose10d)
    delta_arm_action = np.concatenate(relative_arm_list, axis=-1)  # (H, 18)

    delta_action = np.concatenate([delta_arm_action, delta_hand_action], axis=-1)  # (H, 18 + 40)
    return delta_action

def action_6d_to_9d(hand_action, arm_action, robot_num):
    # Convert arm action from 6D (pos3 + rotvec3) to 9D (pos3 + rot6d), hand unchanged
    # hand_action: (H, 40)
    # arm_action: (H, 12) - (pos3 + rotvec3) per arm
    # output: (H, 18 + 40) = (H, 58)
    arm_action_dim = 6

    arm_9d_list = []
    for i in range(robot_num):
        this_arm_action = arm_action[:, i*arm_action_dim:(i+1)*arm_action_dim]
        pose10d = mat_to_pose10d(pose_to_mat(this_arm_action))  # (H, 9)
        arm_9d_list.append(pose10d)
    arm_9d = np.concatenate(arm_9d_list, axis=-1)  # (H, 18)

    return np.concatenate([arm_9d, hand_action], axis=-1)  # (H, 58)

def transform_glove_hand_state(hand_state):
    # hand_state: (10, 6) -> (10, 9)
    return mat_to_pose10d(pose_to_mat(hand_state)).reshape(-1)

def get_glove_delta_action(hand_state, arm_state, hand_action, arm_action):
    # hand_state: (10, 6)
    # arm_state: (2, 6)
    # hand_action: (H, 10, 6)
    # arm_action: (H, 2, 6)
    # output: (H, 9 * (2 + 10))
    relative_action_list = []
    for state, action in zip([arm_state, hand_state], [arm_action, hand_action]):
        assert state.shape[-1] == action.shape[-1] == 6
        for i in range(state.shape[0]):
            this_state = state[i]
            this_action = action[:, i]
            relative_action_mat = np.linalg.inv(pose_to_mat(this_state)) @ pose_to_mat(this_action)
            relative_pose10d = mat_to_pose10d(relative_action_mat)  # (H, 9)
            relative_action_list.append(relative_pose10d)
    delta_action = np.concatenate(relative_action_list, axis=-1)
    return delta_action

def delta_action_to_abs_action(delta_action, hand_state, arm_state, robot_num,
                               hand_absolute=False):
    delta_arm_dim = 9  # pose10d: 3 + 6
    delta_arm_action = delta_action[..., :delta_arm_dim * robot_num]
    delta_hand_action = delta_action[..., delta_arm_dim * robot_num:]

    if hand_absolute:
        # absHand checkpoints emit absolute hand joint angles; pass them through
        # unchanged instead of adding the current hand_state.
        abs_hand_action = delta_hand_action
    else:
        abs_hand_action = delta_hand_action + hand_state[None, :]

    abs_arm_action = np.zeros((delta_action.shape[0], 6 * robot_num),
                              dtype=delta_action.dtype)
    for i in range(robot_num):
        this_arm_state = arm_state[i*6:(i+1)*6]  # pose6d (pos + rotvec)
        this_arm_relative_action = delta_arm_action[:, i*delta_arm_dim:(i+1)*delta_arm_dim]
        abs_action_mat = pose_to_mat(this_arm_state) @ pose10d_to_mat(this_arm_relative_action) 
        abs_arm_action[:, i*6:(i+1)*6] = mat_to_pose(abs_action_mat)  # back to rotvec

    return np.concatenate([abs_arm_action, abs_hand_action], axis=-1)

def abs_action_to_delta_action(abs_action, hand_state, arm_state, robot_num):
    abs_arm_dim = 9  # pose6d: pos + rotvec
    abs_arm_action = abs_action[..., :abs_arm_dim * robot_num]
    abs_hand_action = abs_action[..., abs_arm_dim * robot_num:]

    delta_hand_action = abs_hand_action - hand_state[None, :]

    delta_arm_dim = 9  # pose10d: 3 + 6
    delta_arm_action = np.zeros((abs_action.shape[0], delta_arm_dim * robot_num),
                                dtype=abs_action.dtype)
    for i in range(robot_num):
        this_arm_state = arm_state[i*9:( i+1)*9]  # pose1d (pos + rotvec)
        this_abs_arm_action = abs_arm_action[:, i*abs_arm_dim:(i+1)*abs_arm_dim]
        delta_mat = np.linalg.inv(pose10d_to_mat(this_arm_state)) @ pose10d_to_mat(this_abs_arm_action)
        delta_arm_action[:, i*delta_arm_dim:(i+1)*delta_arm_dim] = mat_to_pose10d(delta_mat)

    return np.concatenate([delta_arm_action, delta_hand_action], axis=-1)


def translate_arm_state(arm_state):
    # input: (12,)  two arms concatenated state
    # output: (18,) transform rot to rot6d
    return mat_to_pose10d(pose_to_mat(arm_state.reshape(-1, 6))).reshape(-1)

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out

def pose6d_to_pose7d(pose6d):
    """Convert pose6d (pos + rotvec) to pose7d (pos + quaternion xyzw)."""
    pos = pose6d[..., :3]
    rotvec = pose6d[..., 3:]
    quat = st.Rotation.from_rotvec(rotvec).as_quat()  # (x, y, z, w)
    return np.concatenate([pos, quat], axis=-1)

def quat_to_rotvec(quat):
    """Convert quaternion (x, y, z, w) to rotation vector."""
    rot = st.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    return rot.as_rotvec()

def xyzquat_to_pose(pose):
    """Convert Pose message to 6D vector (x, y, z, rx, ry, rz)."""
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    rotvec = quat_to_rotvec(pose.orientation)
    return np.concatenate([pos, rotvec])
