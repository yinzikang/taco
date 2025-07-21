#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/3/11 下午8:27   yinzikang      1.0         None
"""
import torch
from isaacgymenvs.utils.torch_jit_utils import *


#####################################################################
### =======================reward functions=======================###
#####################################################################
@torch.jit.script
def compute_pos_reward(relative_pos_body, copter_pos, copter_quat, target_quat,
                       reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target, pos_reward_l0 > pos_reward_l1 > pos_reward_l2
    pos_dist = torch.norm(relative_pos_body, dim=1)
    pos_reward_l0 = 1.0 / (1.0 + pos_dist * pos_dist)
    pos_reward_l1 = 1.0 / (1.0 + 10 * pos_dist * pos_dist)  # closer
    pos_reward = pos_reward_l0 + pos_reward_l1

    # quat similarity
    quat_dist = quat_diff_rad(copter_quat, target_quat)
    rotation_reward_l0 = 1.0 / (1.0 + quat_dist * quat_dist)
    rotation_reward_l1 = 1.0 / (1.0 + 10 * quat_dist * quat_dist)
    rotation_reward = rotation_reward_l0 + rotation_reward_l1

    reward = pos_reward * rotation_reward# / 4 * 8

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(copter_pos[..., 2] < 0.1, ones, die)
    die = torch.where(pos_dist > 10, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward / 100, reset


@torch.jit.script
def compute_rotating_reward(relative_pos, relative_linvel, copter_pos, copter_quat,
                             command, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    r = 1.2  # 机体系下目标位置：r 0 0
    # w = command  # 目标系下角速度： 0 0 w
    v = command[:, -1]

    # 基于目标，建立新的坐标系，原点位于目标，x为目标到机体的连线在水平面的投影，z与世界系重合，y用叉乘
    # new_x,new_y,new_z为新坐标系在世界系下的姿态
    new_z = torch.zeros_like(relative_pos, dtype=torch.float32, device='cuda:0')
    new_z[:, 2] = 1
    new_x = - relative_pos
    new_x[:, 2] = 0
    new_x = new_x / (torch.norm(new_x, dim=1, keepdim=True) + 1e-8)  # 新x
    new_y = torch.cross(new_z, new_x, dim=1)
    new_y = new_y / (torch.norm(new_y, dim=1, keepdim=True) + 1e-8)  # 新y

    # distance to target, pos_reward_l0 > pos_reward_l1 > pos_reward_l2
    hori_dist = torch.norm(relative_pos[:, :2], dim=1) - r  # 世界系下水平距离等于期望半径
    vert_dist = torch.abs(relative_pos[:, 2])  # 世界系下竖直高度保持一致
    pos_dist = torch.sqrt(hori_dist ** 2 + vert_dist ** 2)
    pos_reward_l0 = 1.0 / (1.0 + pos_dist * pos_dist)
    pos_reward_l1 = 1.0 / (1.0 + 10 * pos_dist * pos_dist)  # closer
    pos_reward = pos_reward_l0 + pos_reward_l1

    desired_linvel = torch.zeros_like(relative_linvel, dtype=torch.float32, device='cuda:0')
    desired_linvel[:, 1] = v  # 新坐标系下的期望速度，圆周运动线速度
    normal_linvel = torch.sum(relative_linvel * new_x, dim=1, keepdim=True)  # 新坐标系下x方向速度，法向速度
    tangential_linvel = torch.sum(relative_linvel * new_y, dim=1, keepdim=True)  # 新坐标系下y方向速度，切向速度
    real_linvel = torch.cat((normal_linvel, tangential_linvel, relative_linvel[:, 2].unsqueeze(-1)), dim=1)  # 新坐标系下完整速度
    linvel_dist = torch.norm(real_linvel - desired_linvel, dim=1)
    linvel_reward_l0 = 1.0 / (1.0 + linvel_dist * linvel_dist)
    linvel_reward_l1 = 1.0 / (1.0 + 10 * linvel_dist * linvel_dist)
    linvel_reward = linvel_reward_l0 + linvel_reward_l1

    heading = quaternion_to_matrix(copter_quat)[:, :, 0]  # 世界系下的无人机朝向在水平面上投影,在水平面上投影与new_x反向
    direction_dist = 1 + torch.sum(new_x[:, :2] * heading[:, :2], dim=1) / torch.norm(heading[:, :2], dim=1)
    direction_reward_l0 = 1.0 / (1.0 + direction_dist * direction_dist)
    direction_reward_l1 = 1.0 / (1.0 + 10 * direction_dist * direction_dist)  # closer
    direction_reward = direction_reward_l0 + direction_reward_l1

    reward = pos_reward * linvel_reward * direction_reward# / 8 * 8
    # print(real_linvel)

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(copter_pos[..., 2] < 0.1, ones, die)
    die = torch.where(pos_dist > 10, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward / 100, reset


@torch.jit.script
def compute_flip_reward(relative_pos_body, relative_quat_body, copter_pos,
                        command, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target, pos_reward_l0 > pos_reward_l1 > pos_reward_l2
    pos_dist = torch.norm(relative_pos_body, dim=1)
    pos_reward1_l0 = 1.0 / (1.0 + 1 * pos_dist)
    pos_reward1_l1 = 1.0 / (1.0 + 10 * pos_dist)  # closer
    pos_reward1_l2 = 1.0 / (1.0 + 100 * pos_dist)  # closest
    # pos_reward = pos_reward1_l0 + pos_reward1_l1 + pos_reward1_l2
    pos_reward = pos_reward1_l0 + pos_reward1_l1

    # x_tiltage
    x_tiltage = quaternion_to_matrix(relative_quat_body)[:, 0, 0]
    x_tiltage_dist = 1 - x_tiltage
    x_tiltage_reward_l0 = 1.0 / (1.0 + 10 * x_tiltage_dist)
    x_tiltage_reward = x_tiltage_reward_l0

    # flip
    command_dist = command[:, -1] / 2 / torch.pi  # 圈数
    command_reward_l0 = 1.0 / (1.0 + command_dist * command_dist)
    command_reward_l1 = 1.0 / (1.0 + 10 * command_dist * command_dist)
    command_reward = command_reward_l0 + command_reward_l1
    # command_reward = command_reward_l0

    reward = pos_reward * x_tiltage_reward * command_reward #/ 3 * 8

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(copter_pos[..., 2] < 0.1, ones, die)
    die = torch.where(pos_dist > 10, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward / 100, reset


# @torch.jit.script
def compute_roll_reward(copter_pos, target_quat, relative_pos_body,
                        relative_linvel_body, relative_angvel,
                        command, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    r = 0.5
    v = command[:, -1]
    w = v / r

    desired_pos = torch.zeros_like(relative_pos_body, dtype=torch.float32, device='cuda:0')
    desired_pos[:, 2] = r  # 圆周运动半径，使用机体系下z方向距离
    pos_dist = torch.norm(relative_pos_body - desired_pos, dim=1)
    pos_reward_l0 = 1.0 / (1.0 + pos_dist * pos_dist)
    # pos_reward_l1 = 1.0 / (1.0 + 10 * pos_dist * pos_dist)  # closer
    pos_reward = pos_reward_l0

    desired_linvel_body = torch.zeros_like(relative_linvel_body, dtype=torch.float32, device='cuda:0')
    desired_linvel_body[:, 0] = -v  # 圆周运动线速度，使用机体系下y方向线速度
    linvel_body_dist = torch.norm(relative_linvel_body - desired_linvel_body, dim=1)
    linvel_reward_l0 = 1.0 / (1.0 + 0.1 * linvel_body_dist * linvel_body_dist)
    linvel_reward_l1 = 1.0 / (1.0 + 0.5 * linvel_body_dist * linvel_body_dist)  # closer
    linvel_reward = linvel_reward_l0 + linvel_reward_l1

    desired_angvel_target = torch.zeros_like(relative_angvel, dtype=torch.float32, device='cuda:0')
    desired_angvel_target[:, 1] = w  # 圆周运动角速度，使用目标系下x方向角速度
    relative_angvel_target = quat_rotate(quat_conjugate(target_quat), relative_angvel)
    angvel_dist = torch.norm(relative_angvel_target - desired_angvel_target, dim=1)
    angvel_reward_l0 = 1.0 / (1.0 + 0.1 * angvel_dist * angvel_dist)
    angvel_reward_l1 = 1.0 / (1.0 + 0.5 * angvel_dist * angvel_dist)  # closer
    angvel_reward = angvel_reward_l0 + angvel_reward_l1

    reward = pos_reward * angvel_reward * linvel_reward / 4 * 8  # 能学，但是奖励很低
    # print(reward, pos_reward, linvel_reward, angvel_reward)

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(copter_pos[..., 2] < 0.1, ones, die)
    die = torch.where(pos_dist > 10, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward / 100, reset
