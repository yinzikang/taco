#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Simulation environment for different FPV tasks

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/2/21 16:30   yinzikang      1.0         None
"""
import math
import numpy as np
import os
import torch
import bisect
import xml.etree.ElementTree as ET
from .control.angvel_control import angvel_control
from .control.battery_dynamics import Battery_Dynamics as battery_dynamics
from .control.thrust_dynamics import RotorDynamics as rotor_dynamics
from .control.thrust_dynamics import AeroDynamics as aero_dynamics
from .control.fpv_dynamics import FpvDynamicsReal2Sim as dynamics_real2sim
from .control.task_reward import *
from .control.logger import Logger as logger

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task_asymmetry import VecTask

import matplotlib

matplotlib.use('TkAgg')


class FpvBase(VecTask):
    """
             ^X
    2(ccw)   |   1(cw)
     Y<-----Z up-----
    3(cw)    |   0(ccw)
      T  = f0 + f1 + f2 + f3
      tx = (-f0 - f1 + f2 + f3) * sqrt(2)/2 * d
      ty = (f0 - f1 - f2 + f3) * sqrt(2)/2 * d
      fpv: tz = (-f0 + f1 - f2 + f3) * k
      vpp: tz = (-|f0| + |f1| - |f2| + |f3|) * k

      rotor 0: green, index 6 in sim
      rotor 1: blue, index 8 in sim
      rotor 2: black, index 2 in sim
      rotor 3: red, index 4 in sim
      d: distance from propeller to center
      k: ratio of torque to force
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.substeps = self.cfg["sim"]["substeps"]
        self.randomization_params = self.cfg["task"]["randomization_params"]  # params randomization, required by md

        # fpv
        self.random_copter_pos = self.cfg["random_copter_pos"]  # Whether to randomize the initial position of the UAV
        self.random_copter_quat = self.cfg["random_copter_quat"]  # Whether to randomize the initial attitude of the UAV
        self.random_copter_vel = self.cfg["random_copter_vel"]  # Whether to randomize the initial velocity of the UAV

        # target
        self.random_target_pos = self.cfg["random_target_pos"]  # Whether to randomize the initial position of the target
        self.random_target_yaw = self.cfg["random_target_yaw"]  # Whether to randomize the initial yaw of the target

        # battery
        self.battery_consumption = self.cfg["battery_consumption"]  # Whether to consider battery voltage drop
        self.random_voltage = self.cfg["random_voltage"]  # Whether to randomize the initial battery voltage

        # motor
        self.rotor_response_time = self.cfg["rotor_response_time"]  # Motor response time
        self.rotor_noise = self.cfg["rotor_noise"]  # Whether to consider motor noise
        self.rotor_delay = self.cfg["rotor_delay"]  # Whether to consider motor delay
        self.rotor_response = self.cfg["rotor_response"]  # Whether to consider motor response curve
        self.random_rotordynamic_coe = self.cfg["random_rotordynamic_coe"]  # Whether to randomize motor steady-state response fitting parameters
        self.random_rotor_delay = self.cfg["random_rotor_delay"]  # Whether to randomize motor delay time
        self.random_rotor_response = self.cfg["random_rotor_response"]  # Whether to randomize motor response time
        self.random_rotor_speed = self.cfg["random_rotor_speed"]  # Whether to randomize initial rotor speed
        self.random_aerodynamic_coe = self.cfg["random_aerodynamic_coe"]  # Whether to randomize aerodynamic fitting parameters

        # delay
        self.delay_time_max = self.cfg["delay_time_max"]  # Maximum supported delay simulation time, in ms, now set to 100
        self.delay_time = self.cfg["delay_time"]  # Delay from the real time of the state to the time when the corresponding action actually produces thrust, in ms. Measured to be about [3, 4, 5, 6, 7, 8, 9]-2ms, distribution: [0.0375, 0.125, 0.25, 0.25, 0.2125, 0.1, 0.025]
        self.ramdom_delay_time = self.cfg["ramdom_delay_time"]  # Whether to randomize delay time
        self.ramdom_deploy_time = self.cfg["ramdom_deploy_time"]  # Whether to randomize action execution time, default is 10ms for each action, but there is a certain deviation

        # command
        self.random_command = self.cfg["random_command"]  # Whether to consider command randomization, such as different speeds
        self.num_commands = 2

        # other
        self.difficulty = self.cfg["difficulty"]  # Current training level, used to change task difficulty, cannot change the difficulty of UAV attitude randomization and target attitude randomization. Automatically adjusted in PPO

        self.num_bodies = 10  # 9 + 1

        # Actions: output of the actor
        self.cfg["env"]["numActions"] = self.num_acts = 4

        # Observations: input to the actor: noisy (relative position in body frame, relative rotation matrix of target in body frame, relative velocity in body frame) + voltage + last action + truncated absolute height + task (selection + command), final dimension aligned with the highest dimensional task
        # States: input to the critic: noise-free (relative position in body frame, relative rotation matrix of target in body frame, relative velocity in body frame) + voltage + last action + truncated absolute height + task (selection + command)
        # self.cfg["env"]["numObservations"] = self.num_obs = 13 + 1 + self.num_acts + 1 + self.num_commands  # quat with last actions
        self.cfg["env"]["numObservations"] = self.num_obs = 18 + 1 + self.num_acts + 1 + self.num_commands  # rot with last actions
        self.len_obs = self.cfg["env"]["lenObservations"]  # Number of frames for observation
        self.cfg["env"]["numStates"] = self.num_states = self.num_obs
        self.len_states = self.cfg["env"]["lenStates"]  # Number of frames for state

        self.observation_noise = self.cfg["observation_noise"]  # Whether to consider observation noise

        # record
        self.record_flag = self.cfg["record_flag"]  # Whether to record data
        if self.record_flag:
            self.logger = logger(self.cfg["record_path"])

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # information about the copter and target
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        # information about the joints
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # fpv information
        self.root_states = vec_root_tensor[:, 0, :]
        self.copter_pos = self.root_states[:, 0:3]
        self.copter_quat = self.root_states[:, 3:7]  # xyzw
        self.copter_rpy = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.copter_rpy[:, 0], self.copter_rpy[:, 1], self.copter_rpy[:, 2] = get_euler_xyz_v1(self.copter_quat)
        self.copter_rpy_old = self.copter_rpy.clone()  # rpy at the previous time step, used together with copter_rpy_continuous
        self.copter_rpy_continuous = self.copter_rpy.clone()  # Not constrained in range, to achieve continuous rpy

        self.copter_linvel = self.root_states[:, 7:10]
        self.copter_angvel = self.root_states[:, 10:13]
        self.copter_linvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.copter_linvel)
        self.copter_angvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.copter_angvel)

        # target information
        self.target_states = vec_root_tensor[:, 1, :]
        self.target_pos = self.target_states[:, 0:3]
        self.target_quat = self.target_states[:, 3:7]
        self.target_linvel = self.target_states[:, 7:10]
        self.target_angvel = self.target_states[:, 10:13]

        # command info
        self.command = torch.zeros((self.num_envs, self.num_commands), dtype=torch.float32, device=self.device)
        self.time_index = torch.tensor([500], dtype=torch.long, device=self.device)  # Command reset time point

        # init information
        self.first_reset = True
        self.refresh_state()
        self.first_reset = True

        # controller
        self.angvel_controller = angvel_control(1, 0, self.num_envs, self.device, self.dt)
        self.battery_dynamics = battery_dynamics(self.num_envs, self.device, self.battery_consumption, self.dt)
        self.rotor_dynamics = rotor_dynamics(self.num_envs, self.device, self.rotor_response_time)
        self.aero_dynamics = aero_dynamics(self.num_envs, self.device)
        self.dynamics_real2sim = dynamics_real2sim()

        # control signal
        # u is a 4D vector consisting of the resultant force on the body and three moment abstractions
        self.u = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # throttle is the throttle for the four rotors
        self.throttle = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # voltage is the current battery voltage, initially 26v, this is just a receiving place, the actual voltage is completely determined by self.battery_dynamics
        self.battery_voltage = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        # rotor_speed is the speed of the four propellers, in revolutions per second
        self.rotor_speed = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # rotor_force and rotor_torque are the forces and torques on each rotor considering motor response, generated by aerodynamics, including lift and torque of each rotor
        self.rotor_force = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.rotor_torque = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # body_force and body_torque are the forces and torques on the body considering motor response, generated by aerodynamics
        self.body_force = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.body_torque = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        # Various reset buffers
        self.reset_env_ids = []

        self.actions = torch.zeros((self.num_envs, self.num_acts), dtype=torch.float32, device=self.device)
        self.actions_old = torch.zeros((self.num_envs, self.num_acts), dtype=torch.float32, device=self.device)

        # delay
        self.actions_remained_buffer = torch.zeros((self.num_envs, self.num_acts, self.delay_time_max), dtype=torch.float32, device=self.device)  # Actions that have been calculated but not yet executed. 10 new actions are added each time
        if self.ramdom_delay_time:  # The number of actions not yet executed in each environment
            self.actions_remained_length = torch.clamp(self.delay_time - torch.clamp(torch.round(torch.normal(0, 1, size=(self.num_envs, 1))), -3, 3), min=0).long().to(self.device)
        else:
            self.actions_remained_length = self.delay_time * torch.ones((self.num_envs, 1), dtype=torch.int32, device=self.device)
        self.mid_step_count = 0  # Used together with delay in mid_step

        # Interact with gym, apply forces and torques to each body, basically only applied to the body and four rotors
        self.forces_sim = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        self.torques_sim = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        if self.viewer:
            cam_pos = gymapi.Vec3(-3.0, -2.0, 3.8)
            cam_target = gymapi.Vec3(1.2, 1.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, -1, 13)
            self.rb_positions = self.rb_states[:, 0:3]
            self.rb_quats = self.rb_states[:, 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = - 9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self.sim_params.substeps = self.substeps

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        # Set the ground height to -0.2m to avoid collision with the UAV
        # In fact, the ground is only used to distinguish up and down, and there is no test for collision
        # Because 'dead' is already set in the reward before colliding with the ground
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.2
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # fpv xml
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/xml')
        asset_file = "fpv_without_duct.xml"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Load asset with default control type of position for all joints
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.linear_damping = 0.0
        asset_options.angular_damping = 0.0
        asset_options.max_linear_velocity = torch.inf  # WARNING: maximum linear velocity
        asset_options.max_angular_velocity = torch.inf  # WARNING: maximum angular velocity
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # create target balls
        # fix the ball to the axis
        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.05, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(0.0, 0.0, 4.0)
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.fpv_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            fpv_handle = self.gym.create_actor(env, asset, default_pose, "fpv", i, 1, 0)
            self.fpv_handles.append(fpv_handle)

            # Configure DOF properties
            props = self.gym.get_actor_dof_properties(env, fpv_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)  # keep rotor geo display fixed
            props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env, fpv_handle, props)

            # pretty colors, colors set in xml cannot be loaded
            chassis_color = gymapi.Vec3(0.8, 0.6, 0.2)
            rotor_0 = gymapi.Vec3(0., 0., 0.)
            rotor_1 = gymapi.Vec3(1, 0, 0)
            rotor_2 = gymapi.Vec3(0., 1., 0.)
            rotor_3 = gymapi.Vec3(0., 0., 1.)
            arm_color = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(env, fpv_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)
            self.gym.set_rigid_body_color(env, fpv_handle, 1, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, fpv_handle, 3, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, fpv_handle, 5, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, fpv_handle, 7, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, fpv_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, rotor_0)
            self.gym.set_rigid_body_color(env, fpv_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, rotor_1)
            self.gym.set_rigid_body_color(env, fpv_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, rotor_2)
            self.gym.set_rigid_body_color(env, fpv_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, rotor_3)

            marker_handle = self.gym.create_actor(env, marker_asset, default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    #####################################################################
    ### =======================rl functions=======================###
    #####################################################################
    def pre_physics_step(self, _actions):
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        self.reset_idx(self.reset_env_ids)
        self.actions_old.copy_(self.actions)
        self.actions.copy_(_actions)
        if self.ramdom_deploy_time:  # The action may be executed for 9-11ms
            timesteps_this_action = 10 - torch.clamp(torch.round(torch.normal(0, 1, size=(self.num_envs, 1))), -1, 1).long().to(self.device)
        else:
            timesteps_this_action = 10 * torch.ones((self.num_envs, 1), device=self.device).long()
        start_idx = self.actions_remained_length.expand(-1, 4)
        end_idx = (self.actions_remained_length + timesteps_this_action).expand(-1, 4)
        mask = (torch.arange(100, device=self.device).view(1, 1, -1) >= start_idx.unsqueeze(-1)) & (torch.arange(100, device=self.device).view(1, 1, -1) < end_idx.unsqueeze(-1))
        self.actions_remained_buffer[mask] = _actions.unsqueeze(-1).expand_as(self.actions_remained_buffer)[mask]  # Put this action into the pending buffer
        self.actions_remained_length += timesteps_this_action  # Update the length of pending actions for each environment
        self.mid_step_count = 0

    def refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Update UAV states that are not automatically updated
        self.copter_rpy[:, 0], self.copter_rpy[:, 1], self.copter_rpy[:, 2] = get_euler_xyz_v1(self.copter_quat)
        copter_rpy_delta = self.copter_rpy - self.copter_rpy_old
        if self.first_reset:  # During the first reset, the attitude of the UAV and the target will jump, so it cannot be regarded as a jump caused by the attitude angle crossing the boundary
            self.first_reset = False
        else:
            copter_rpy_delta = torch.where(copter_rpy_delta > 1, copter_rpy_delta - 2 * np.pi, copter_rpy_delta)
            copter_rpy_delta = torch.where(copter_rpy_delta < -1, copter_rpy_delta + 2 * np.pi, copter_rpy_delta)
        self.copter_rpy_continuous += copter_rpy_delta
        self.copter_rpy_old = self.copter_rpy.clone()

        self.copter_linvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.copter_linvel)  # In the body frame, the absolute linear velocity of the UAV relative to the world frame
        self.copter_angvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.copter_angvel)  # In the body frame, the absolute angular velocity of the UAV relative to the world frame
        # print(self.copter_angvel_body[0])

        # relative information
        self.relative_pos = self.target_pos - self.copter_pos  # In the world frame, the relative position of the UAV to the target
        self.relative_pos_body = quat_rotate(quat_conjugate(self.copter_quat), self.relative_pos)  # In the body frame, the relative position of the UAV to the target
        self.relative_quat_body = quat_mul(quat_conjugate(self.copter_quat), self.target_quat)  # In the body frame, the relative attitude of the UAV to the target
        self.relative_linvel = self.target_linvel - self.copter_linvel  # In the world frame, the relative linear velocity of the UAV to the target
        self.relative_angvel = self.target_angvel - self.copter_angvel  # In the world frame, the relative angular velocity of the UAV to the target
        self.relative_linvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.relative_linvel)  # In the body frame, the relative angular velocity of the UAV to the target
        self.relative_angvel_body = quat_rotate(quat_conjugate(self.copter_quat), self.relative_angvel)  # In the body frame, the relative angular velocity of the UAV to the target

    def mid_physics_step(self):
        self.refresh_state()

        # Low-level control law
        delayed_action = self.actions_remained_buffer[torch.arange(self.num_envs), :, torch.clamp(self.actions_remained_length - 1, max=self.mid_step_count).squeeze(-1)]  # Only take valid actions
        # print(delayed_action)
        self.mid_step_count += 1
        self.angular_vel_control(delayed_action)

        # Low-level thrust simulation
        self.control_with_thrusts(self.reset_env_ids)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1  # 根据domain_randomization.md要求，增加此行代码

        self.actions_remained_buffer[:, :, 0:- 10] = self.actions_remained_buffer[:, :, 10:]
        self.actions_remained_length -= 10
        self.actions_remained_length = torch.clamp(self.actions_remained_length, min=0)

        self.refresh_state()  # 更新仿真状态：包括各种仿真器中量以及自定义的相对量
        self.compute_observation_state()
        self.compute_reward()

        # record
        if self.record_flag:
            self.record()

    def compute_observation_state(self):
        # 处理旧观测
        self.obs_buf[:, :-1, :] = self.obs_buf[:, 1:, :].clone()
        # 引入新观测
        self.obs_buf[:, -1, 0:3] = self.relative_pos_body.clone() / 3
        self.obs_buf[:, -1, 3:12] = quaternion_to_matrix(self.relative_quat_body.clone()).reshape(self.num_envs, 9)  # 按行拼接
        self.obs_buf[:, -1, 12:15] = self.relative_linvel_body.clone() / 2
        self.obs_buf[:, -1, 15:18] = self.relative_angvel_body.clone() / torch.pi
        self.obs_buf[:, -1, 18] = (self.battery_voltage.squeeze().clone() - 23) / 3
        self.obs_buf[:, -1, 19:23] = self.actions.clone()
        self.obs_buf[:, -1, 23] = 4 * torch.clamp(self.copter_pos[:, 2].clone(), 0, 0.5) - 1  # 离地面的距离，截断到0-0.5,放缩到-1-1

        if self.observation_noise:
            self.obs_buf[:, -1, 0:3] += (self.difficulty * torch.normal(0, 0.06 / 3 / 3, size=self.obs_buf[:, -1, 0:3].size(), device=self.device))
            # 0.0174533 rad = 1 deg。以下均为弧度值，0.1弧度=5.7度
            noise_quat = self.rand_quat(self.num_envs, self.difficulty * 0.05, self.difficulty * 0.05, self.difficulty * 0.05)
            self.obs_buf[:, -1, 3:12] = quaternion_to_matrix(quat_mul(self.relative_quat_body, noise_quat)).reshape(self.num_envs, 9)  # 按行拼接
            self.obs_buf[:, -1, 12:15] += (self.difficulty * torch.normal(0, 0.1 / 3 / 2, size=self.obs_buf[:, -1, 12:15].size(), device=self.device))  # 3sigma原则设置10cm/s抖动
            self.obs_buf[:, -1, 15:18] += (self.difficulty * torch.normal(0, 60 / 3 / 180, size=self.obs_buf[:, -1, 15:18].size(), device=self.device))  # 3sigma原则设置60度/s抖动，放缩的pi消掉了
            self.obs_buf[:, -1, 18] += (self.difficulty * torch.normal(0, 0.06 / 3, size=self.obs_buf[:, -1, 19].size(), device=self.device))  # 3sigma原则设置0.06v
            self.obs_buf[:, -1, 23] += (self.difficulty * torch.normal(0, 0.06 / 3 / 3, size=self.obs_buf[:, -1, 24].size(), device=self.device))

        # 处理旧状态
        self.states_buf[:, :-1, :] = self.states_buf[:, 1:, :].clone()
        # 引入新状态
        self.states_buf[:, -1, 0:3] = self.relative_pos_body.clone() / 3
        self.states_buf[:, -1, 3:12] = quaternion_to_matrix(self.relative_quat_body.clone()).reshape(self.num_envs, 9)  # 按行拼接
        self.states_buf[:, -1, 12:15] = self.relative_linvel_body.clone() / 2
        self.states_buf[:, -1, 15:18] = self.relative_angvel_body.clone() / torch.pi
        self.states_buf[:, -1, 18] = (self.battery_voltage.squeeze().clone() - 23) / 3
        self.states_buf[:, -1, 19:23] = self.actions.clone()
        self.states_buf[:, -1, 23] = 4 * torch.clamp(self.copter_pos[:, 2].clone(), 0, 0.5) - 1  # 离地面的距离，截断到0-0.5,放缩到-1-1

    # def compute_observation_state(self):
    #     # 处理旧观测
    #     self.obs_buf[:, :-1, :] = self.obs_buf[:, 1:, :].clone()
    #     # 引入新观测
    #     self.obs_buf[:, -1, 0:3] = self.relative_pos_body.clone() / 3
    #
    #     quat_processed = torch.zeros_like(self.relative_quat_body)
    #     mask = self.relative_quat_body[:, -1] < 0  # 创建一个布尔掩码，表示实部是否小于0
    #     quat_processed[mask] = -self.relative_quat_body[mask]  # 对满足条件的行进行取反操作
    #     quat_processed[~mask] = self.relative_quat_body[~mask]  # 对不满足条件的行直接赋值
    #     self.obs_buf[:, -1, 3:7] = quat_processed
    #
    #     self.obs_buf[:, -1, 7:10] = self.relative_linvel_body.clone() / 2
    #     self.obs_buf[:, -1, 10:13] = self.relative_angvel_body.clone() / torch.pi
    #     self.obs_buf[:, -1, 13] = (self.battery_voltage.squeeze().clone() - 23) / 3
    #     self.obs_buf[:, -1, 14:18] = self.actions.clone()
    #     self.obs_buf[:, -1, 18] = 4 * torch.clamp(self.copter_pos[:, 2].clone(), 0, 0.5) - 1  # 离地面的距离，截断到0-0.5,放缩到-1-1
    #
    #     if self.observation_noise:
    #         self.obs_buf[:, -1, 0:3] += (self.difficulty * torch.normal(0, 0.06 / 3, size=self.obs_buf[:, -1, 0:3].size(), device=self.device))
    #         # 0.0174533 rad = 1 deg。以下均为弧度值，0.1弧度=5.7度
    #         # noise_quat = self.rand_quat(self.num_envs, self.difficulty * 0.05, self.difficulty * 0.05, self.difficulty * 0.05)
    #         # self.obs_buf[:, -1, 3:7] = quaternion_to_matrix(quat_mul(self.relative_quat_body, noise_quat)).reshape(self.num_envs, 9)  # 按行拼接
    #         self.obs_buf[:, -1, 7:10] += (self.difficulty * torch.normal(0, 0.1 / 3 / 2, size=self.obs_buf[:, -1, 7:10].size(), device=self.device))
    #         self.obs_buf[:, -1, 10:13] += (self.difficulty * torch.normal(0, 60 / 3 / 180, size=self.obs_buf[:, -1, 10:13].size(), device=self.device))
    #         self.obs_buf[:, -1, 13] += (self.difficulty * torch.normal(0, 0.06 / 3, size=self.obs_buf[:, -1, 14].size(), device=self.device))  # 3sigma原则设置0.06v
    #         self.obs_buf[:, -1, 18] += (self.difficulty * torch.normal(0, 0.06 / 3, size=self.obs_buf[:, -1, 19].size(), device=self.device))
    #
    #     # 处理旧状态
    #     self.states_buf[:, :-1, :] = self.states_buf[:, 1:, :].clone()
    #     # 引入新状态
    #     self.states_buf[:, -1, 0:3] = self.relative_pos_body.clone() / 3
    #
    #     quat_processed = torch.zeros_like(self.relative_quat_body)
    #     mask = self.relative_quat_body[:, -1] < 0  # 创建一个布尔掩码，表示实部是否小于0
    #     quat_processed[mask] = -self.relative_quat_body[mask]  # 对满足条件的行进行取反操作
    #     quat_processed[~mask] = self.relative_quat_body[~mask]  # 对不满足条件的行直接赋值
    #     self.states_buf[:, -1, 3:7] = quat_processed
    #
    #     self.states_buf[:, -1, 7:10] = self.relative_linvel_body.clone() / 2
    #     self.states_buf[:, -1, 10:13] = self.relative_angvel_body.clone() / torch.pi
    #     self.states_buf[:, -1, 13] = (self.battery_voltage.squeeze().clone() - 23) / 3
    #     self.states_buf[:, -1, 14:18] = self.actions.clone()
    #     self.states_buf[:, -1, 18] = 4 * torch.clamp(self.copter_pos[:, 2].clone(), 0, 0.5) - 1  # 离地面的距离，截断到0-0.5,放缩到-1-1/

    def compute_reward(self):
        """compute reward according to the task."""
        raise NotImplementedError("compute_reward must be implemented in subclass")

    #####################################################################
    ### ========================reset functions========================###
    #####################################################################
    def reset_idx(self, env_ids):
        # reset copter, 0-env_num
        reset_copter_ids = env_ids
        if len(reset_copter_ids) > 0:  # 0,2,4,...,2*env_num-2
            copter_actor_indices = self.reset_copter_idx(reset_copter_ids)
        else:
            copter_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)

        # reset controllers, 0-env_num
        reset_controller_ids = reset_copter_ids
        if len(reset_controller_ids) > 0:
            self.reset_controller_idx(reset_controller_ids)

        # reset env, 0-env_num
        reset_env_ids = reset_copter_ids
        if len(reset_env_ids) > 0:
            self.reset_env_idx(reset_env_ids)

        # reset target, 0-env_num
        reset_target_ids = env_ids
        if len(reset_target_ids) > 0:  # 1,3,5,...,2*env_num-1
            target_actor_indices = self.reset_target_idx(reset_target_ids)
        else:
            target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)

        # reset command, 0-env_num
        reset_command_ids = self.reset_command_condition()
        if len(reset_command_ids) > 0:
            self.reset_command_idx(reset_command_ids)

        # do reset
        reset_indices = torch.unique(torch.cat([copter_actor_indices, target_actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # dump record
        if self.record_flag:
            if 0 in reset_copter_ids:
                self.logger.dump_buffer()
                self.logger.reset_buffer()

    def reset_copter_idx(self, env_ids):
        """不同任务中，copter的随机化内容、尺度不同，相互独立"""
        raise NotImplementedError("reset_copter_idx must be implemented in subclass")

    def reset_target_idx(self, env_ids):
        """
        正常训练：开关分别控制位置、姿态、速度、电机转速的随机化
        """
        num_resets = len(env_ids)

        # pos
        if self.random_target_pos:
            self.target_pos[env_ids, 0:2] = self.difficulty * torch_rand_float(-2, 2, (num_resets, 2), self.device)
            self.target_pos[env_ids, 2] = 3 + self.difficulty * torch_rand_float(-2, 2, (num_resets, 1), self.device).squeeze()
            # self.target_pos[env_ids, 0] = 0
        else:
            self.target_pos[env_ids, 0:2] = 0
            self.target_pos[env_ids, 2] = 3

        # quat
        # roll pitch
        rp = torch.zeros(num_resets, dtype=torch.float32, device=self.device)
        # yaw
        if self.random_target_yaw:
            yaw = torch_rand_float(-torch.pi, torch.pi, (num_resets, 1), self.device).squeeze()
        else:
            yaw = torch.zeros(num_resets, dtype=torch.float32, device=self.device)
        self.target_quat[env_ids, :] = quat_from_euler_xyz(rp, rp, yaw).squeeze()

        return self.all_actor_indices[env_ids, 1].flatten()

    def reset_controller_idx(self, env_ids):
        num_resets = len(env_ids)

        # reset controller
        self.angvel_controller.reset(env_ids)
        self.battery_dynamics.reset(env_ids, self.random_voltage)
        self.rotor_speed[env_ids] = self.rotor_dynamics.reset(env_ids, self.difficulty, self.rotor_noise, self.rotor_delay, self.rotor_response,
                                                              self.random_rotordynamic_coe, self.random_rotor_delay, self.random_rotor_response, self.random_rotor_speed)
        self.aero_dynamics.reset(env_ids, self.random_aerodynamic_coe, self.difficulty)

    def reset_env_idx(self, env_ids):
        num_resets = len(env_ids)

        # reset control signal
        self.u[env_ids] = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
        self.throttle[env_ids] = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
        self.battery_voltage[env_ids] = torch.zeros((num_resets, 1), dtype=torch.float32, device=self.device)
        self.rotor_force[env_ids] = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
        self.rotor_torque[env_ids] = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
        self.body_force[env_ids] = torch.zeros((num_resets, 1), dtype=torch.float32, device=self.device)
        self.body_torque[env_ids] = torch.zeros((num_resets, 1), dtype=torch.float32, device=self.device)

        self.actions[env_ids] = torch.zeros((num_resets, self.num_acts), dtype=torch.float32, device=self.device)
        self.actions_old[env_ids] = torch.zeros((num_resets, self.num_acts), dtype=torch.float32, device=self.device)
        self.actions_remained_buffer[env_ids] = torch.zeros((num_resets, self.num_acts, self.delay_time_max), dtype=torch.float32, device=self.device)  # 计算得到但还没执行的动作。每次加入10个新action
        if self.ramdom_delay_time:  # 每个环境中还没有执行的动作的数量
            self.actions_remained_length[env_ids] = torch.clamp(self.delay_time - torch.clamp(torch.round(torch.normal(0, 1, size=(num_resets, 1))), -3, 3), min=0).long().to(self.device)
        else:
            self.actions_remained_length[env_ids, :] = self.delay_time * torch.ones((num_resets, 1), dtype=torch.int32, device=self.device)

        self.forces_sim[env_ids] = torch.zeros((num_resets, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        self.torques_sim[env_ids] = torch.zeros((num_resets, self.num_bodies, 3), dtype=torch.float32, device=self.device)

    def reset_command_idx(self, env_ids):
        """不同任务中，command的随机化内容、尺度不同，相互独立"""
        raise NotImplementedError("reset_command_idx must be implemented in subclass")

    def reset_command_condition(self):
        """
        resets modes (can be combined):
        1. copter_reset: reset the command if copter is reset
        2. time_index_reset: reset the command at specific time index (500)
        """
        copter_reset_buf = self.reset_buf

        # 在time_index=500时重置指令
        time_index_reset_buf = torch.zeros_like(self.reset_buf)
        time_index_mask = torch.isin(self.progress_buf, self.time_index)
        time_index_reset_buf[time_index_mask] = 1

        reset_env_buf = copter_reset_buf + time_index_reset_buf
        reset_env_ids = reset_env_buf.nonzero(as_tuple=False).squeeze(-1)

        return reset_env_ids

    #####################################################################
    ### =======================control functions=======================###
    #####################################################################
    def control_with_thrusts(self, reset_env_ids):
        """
        power to battery_voltage, battery_voltage + throttle to rotor_speed, rotor_speed to lift and other aerodynamic
        """

        # P_m = torch.mul(self.rotor_speed, self.rotor_torque).sum(dim=1).unsqueeze(-1)
        P_m = torch.sum(400 * (self.rotor_speed * 2 * torch.pi / 4500) ** 3, dim=1).unsqueeze(1)  # TODO 450的随机化
        self.battery_voltage = self.battery_dynamics.sim_process(P_m)
        self.rotor_speed = self.rotor_dynamics.sim_process(self.battery_voltage, self.throttle, self.rotor_speed)
        self.rotor_force, self.rotor_torque, self.body_force, self.body_torque = self.aero_dynamics.sim_process(self.copter_linvel_body, self.rotor_speed)
        rotor_force_sim, rotor_torque_sim = self.dynamics_real2sim.sim_process(self.rotor_force, self.rotor_torque)  # 输出的力、力矩全正，在下面设置方向

        self.forces_sim.zero_()
        self.torques_sim.zero_()
        # 基础的推力、力矩
        self.forces_sim[:, [2, 4, 6, 8], 2] = rotor_force_sim[:, [0, 1, 2, 3]]  # rotor2301
        self.torques_sim[:, [2, 4, 6, 8], 2] = rotor_torque_sim[:, [0, 1, 2, 3]]  # rotor2301
        # 高阶气动力
        self.forces_sim[:, 0, :] = self.body_force
        self.torques_sim[:, 0, :] = self.body_torque
        # 死掉的环境不给力，以下行代码的必要性存疑
        self.forces_sim[reset_env_ids] = 0.0
        self.torques_sim[reset_env_ids] = 0.0

        # apply force
        forces_tensor = gymtorch.unwrap_tensor(self.forces_sim)
        torques_tensor = gymtorch.unwrap_tensor(self.torques_sim)
        self.gym.apply_rigid_body_force_tensors(self.sim, forces_tensor, torques_tensor, gymapi.LOCAL_SPACE)

    def angular_vel_control(self, actions):
        """
        action0为油门，无单位
        action1-3为期望角速度，有单位
        """

        # 对网络输出进行反归一化，与真机保持一致
        temp = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        temp[:, 0] = (actions[:, 0] + 1) / 2 * 1000  # action=-1对应无油门，action=1对应满油门
        temp[:, 1:] = actions[:, 1:] * 20  # 转换为-20～20的期望角速度。线性转换

        self.u[:, 0] = temp[:, 0]
        self.u[:, 1:] = self.angvel_controller.compute(temp[:, 1:], self.copter_angvel_body)  # 利用机体系下的期望角速度与实际角速度，计算力矩
        self.throttle = self.dynamics_real2sim.control_allocator(self.u)

    #####################################################################
    ### ========================utils functions========================###
    #####################################################################
    def record(self):
        # record的频率与rl的频率相同，因此更高频的推力变化、转速变化无法被记录
        self.logger.store_buffer(
            # copter info
            copter_pos=self.copter_pos[0].cpu().numpy(),
            copter_quat=self.copter_quat[0].cpu().numpy(),
            copter_rpy=self.copter_rpy[0].cpu().numpy(),
            copter_rpy_continuous=self.copter_rpy_continuous[0].cpu().numpy(),
            copter_linvel=self.copter_linvel[0].cpu().numpy(),
            copter_angvel=self.copter_angvel[0].cpu().numpy(),
            copter_linvel_body=self.copter_linvel_body[0].cpu().numpy(),
            copter_angvel_body=self.copter_angvel_body[0].cpu().numpy(),
            # target info
            target_pos=self.target_pos[0].cpu().numpy(),
            target_quat=self.target_quat[0].cpu().numpy(),
            target_linvel=self.target_linvel[0].cpu().numpy(),
            target_angvel=self.target_angvel[0].cpu().numpy(),
            # relative info
            relative_pos=self.relative_pos[0].cpu().numpy(),
            relative_pos_body=self.relative_pos_body[0].cpu().numpy(),
            relative_quat_body=self.relative_quat_body[0].cpu().numpy(),
            relative_linvel=self.relative_linvel[0].cpu().numpy(),
            relative_angvel=self.relative_angvel[0].cpu().numpy(),
            relative_linvel_body=self.relative_linvel_body[0].cpu().numpy(),
            relative_angvel_body=self.relative_angvel_body[0].cpu().numpy(),
            # control info
            u=self.u[0].cpu().numpy(),
            throttle=self.throttle[0].cpu().numpy(),
            battery_voltage=self.battery_voltage[0].cpu().numpy(),
            rotor_speed=self.rotor_speed[0].cpu().numpy(),
            rotor_force=self.rotor_force[0].cpu().numpy(),
            rotor_torque=self.rotor_torque[0].cpu().numpy(),
            body_force=self.body_force[0].cpu().numpy(),
            body_torque=self.body_torque[0].cpu().numpy(),
            # command info
            command=self.command[0].cpu().numpy(),
            # rl info
            observations=self.obs_buf[0, -1].cpu().numpy(),  # 只保留最后一帧，不然会保存三维数组到csv从而出bug
            actions=self.actions[0].cpu().numpy(),
            actions_old=self.actions_old[0].cpu().numpy(),
            reward=self.rew_buf[0].cpu().numpy(),
            done=self.reset_buf[0].cpu().numpy(), )

    def rand_quat(self, num_sets, pitch_limit=torch.pi, roll_limit=torch.pi, yaw_limit=torch.pi):
        rand_pitch = torch_rand_float(-pitch_limit, pitch_limit, (num_sets, 1), self.device).flatten()
        rand_roll = torch_rand_float(-roll_limit, roll_limit, (num_sets, 1), self.device).flatten()
        rand_yaw = torch_rand_float(-yaw_limit, yaw_limit, (num_sets, 1), self.device).flatten()

        quats = quat_from_euler_xyz(rand_pitch, rand_roll, rand_yaw).squeeze()
        return quats


class FpvPos(FpvBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def compute_observation_state(self):
        super().compute_observation_state()
        self.obs_buf[:, -1, -2:] = self.command.squeeze().clone()
        self.states_buf[:, -1, -2:] = self.command.squeeze().clone()

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_pos_reward(
            self.relative_pos_body,
            self.copter_pos,
            self.copter_quat,
            self.target_quat,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_copter_idx(self, env_ids):
        """
        正常训练：开关分别控制位置、姿态、速度、电机转速的随机化
        """
        num_resets = len(env_ids)
        if self.random_copter_pos:
            self.copter_pos[env_ids, 0] = torch_rand_float(-2, 2, (num_resets, 1), self.device).flatten()
            self.copter_pos[env_ids, 1] = torch_rand_float(-2, 2, (num_resets, 1), self.device).flatten()
            self.copter_pos[env_ids, 2] = 2.5 + torch_rand_float(-2, 2, (num_resets, 1), self.device).flatten()
        else:
            self.copter_pos[env_ids, 0] = 0
            self.copter_pos[env_ids, 1] = 0
            self.copter_pos[env_ids, 2] = 2.5

        if self.random_copter_quat:
            self.copter_quat[env_ids, :] = self.rand_quat(num_resets, torch.pi, torch.pi, torch.pi)
        else:
            self.copter_quat[env_ids, 0:3] = 0
            self.copter_quat[env_ids, 3] = 1

        if self.random_copter_vel:
            self.copter_linvel[env_ids, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
            self.copter_angvel[env_ids, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        else:
            self.copter_linvel[env_ids, :] = 0
            self.copter_angvel[env_ids, :] = 0

        self.copter_rpy[env_ids, 0], self.copter_rpy[env_ids, 1], self.copter_rpy[env_ids, 2] = get_euler_xyz_v1(self.copter_quat[env_ids])
        self.copter_rpy_old[env_ids] = self.copter_rpy[env_ids].clone()  # 上时刻rpy，用于与copter_rpy_continuous搭配使用
        self.copter_rpy_continuous[env_ids] = self.copter_rpy[env_ids].clone()  # 不约束范围，实现连续的rpy

        return self.all_actor_indices[env_ids, 0].flatten()

    def reset_command_idx(self, env_ids):
        self.command[env_ids, :] = 0  # pos任务的id为0，没有指令


class FpvRotate(FpvBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def compute_observation_state(self):
        super().compute_observation_state()
        self.obs_buf[:, -1, -2] = self.command[:, 0].squeeze().clone()
        self.obs_buf[:, -1, -1] = self.command[:, 1].squeeze().clone() / 6  # -6-6的期望线速度转换到 -1 - 1
        self.states_buf[:, -1, -2] = self.command[:, 0].squeeze().clone()
        self.states_buf[:, -1, -1] = self.command[:, 1].squeeze().clone() / 6  # -6-6的期望线速度转换到 -1 - 1=

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_rotating_reward(
            self.relative_pos,
            self.relative_linvel,
            self.copter_pos,
            self.copter_quat,
            self.command,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_copter_idx(self, env_ids):
        """
        正常训练：开关分别控制位置、姿态、速度、电机转速的随机化
        """
        num_resets = len(env_ids)
        if self.random_copter_pos:  # z方向初始随机化范围更小，xy范围更大，因为并不是初始化在原点难度最低
            self.copter_pos[env_ids, 0:2] = torch_rand_float(-2, 2, (num_resets, 2), self.device)
            self.copter_pos[env_ids, 2] = 2.5 + torch_rand_float(-2, 2, (num_resets, 1), self.device).flatten()
        else:
            self.copter_pos[env_ids, 0:2] = torch_rand_float(-0.5, 0.5, (num_resets, 2), self.device)
            self.copter_pos[env_ids, 2] = 2.5

        if self.random_copter_quat:
            self.copter_quat[env_ids, :] = self.rand_quat(num_resets, torch.pi, torch.pi, torch.pi)
        else:
            self.copter_quat[env_ids, 0:3] = 0
            self.copter_quat[env_ids, 3] = 1

        if self.random_copter_vel:
            self.copter_linvel[env_ids, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
            self.copter_angvel[env_ids, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        else:
            self.copter_linvel[env_ids, :] = 0
            self.copter_angvel[env_ids, :] = 0

        self.copter_rpy[env_ids, 0], self.copter_rpy[env_ids, 1], self.copter_rpy[env_ids, 2] = get_euler_xyz_v1(self.copter_quat[env_ids])
        self.copter_rpy_old[env_ids] = self.copter_rpy[env_ids].clone()  # 上时刻rpy，用于与copter_rpy_continuous搭配使用
        self.copter_rpy_continuous[env_ids] = self.copter_rpy[env_ids].clone()  # 不约束范围，实现连续的rpy

        return self.all_actor_indices[env_ids, 0].flatten()

    def reset_command_idx(self, env_ids):
        num_resets = len(env_ids)
        if self.random_command:
            self.command[env_ids, 0] = 1  # rotate任务的id为0
            self.command[env_ids, 1] = torch_rand_float(-6, 6, (num_resets, 1), self.device).squeeze()  # y方向期望线速度为-6到-6
        else:
            self.command[env_ids, 0] = 1
            self.command[env_ids, 1] = 1  # y方向期望线速度为1


class FpvFlip(FpvBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        self.flip_radian = torch.zeros_like(self.reset_buf, dtype=torch.float32, device=self.device)  # 余下有待执行的flip的弧度

    def compute_observation_state(self):
        self.command[:, 1] = self.flip_radian - self.copter_rpy_continuous[:, 0]  # 任务参数为roll待旋转的角度
        self.command[:, 1] = torch.clamp(self.command[:, 1], min=-2 * torch.pi, max=2 * torch.pi)

        super().compute_observation_state()
        self.obs_buf[:, -1, -2] = self.command[:, 0].squeeze().clone()
        self.obs_buf[:, -1, -1] = self.command[:, 1].squeeze().clone() / 2 / torch.pi
        self.states_buf[:, -1, -2] = self.command[:, 0].squeeze().clone()
        self.states_buf[:, -1, -1] = self.command[:, 1].squeeze().clone() / 2 / torch.pi
        # print(self.command[:, 1])

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_flip_reward(
            self.relative_pos_body,
            self.relative_quat_body,
            self.copter_pos,
            self.command,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_copter_idx(self, env_ids):
        """
        正常训练：开关分别控制位置、姿态、速度、电机转速的随机化
        """
        num_resets = len(env_ids)
        if self.random_copter_pos:  # z方向初始随机化范围更小，xy范围更大，因为并不是初始化在原点难度最低
            self.copter_pos[env_ids, 0:2] = torch_rand_float(-0.5 - 1.5 * self.difficulty, 0.5 + 1.5 * self.difficulty, (num_resets, 2), self.device)
            self.copter_pos[env_ids, 2] = 3 + (self.difficulty * torch_rand_float(-2, 2, (num_resets, 1), self.device).flatten())
            # self.copter_pos[env_ids, 0] = 0
        else:
            self.copter_pos[env_ids, 0:2] = torch_rand_float(-0.5, 0.5, (num_resets, 2), self.device)
            self.copter_pos[env_ids, 2] = 3

        if self.random_copter_quat:
            self.copter_quat[env_ids, :] = self.rand_quat(num_resets, torch.pi, 0, 0)
        else:
            self.copter_quat[env_ids, 0:3] = 0
            self.copter_quat[env_ids, 3] = 1

        if self.random_copter_vel:
            self.copter_linvel[env_ids, :] = torch_rand_float(-3 * self.difficulty, 3 * self.difficulty, (num_resets, 3), self.device)

            # self.copter_angvel[env_ids, 0] = 10  # 单向
            temp = torch.rand(self.num_envs, device=self.device)
            temp[temp < 1 / 2] = -1
            temp[temp >= 1 / 2] = 1
            self.copter_angvel[env_ids, 0] = 10 * temp[env_ids]  # 双向
        else:
            self.copter_linvel[env_ids, :] = 0

        self.copter_rpy[env_ids, 0], self.copter_rpy[env_ids, 1], self.copter_rpy[env_ids, 2] = get_euler_xyz_v1(self.copter_quat[env_ids])
        self.copter_rpy_old[env_ids] = self.copter_rpy[env_ids].clone()  # 上时刻rpy，用于与copter_rpy_continuous搭配使用
        self.copter_rpy_continuous[env_ids] = self.copter_rpy[env_ids].clone()  # 不约束范围，实现连续的rpy

        return self.all_actor_indices[env_ids, 0].flatten()

    def reset_command_idx(self, env_ids):
        # 时间到了就触发，随机产生要求无人机绕x进行flip的圈数
        time_mask = torch.isin(self.progress_buf, self.time_index)
        time_env_ids = time_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(time_env_ids) > 0:
            # self.flip_radian[mask] += 2 * torch.pi  # 单向
            temp_rand = torch.rand(self.num_envs, device=self.device)  # 均匀分布
            temp = torch.zeros_like(temp_rand, device=self.device)
            temp[temp_rand < 1 / 8] = -3
            temp[(temp_rand >= 1 / 8) & (temp_rand < 2 / 8)] = -2
            temp[(temp_rand >= 2 / 8) & (temp_rand < 3 / 8)] = -1
            temp[(temp_rand >= 3 / 8) & (temp_rand < 5 / 8)] = 0
            temp[(temp_rand >= 5 / 8) & (temp_rand < 6 / 8)] = 1
            temp[(temp_rand >= 6 / 8) & (temp_rand < 7 / 8)] = 2
            temp[temp_rand >= 7 / 8] = 3
            self.flip_radian[time_env_ids] += 2 * torch.pi * temp[time_env_ids]  # 双向

        # 环境初始化时，随机产生要求无人机绕x进行flip的圈数
        copter_reset_buf = self.reset_buf
        copter_env_ids = copter_reset_buf.nonzero(as_tuple=False).squeeze(-1)  # 情况1
        if len(copter_env_ids) > 0:
            # self.flip_radian[env_ids] = 0
            # self.flip_radian[env_ids] = 2 * torch.pi  # 单向
            temp = torch.rand(self.num_envs, device=self.device)
            temp[temp < 1 / 3] = -1
            temp[(temp >= 1 / 3) & (temp < 2 / 3)] = 0
            temp[temp >= 2 / 3] = 1
            self.flip_radian[copter_env_ids] = torch.where(self.copter_angvel[copter_env_ids, 0] > 5, 2 * torch.pi, - 2 * torch.pi)  # 双向
            # self.flip_radian = torch.clamp(self.flip_radian, min=-4 * torch.pi, max=4 * torch.pi)  # 最多连续转两圈
            # print(self.flip_radian[0])

        self.command[:, 0] = -1  # flip任务的id为-1


class FpvMix(FpvBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        n1 = int(self.num_envs / 3 * 1)
        n2 = int(self.num_envs / 3 * 2)
        self.task_group_index = [0, n1, n2, self.num_envs]
        self.flip_radian = torch.zeros_like(self.reset_buf, dtype=torch.float32, device=self.device)

    def compute_observation_state(self):
        self.command[self.task_group_index[2]:self.task_group_index[3], -1] = self.flip_radian[self.task_group_index[2]:self.task_group_index[3]] - self.copter_rpy_continuous[self.task_group_index[2]:self.task_group_index[3], 0]
        self.command[self.task_group_index[2]:self.task_group_index[3], -1] = torch.clamp(self.command[self.task_group_index[2]:self.task_group_index[3], -1], min=-2 * torch.pi, max=2 * torch.pi)

        super().compute_observation_state()
        # pos
        self.obs_buf[self.task_group_index[0]:self.task_group_index[1], -1, -2:] = self.command[self.task_group_index[0]:self.task_group_index[1]].squeeze().clone()
        self.states_buf[self.task_group_index[0]:self.task_group_index[1], -1, -2:] = self.command[self.task_group_index[0]:self.task_group_index[1]].squeeze().clone()
        # rotate
        self.obs_buf[self.task_group_index[1]:self.task_group_index[2], -1, -2] = self.command[self.task_group_index[1]:self.task_group_index[2], -2].squeeze().clone()
        self.obs_buf[self.task_group_index[1]:self.task_group_index[2], -1, -1] = self.command[self.task_group_index[1]:self.task_group_index[2], -1].squeeze().clone() / 6
        self.states_buf[self.task_group_index[1]:self.task_group_index[2], -1, -2] = self.command[self.task_group_index[1]:self.task_group_index[2], -2].squeeze().clone()
        self.states_buf[self.task_group_index[1]:self.task_group_index[2], -1, -1] = self.command[self.task_group_index[1]:self.task_group_index[2], -1].squeeze().clone() / 6
        # flip
        self.obs_buf[self.task_group_index[2]:self.task_group_index[3], -1, -2] = self.command[self.task_group_index[2]:self.task_group_index[3], -2].squeeze().clone()
        self.obs_buf[self.task_group_index[2]:self.task_group_index[3], -1, -1] = self.command[self.task_group_index[2]:self.task_group_index[3], -1].squeeze().clone() / 2 / torch.pi
        self.states_buf[self.task_group_index[2]:self.task_group_index[3], -1, -2] = self.command[self.task_group_index[2]:self.task_group_index[3], -2].squeeze().clone()
        self.states_buf[self.task_group_index[2]:self.task_group_index[3], -1, -1] = self.command[self.task_group_index[2]:self.task_group_index[3], -1].squeeze().clone() / 2 / torch.pi

    def compute_reward(self):
        # pos
        (self.rew_buf[self.task_group_index[0]:self.task_group_index[1]],
         self.reset_buf[self.task_group_index[0]:self.task_group_index[1]]) = compute_pos_reward(
            self.relative_pos_body[self.task_group_index[0]:self.task_group_index[1]],
            self.copter_pos[self.task_group_index[0]:self.task_group_index[1]],
            self.copter_quat[self.task_group_index[0]:self.task_group_index[1]],
            self.target_quat[self.task_group_index[0]:self.task_group_index[1]],
            self.reset_buf[self.task_group_index[0]:self.task_group_index[1]],
            self.progress_buf[self.task_group_index[0]:self.task_group_index[1]],
            self.max_episode_length)
        # rotate
        (self.rew_buf[self.task_group_index[1]:self.task_group_index[2]],
         self.reset_buf[self.task_group_index[1]:self.task_group_index[2]]) = compute_rotating_reward(
            self.relative_pos[self.task_group_index[1]:self.task_group_index[2]],
            self.relative_linvel[self.task_group_index[1]:self.task_group_index[2]],
            self.copter_pos[self.task_group_index[1]:self.task_group_index[2]],
            self.copter_quat[self.task_group_index[1]:self.task_group_index[2]],
            self.command[self.task_group_index[1]:self.task_group_index[2]],
            self.reset_buf[self.task_group_index[1]:self.task_group_index[2]],
            self.progress_buf[self.task_group_index[1]:self.task_group_index[2]],
            self.max_episode_length)
        # flip
        (self.rew_buf[self.task_group_index[2]:self.task_group_index[3]],
         self.reset_buf[self.task_group_index[2]:self.task_group_index[3]]) = compute_flip_reward(
            self.relative_pos_body[self.task_group_index[2]:self.task_group_index[3]],
            self.relative_quat_body[self.task_group_index[2]:self.task_group_index[3]],
            self.copter_pos[self.task_group_index[2]:self.task_group_index[3]],
            self.command[self.task_group_index[2]:self.task_group_index[3]],
            self.reset_buf[self.task_group_index[2]:self.task_group_index[3]],
            self.progress_buf[self.task_group_index[2]:self.task_group_index[3]],
            self.max_episode_length)

    def reset_copter_idx(self, env_ids):
        env_ids_list = env_ids.tolist()
        idx1 = bisect.bisect_left(env_ids_list, self.task_group_index[1])
        idx2 = bisect.bisect_left(env_ids_list, self.task_group_index[2])
        env_ids0 = env_ids_list[:idx1]
        env_ids1 = env_ids_list[idx1:idx2]
        env_ids2 = env_ids_list[idx2:]
        num_resets0 = len(env_ids0)
        num_resets1 = len(env_ids1)
        num_resets2 = len(env_ids2)

        # pos
        if self.random_copter_pos:
            self.copter_pos[env_ids0, 0:2] = torch_rand_float(-2, 2, (num_resets0, 2), self.device)
            self.copter_pos[env_ids0, 2] = 2.5 + torch_rand_float(-2, 2, (num_resets0, 1), self.device).flatten()
        else:
            self.copter_pos[env_ids0, 0:2] = 0
            self.copter_pos[env_ids0, 2] = 2.5
        if self.random_copter_quat:
            self.copter_quat[env_ids0, :] = self.rand_quat(num_resets0, torch.pi, torch.pi, torch.pi)
        else:
            self.copter_quat[env_ids0, 0:3] = 0
            self.copter_quat[env_ids0, 3] = 1
        if self.random_copter_vel:
            self.copter_linvel[env_ids0, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets0, 3), self.device)
            self.copter_angvel[env_ids0, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets0, 3), self.device)
        else:
            self.copter_linvel[env_ids0, :] = 0
            self.copter_angvel[env_ids0, :] = 0

        # rotate    
        if self.random_copter_pos:
            self.copter_pos[env_ids1, 0:2] = torch_rand_float(-2, 2, (num_resets1, 2), self.device)
            self.copter_pos[env_ids1, 2] = 2.5 + torch_rand_float(-2, 2, (num_resets1, 1), self.device).flatten()
        else:
            self.copter_pos[env_ids1, 0:2] = 0
            self.copter_pos[env_ids1, 2] = 2.5
        if self.random_copter_quat:
            self.copter_quat[env_ids1, :] = self.rand_quat(num_resets1, torch.pi, torch.pi, torch.pi)
        else:
            self.copter_quat[env_ids1, 0:3] = 0
            self.copter_quat[env_ids1, 3] = 1
        if self.random_copter_vel:
            self.copter_linvel[env_ids1, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets1, 3), self.device)
            self.copter_angvel[env_ids1, :] = 3 * torch_rand_float(-1.0, 1.0, (num_resets1, 3), self.device)
        else:
            self.copter_linvel[env_ids1, :] = 0
            self.copter_angvel[env_ids1, :] = 0

        # flip
        if self.random_copter_pos:
            self.copter_pos[env_ids2, 0:2] = torch_rand_float(-2, 2, (num_resets2, 2), self.device)
            self.copter_pos[env_ids2, 2] = 2.5 + torch_rand_float(-2, 2, (num_resets2, 1), self.device).flatten()
        else:
            self.copter_pos[env_ids2, 0:2] = 0
            self.copter_pos[env_ids2, 2] = 2.5
        if self.random_copter_quat:
            self.copter_quat[env_ids2, :] = self.rand_quat(num_resets2, torch.pi, 0, 0)
        else:
            self.copter_quat[env_ids2, 0:3] = 0
            self.copter_quat[env_ids2, 3] = 1
        if self.random_copter_vel:
            self.copter_linvel[env_ids2, :] = torch_rand_float(-3 * self.difficulty, 3 * self.difficulty, (num_resets2, 3), self.device)
            temp = torch.rand(self.num_envs, device=self.device)
            temp[temp < 1 / 2] = -1
            temp[temp >= 1 / 2] = 1
            self.copter_angvel[env_ids2, 0] = 10 * temp[env_ids2]
        else:
            self.copter_linvel[env_ids2, :] = 0
            self.copter_angvel[env_ids2, :] = 0

        self.copter_rpy[env_ids, 0], self.copter_rpy[env_ids, 1], self.copter_rpy[env_ids, 2] = get_euler_xyz_v1(self.copter_quat[env_ids])
        self.copter_rpy_old[env_ids] = self.copter_rpy[env_ids].clone()
        self.copter_rpy_continuous[env_ids] = self.copter_rpy[env_ids].clone()

        return self.all_actor_indices[env_ids, 0].flatten()

    def reset_command_idx(self, env_ids):
        env_ids_list = env_ids.tolist()
        idx1 = bisect.bisect_left(env_ids_list, self.task_group_index[1])
        idx2 = bisect.bisect_left(env_ids_list, self.task_group_index[2])
        idx3 = bisect.bisect_left(env_ids_list, self.task_group_index[3])
        env_ids0 = env_ids_list[:idx1]
        env_ids1 = env_ids_list[idx1:idx2]
        env_ids2 = env_ids_list[idx2:]

        # POS task command
        if len(env_ids0) > 0:
            self.command[env_ids0, :] = 0

        # ROTATE task command
        if len(env_ids1) > 0:
            if self.random_command:
                self.command[env_ids1, -2] = 1
                self.command[env_ids1, -1] = torch_rand_float(-6, 6, (len(env_ids1), 1), self.device).squeeze()
            else:
                self.command[env_ids1, -2] = 1
                self.command[env_ids1, -1] = 1

        # FLIP task command - 与FpvFlip相同的逻辑，但只处理env_ids2中的环境
        if len(env_ids2) > 0:
            # 时间到了就触发，随机产生要求无人机绕x进行flip的圈数
            time_mask = torch.isin(self.progress_buf, self.time_index)
            time_env_ids = time_mask.nonzero(as_tuple=False).squeeze(-1)
            flip_time_mask = torch.isin(time_env_ids, torch.tensor(env_ids2, device=self.device))
            flip_time_env_ids = time_env_ids[flip_time_mask]
            if len(flip_time_env_ids) > 0:
                temp_rand = torch.rand(self.num_envs, device=self.device)  # 均匀分布
                temp = torch.zeros_like(temp_rand, device=self.device)
                temp[temp_rand < 1 / 8] = -3
                temp[(temp_rand >= 1 / 8) & (temp_rand < 2 / 8)] = -2
                temp[(temp_rand >= 2 / 8) & (temp_rand < 3 / 8)] = -1
                temp[(temp_rand >= 3 / 8) & (temp_rand < 5 / 8)] = 0
                temp[(temp_rand >= 5 / 8) & (temp_rand < 6 / 8)] = 1
                temp[(temp_rand >= 6 / 8) & (temp_rand < 7 / 8)] = 2
                temp[temp_rand >= 7 / 8] = 3
                self.flip_radian[flip_time_env_ids] += 2 * torch.pi * temp[flip_time_env_ids]  # 双向

            # 环境初始化时，随机产生要求无人机绕x进行flip的圈数
            copter_reset_buf = self.reset_buf
            copter_env_ids = copter_reset_buf.nonzero(as_tuple=False).squeeze(-1)
            flip_copter_mask = torch.isin(copter_env_ids, torch.tensor(env_ids2, device=self.device))
            flip_copter_env_ids = copter_env_ids[flip_copter_mask]
            if len(flip_copter_env_ids) > 0:
                temp = torch.rand(self.num_envs, device=self.device)
                temp[temp < 1 / 3] = -1
                temp[(temp >= 1 / 3) & (temp < 2 / 3)] = 0
                temp[temp >= 2 / 3] = 1
                self.flip_radian[flip_copter_env_ids] = torch.where(self.copter_angvel[flip_copter_env_ids, 0] > 5, 2 * torch.pi, - 2 * torch.pi)  # 双向

            # 更新FLIP任务的command。维度1需要每个时刻都更新，因此放在compute observation里面
            self.command[env_ids2, -2] = -1
