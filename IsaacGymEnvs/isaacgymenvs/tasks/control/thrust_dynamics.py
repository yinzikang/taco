#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用一阶系统来仿真电机模型
仿真悬停油门：276.6
实际悬停油门：小于300

需要使用时，首先创建实例，然后必须进行reset，才能正常仿真
有些设置只有在reset中才能给定

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/3/5 下午10:16   yinzikang      1.0         None
"""
import numpy as np
import matplotlib.pyplot as plt
from isaacgym.torch_utils import torch_rand_float
import torch


class RotorDynamics:
    """
    油门指令、电池电压 与 电机稳态转速 的关系
    use_duct：是否使用涵道
    rotor_mode：稳态转速的计算方式：分为仅与电压相关、四次拟合、nature拟合
        四次拟合(omega_para_mode0)：x为油门，y为电压：omega = p00 + p10x + p01y + p20x^2 + p11xy + p30x^3 + p21x^2y + p40x^4 + p31x^3*y
    """

    def __init__(self, num_envs, device='cuda:0', response_time_init=0.017):
        self.num_envs = num_envs
        self.device = device

        # 以下量在reset中被设置
        # 采样时间
        self.sample_time = 0.001  # 不再支持外部修改
        # 响应时间
        self.response_time_init = response_time_init
        self.response_time = response_time_init * torch.ones((self.num_envs, 4), device=self.device)  # 响应时间，每个episode更新一次
        # 延迟时间，以下不再使用
        self.delay_time = 0  # 延迟在reset中被赋值、修改，即每个episode更新一次
        self.delay_idx = 0  # 在reset中被赋值、修改，即每个episode更新一次
        # 噪声
        self.rotor_noise = False  # 是否考虑转速噪声
        # 转速buffer
        self.omega_buf = torch.zeros((self.num_envs, 4, self.delay_idx + 1), device=self.device)  # 每个episode更新一次

        omega_para = torch.tensor([0, 12.9466, 0.1872, -5.1220, 0.5906], device=self.device).expand(num_envs, 5)
        self.omega_para_init, self.omega_para = omega_para.clone(), omega_para.clone()

        throttle_para = torch.tensor([-0.0035, -0.0171, 0.0731, 0.0166, -0.0103, 0.0046], device=self.device).expand(num_envs, 6)
        self.throttle_para_init, self.throttle_para = throttle_para.clone(), throttle_para.clone()

    def throttle_voltage2omega(self, battery_voltage, throttle):
        """
        归一化油门、电池电压到稳态转速的映射
        """
        throttle_one = throttle / 1000
        battery_voltage_one = (battery_voltage - 23) / 3

        omega_target = (self.omega_para[:, 0].unsqueeze(-1) * torch.ones_like(throttle_one, device=self.device) +
                        self.omega_para[:, 1].unsqueeze(-1) * throttle_one +
                        self.omega_para[:, 2].unsqueeze(-1) * battery_voltage_one +
                        self.omega_para[:, 3].unsqueeze(-1) * throttle_one ** 2 +
                        self.omega_para[:, 4].unsqueeze(-1) * throttle_one * battery_voltage_one) * 100


        return omega_target

    def omega_noise(self, omega):
        """
        电机的转速变化通常无法稳定，越大的转速由于空气阻力与油门的问题通常会产生一定的波动
        实际转速为期望转速按比例变化
        """
        if self.rotor_noise:
            ratio = 10 / 700  # 50/700 standard
            omega_noised = omega * torch_rand_float(1 - ratio, 1 + ratio, (self.num_envs, 4), self.device)
        else:
            omega_noised = omega
        return omega_noised

    def omega_compute(self, omega_target, omega_current):
        """
        仿真转速响应：一阶系统的利用前向差分的离散化模型，转速的变化率正比于转速的偏差
        """
        omega_response = omega_current + self.sample_time / self.response_time * (omega_target - omega_current)  # 这个是绝对正确的，不要该了

        return omega_response

    def omega_delay(self, omega):
        """
        仿真action到pwm的延迟，所有电机延迟相同
        """
        self.omega_buf[:, :, :-1] = self.omega_buf[:, :, 1:]
        self.omega_buf[:, :, -1] = omega
        delayed_output = self.omega_buf[:, :, -1 - self.delay_idx]

        return delayed_output

    def sim_process(self, battery_voltage, throttle, omega_current):
        omega_target = self.throttle_voltage2omega(battery_voltage, throttle)
        omega_response = self.omega_compute(omega_target, omega_current)
        omega_response_delayed = self.omega_delay(omega_response)
        omega_response_delayed_noised = self.omega_noise(omega_response_delayed)

        return omega_response_delayed_noised

    def get_omega(self):
        return self.omega_buf[:, :, -1 - self.delay_idx]

    def reset(self, reset_env_ids, difficulty=0, rotor_noise=True, rotor_delay=True, rotor_response=True,
              random_coe=True, random_rotor_delay=True, random_rotor_response=True, random_motor_speed=True):
        """
        重置，并随机化拟合参数
        """
        num_resets = len(reset_env_ids)

        self.rotor_noise = rotor_noise

        if random_coe:
            self.omega_para[reset_env_ids] = self.omega_para_init[reset_env_ids] * torch_rand_float(1 - 0.05 * difficulty, 1 + 0.05 * difficulty, self.omega_para[reset_env_ids].shape, self.device)
            self.throttle_para[reset_env_ids] = self.throttle_para_init[reset_env_ids] * torch_rand_float(1 - 0.05 * difficulty, 1 + 0.05 * difficulty, self.throttle_para[reset_env_ids].shape, self.device)
        else:
            self.omega_para[reset_env_ids] = self.omega_para_init[reset_env_ids]
            self.throttle_para[reset_env_ids] = self.throttle_para_init[reset_env_ids]

        if rotor_delay:  # 用不到了，现在所有延迟都放在了一起
            if random_rotor_delay:
                self.delay_time = 0.00  # todo 现在还不支持随机延迟
            else:
                self.delay_time = 0.00
        else:
            self.delay_time = 0
        self.delay_idx = int(self.delay_time / self.sample_time)
        self.omega_buf[reset_env_ids, :, :] = torch.zeros((num_resets, 4, self.delay_idx + 1), device=self.device)

        if rotor_response:  # 电机响应时间
            if random_rotor_response:
                self.response_time[reset_env_ids, :] = torch_rand_float(self.response_time_init - 0.001, self.response_time_init + 0.001, (num_resets, 4), self.device)
            else:
                self.response_time[reset_env_ids, :] = self.response_time_init * torch.ones((num_resets, 4), device=self.device)
        else:
            self.response_time[reset_env_ids, :] = self.sample_time * torch.ones((num_resets, 4), device=self.device)  # 最小响应时间 等于 采样时间

        if random_motor_speed:
            self.omega_buf[reset_env_ids, :, :] = torch_rand_float(0, 400, (num_resets, 4), self.device).unsqueeze(-1)  # 把整个buffer随机化为某一个转速
        else:
            self.omega_buf[reset_env_ids, :, :] = 0

        return self.omega_buf[reset_env_ids, :, -1 - self.delay_idx]


class AeroDynamics:
    def __init__(self, num_envs, device='cuda:0'):
        self.num_envs = num_envs
        self.device = device

        para_force_torque = torch.tensor([1.13e-05, 0.05], device=self.device).expand(num_envs, 2)
        para_d = torch.tensor([-0.386, -0.53], device=self.device).expand(num_envs, 2)  # rotor drag与速度线性相关，测试结果与文章基本一致
        # para_d = torch.tensor([-0.005527, -0.005527, 0, 0], device=self.device).expand(num_envs, 4)  # rotor drag与速度平方相关
        para_t = torch.tensor([0.009], device=self.device).expand(num_envs, 1)
        # para_t = torch.tensor([-2.181e-8 * 3600, 1.429e-4], device=self.device).expand(num_envs, 1)

        # 升力与力矩参数，分别为电机转速与螺旋桨产生的升力的关系、螺旋桨产生的力矩与升力的比例
        self.para_force_torque, self.para_force_torque_init = para_force_torque.clone(), para_force_torque.clone()
        # rotor drag coefficient
        self.para_d, self.para_d_init = para_d.clone(), para_d.clone()
        # thrust model
        self.para_t, self.para_t_init = para_t.clone(), para_t.clone()

    def thrust2omega(self, thrust):
        # 推力转换为转速
        return torch.sqrt(thrust / self.para_force_torque[:, 0].unsqueeze(1))

    def sim_process(self, copter_linvel_body, rotor_speed):
        """
        根据不同的mode，输出不同的气动力计算结果
        结果的参考系均为机体系，isaac gym同样也按照机体系施加外力
        与UZH的Flightmare一致，考虑四个电机上的升力与力矩，以及与速度、转速平方挂钩的其他气动力
        rotor drag：机体上，阻止无人机水平方向移动的阻尼
        thrust model：旋翼上，阻止无人机竖直方向移动的阻尼
        todo:当前未考虑电机转速发生变化所引发的力矩
        """
        rotor_force = self.para_force_torque[:, 0].unsqueeze(-1) * rotor_speed * rotor_speed
        rotor_torque = self.para_force_torque[:, 1].unsqueeze(-1) * rotor_force
        body_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        body_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        # rotor drag
        body_force[:, 0] = self.para_d[:, 0] * copter_linvel_body[:, 0]
        body_force[:, 1] = self.para_d[:, 1] * copter_linvel_body[:, 1]
        # body_force[:, 0] = self.para_d[:, 0] * copter_linvel_body[:, 0] * abs(copter_linvel_body[:, 0])
        # body_force[:, 1] = self.para_d[:, 1] * copter_linvel_body[:, 1] * abs(copter_linvel_body[:, 1])

        # thrust model of every single rotor
        v_xy = torch.norm(copter_linvel_body[:, :2], dim=1)
        body_force[:, 2] = self.para_t[:, 0] * v_xy * v_xy
        # body_force[:, 2] = (self.para_t[:, 0].unsqueeze(-1) * copter_linvel_body[:, 2].unsqueeze(-1) * rotor_speed +
        #                     self.para_t[:, 1].unsqueeze(-1) * (v_xy * v_xy).unsqueeze(-1))

        return rotor_force, rotor_torque, body_force, body_torque

    def reset(self, reset_env_ids, random_coe=False, difficulty=0):
        """
        重置，并随机化拟合参数
        """
        num_resets = len(reset_env_ids)
        if random_coe:  # 推力与力矩参数满准的，随机化范围可以小一点
            self.para_force_torque[reset_env_ids] = self.para_force_torque_init[reset_env_ids] * torch_rand_float(1 - 0.05 * difficulty, 1 + 0.05 * difficulty,
                                                                                                                  self.para_force_torque[reset_env_ids].shape, self.device)
            self.para_d[reset_env_ids] = self.para_d_init[reset_env_ids] * torch_rand_float(1 - 0.05 * difficulty, 1 + 0.05 * difficulty, self.para_d[reset_env_ids].shape, self.device)
            self.para_t[reset_env_ids] = self.para_t_init[reset_env_ids] * torch_rand_float(1 - 0.05 * difficulty, 1 + 0.05 * difficulty, self.para_t[reset_env_ids].shape, self.device)


if __name__ == '__main__':
    device = 'cuda:0'
    num_envs = 2
    sample_time = 0.001

    rotor_dynamics = RotorDynamics(num_envs, device=device, response_time_init=0.05)
    aero_dynamics = AeroDynamics(num_envs, device=device)

    turns = 3
    time = 500

    for i in range(turns):
        rotor_speed = rotor_dynamics.reset([0, 1], difficulty=1, rotor_noise=False, rotor_delay=False, rotor_response=True,
                                           random_coe=False, random_rotor_delay=False, random_rotor_response=False, random_motor_speed=False)
        aero_dynamics.reset([0, 1], random_coe=False, difficulty=1)

        # battery_voltage = 4.25 * 6
        battery_voltage = 24.2
        copter_linvel_body = torch.zeros(num_envs, 3, device=device)

        t_buffer = []
        rotor_speed_buffer = []
        rotor_force_buffer = []
        rotor_torque_buffer = []
        throttle_buffer = []

        throttle = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        rotor_force = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        rotor_torque = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)

        rotor_force, rotor_torque, body_force, body_torque = aero_dynamics.sim_process(copter_linvel_body, rotor_speed)
        t_buffer.append(sample_time * -1)
        rotor_speed_buffer.append(rotor_speed[0].squeeze().cpu().numpy().tolist())
        rotor_force_buffer.append(rotor_force[0].squeeze().cpu().numpy().tolist())
        throttle_buffer.append(throttle[0].squeeze().cpu().numpy().tolist())

        for j in range(time):
            # throttle[:, 0] += 1
            # throttle[:, 1] += 2
            # throttle[:, 2] += 5
            # throttle[:, 3] += 10
            throttle[:, :] = 100
            throttle = throttle.clip(0, 300)

            rotor_speed = rotor_dynamics.sim_process(battery_voltage, throttle, rotor_speed)
            rotor_force, rotor_torque, body_force, body_torque = aero_dynamics.sim_process(copter_linvel_body, rotor_speed)

            t_buffer.append(sample_time * j)
            rotor_speed_buffer.append(rotor_speed[0].squeeze().cpu().numpy().tolist())
            rotor_force_buffer.append(rotor_force[0].squeeze().cpu().numpy().tolist())
            throttle_buffer.append(throttle[0].squeeze().cpu().numpy().tolist())

        t_buffer = np.array(t_buffer)
        rotor_speed_buffer = np.array(rotor_speed_buffer)
        rotor_force_buffer = np.array(rotor_force_buffer)
        rotor_torque_buffer = np.array(rotor_torque_buffer)
        throttle_buffer = np.array(throttle_buffer)

        plt.figure()
        plt.grid()
        plt.plot(t_buffer, rotor_speed_buffer, '.')
        plt.xlabel('time')
        plt.ylabel('rotor_speed')
        plt.title('rotor speed')

        # plt.figure()
        # plt.grid()
        # plt.plot(t_buffer, rotor_force_buffer, '.')
        # plt.title('rotor force')
        # plt.xlabel('time')
        # plt.ylabel('rotor_force')
        #
        # plt.figure()
        # plt.grid()
        # plt.plot(throttle_buffer, rotor_speed_buffer, '.')
        # plt.title('rotor speed')
        # plt.xlabel('throttle')
        # plt.ylabel('rotor_speed')
        #
        # plt.figure()
        # plt.grid()
        # plt.plot(throttle_buffer, rotor_force_buffer, '.')
        # plt.title('rotor force')
        # plt.xlabel('throttle')
        # plt.ylabel('rotor_force')

        plt.show()
