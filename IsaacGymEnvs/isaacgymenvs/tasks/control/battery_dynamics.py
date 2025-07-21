"""
battery model that simulate the voltage change
params are from UZH

V1.0 Wzk
"""
import numpy as np
import matplotlib.pyplot as plt
from isaacgym.torch_utils import torch_rand_float
import torch


class Battery_Dynamics():
    def __init__(self, num_envs, device, battery_dynamics=False, sample_time=0.01) -> None:
        self.device = device
        self.battery_dynamics = battery_dynamics
        self.dt = sample_time

        self.a = [4.35, -0.1102178, 0.0103368, -4.3778e-4]  # 4.35V LiHv; 4.2V Lipo
        self.a0 = torch.ones((num_envs, 1), device=self.device) * self.a[0]
        self.b = [0.0015778, -7.7608e-5, 0.0069498]
        self.R_min = 4.5
        self.k = 0.00104846
        self.tau_rc = 3.3
        self.efficiency = 0.75
        self.lowest_run_voltage = 3.9
        C_all = 1500  # 容量 mAh
        self.N_c = 6
        self.N_p = 1
        self.C_c = C_all / self.N_p

        self.u_1 = torch.zeros((num_envs, 1), device=self.device)
        self.E_c = torch.zeros((num_envs, 1), device=self.device)
        self.time = torch.zeros((num_envs, 1), device=self.device)

        self.temp = 0

    def reset(self, reset_env_ids, random_voltage=False):
        num_resets = len(reset_env_ids)

        self.u_1[reset_env_ids, ...] = torch.zeros((num_resets, 1), device=self.device)
        self.E_c[reset_env_ids, ...] = torch.zeros((num_resets, 1), device=self.device)
        self.time[reset_env_ids, ...] = torch.zeros((num_resets, 1), device=self.device)
        if random_voltage:
            self.E_c[reset_env_ids] = torch_rand_float(0, 2.2, (num_resets, 1), self.device)
    
    def sim_process(self, P_m):
        """
        input P_m is the mech power of all motor-propeller pairs
        size:[num_envs, 1]
        FPV params: P_m = (omega[i] /4500)*(omega[i] /4500)*(omega[i] /4500) * 450
        """

        if self.battery_dynamics:
            self.time += self.dt
            p_c = P_m / self.efficiency / (self.N_c * self.C_c)  # eq 14
            # print( p_c * self.dt)
            self.E_c += p_c * self.dt  # eq 14, Ws

            P_avg = self.E_c / self.time
            r_0_ = (self.b[0] + self.b[1] * P_avg + self.b[2] * self.C_c)
            r_0 = torch.where(r_0_ > self.R_min, r_0_, self.R_min)  # eq 16

            u_0 = self.a0 + self.a[1] * self.E_c + self.a[2] * self.E_c ** 2 + self.a[3] * self.E_c ** 3  # eq 15

            u_1_dot = (self.k * p_c - self.u_1) / self.tau_rc  # eq 10
            self.u_1 += u_1_dot * self.dt  # u_cap
            u_t = 1 / 2 * (u_0 - self.u_1 + torch.sqrt((u_0 - self.u_1) ** 2 - 4 * r_0 * p_c)) * self.N_c  # eq 12, u_battery

            # self.temp += p_c * self.dt / u_t * 1000 * self.C_c * self.N_c
            self.temp += P_m / self.efficiency * self.dt / u_t * 1000 / 3600

            return u_t
        else:
            return self.a0 * self.N_c


if __name__ == '__main__':
    device = 'cuda:0'
    num_envs = 2
    sample_time = 0.01
    battery_dynamics = Battery_Dynamics(num_envs, device=device, battery_dynamics=True, sample_time=sample_time)
    t = []
    u = []
    ec = []
    # battery_dynamics.a[0] = 23.3/6
    # battery_dynamics.a0 = 23.3/6
    battery_dynamics.efficiency = 0.75
    # battery_dynamics.reset([0], False, True)
    battery_dynamics.reset([0,1], True)
    for i in range(270 * int(1 / sample_time)):
        # P_m = (25 + 25 * np.random.rand()) * torch.ones(num_envs, 1, device=device)
        P_m = 300 * torch.ones(num_envs, 1, device=device)
        # P_m = (450 * (300 * 2 * torch.pi / 4500) ** 3)*4
        if i == 0:
            P_m=0
        u_t = battery_dynamics.sim_process(P_m)

        t.append(sample_time * i)
        u.append(u_t[..., 0].squeeze().cpu().numpy().tolist())
        ec.append(battery_dynamics.E_c[0].cpu().numpy().tolist())

    print(battery_dynamics.E_c)
    print(battery_dynamics.temp)
    print(P_m)

    plt.figure()
    plt.grid()
    plt.plot(t, u)
    
    plt.figure()
    plt.grid()
    plt.plot(t, ec)
    plt.show()
