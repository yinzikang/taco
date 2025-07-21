import torch
from scipy.spatial.transform import Rotation as R


class angvel_control():
    """
    destination_angvel mul 20 outside
    """

    def __init__(self, num_agents, num_tar_agents, num_envs, device, dt):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_tar_agents = num_tar_agents
        self.active_agents = num_agents + num_tar_agents
        self.device = device

        self.kp = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # rpy
        self.kp[:, 0] = 27.5
        self.kp[:, 1] = 50
        self.kp[:, 2] = 200

        self.ki = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # rpy
        self.ki[:, 0] = 0#1000
        self.ki[:, 1] = 0#1000
        self.ki[:, 2] = 0#1000

        self.kd = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)
        self.kd[:, 0] = 0.5
        self.kd[:, 1] = 0.5
        self.kd[:, 2] = 0.5
        
        self.Kf = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # feedforward
        self.Kf[:, 0] = 0
        self.Kf[:, 1] = 0#30
        self.Kf[:, 2] = 0#10
        
        self.EM = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # error max
        self.EM[:, 0] = 400
        self.EM[:, 1] = 400
        self.EM[:, 2] = 400

        self.IM = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # integration max
        self.IM[:, 0] = 500
        self.IM[:, 1] = 500 
        self.IM[:, 2] = 500 
        
        self.DM = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # integration max
        self.DM[:, 0] = 150
        self.DM[:, 1] = 150 
        self.DM[:, 2] = 150 
        
        self.fg = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # final gain
        self.fg[:, 0] = 0.4
        self.fg[:, 1] = 0.4
        self.fg[:, 2] = 0.4

        self.Dfrenq = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)  # 滤波，现在没加
        self.Dfrenq[:, 0] = 15
        self.Dfrenq[:, 1] = 15
        self.Dfrenq[:, 2] = 15

        self.dt = dt

        self.previous_error = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)
        self.integral = torch.zeros((num_envs * self.active_agents, 3), device=device, dtype=torch.float)

    def compute(self, destination_angvel, current_w):
        # error,保证初始状态下的previous_error不要过大
        error = torch.clip(destination_angvel - current_w, -self.EM, self.EM)
        self.previous_error = torch.where(self.previous_error == 0, error, self.previous_error)
        # P
        P_term = self.kp * error
        # I
        self.integral = self.integral + error * self.dt
        self.integral = torch.clip(self.integral, -self.IM, self.IM)
        I_term = self.ki * self.integral
        # D
        derivative = (error - self.previous_error) / self.dt
        D_term = torch.clip(self.kd * derivative, -self.DM, self.DM)
        # feedforward
        FF_term = self.Kf * destination_angvel
        # final
        torque_amt = self.fg * (P_term + I_term + D_term + FF_term)
        # print("error",error, "p",P_term, "i",self.integral,"d",D_term,"out",torque_amt)

        self.previous_error = error

        return torque_amt

    def reset(self, reset_env_ids):
        num_resets = len(reset_env_ids)

        self.integral[reset_env_ids, :] = torch.zeros(num_resets * self.active_agents, 3, device=self.device, dtype=torch.float)
        self.previous_error[reset_env_ids, :] = torch.zeros(num_resets * self.active_agents, 3, device=self.device, dtype=torch.float)


if __name__ == "__main__":
    num_envs = 1
    num_agents = 1
    num_tar_agents = 0
    dt = 0.001
    device = 'cuda:0'
    num_agents_active = num_agents + num_tar_agents
    angvel = angvel_control(num_agents, num_tar_agents, num_envs, device, dt)

    # destination = torch.zeros((1,1,4), device=device)
    # current_w = torch.zeros((1,1,3), device=device)
    destination = torch.tensor([0, 0, 1], device=device)
    current_w = torch.tensor([0, 0, 1], device=device)

    tau = angvel.compute(destination, current_w)
    print(tau)
