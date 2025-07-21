import torch

try:
    import nets_asymmetry as core
except:
    import algorithms.nets_asymmetry as core


class PPOReplayBuffer:
    def __init__(self, num_envs, obs_dim, obs_len, states_dim, states_len, act_dim, horizon_len, mini_batch_num, gamma, lam, device):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.states_dim = states_dim  # critic输入维度
        self.states_len = states_len
        self.act_dim = act_dim
        self.horizon_len = horizon_len  # replay buffer中单个环境在一个epoch需要收集的交互数量
        self.mini_batch_num = mini_batch_num  # 将replay buffer所有数据，分成多少次进行更新
        # para
        self.gamma = gamma
        self.lam = lam  # gae parameter
        self.device = device

        # o s a r + done，但是貌似保存的下时刻观测、状态都没有用上，实际上也就一两百mb显存
        self.obs_buf = torch.zeros(self.horizon_len, self.num_envs, self.obs_len, self.obs_dim, dtype=torch.float32,
                                   device=self.device)
        self.states_buf = torch.zeros(self.horizon_len, self.num_envs, self.states_len, self.states_dim, dtype=torch.float32,
                                           device=self.device)
        self.act_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, dtype=torch.float32,
                                   device=self.device)
        self.rew_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)
        # 下一时刻的观测/状态在更新时用不着，因此注释掉了，免得占用显存
        # self.nex_obs_buf = torch.zeros(self.horizon_len, self.num_envs, self.obs_dim, dtype=torch.float32,
        #                                device=self.device)
        # self.nex_states_buf = torch.zeros(self.horizon_len, self.num_envs, self.states_dim, dtype=torch.float32,
        #                                device=self.device)
        # critic
        self.ret_buf = torch.zeros(self.horizon_len, self.num_envs, 1, device=self.device)
        self.value_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.adv_buf = torch.zeros(self.horizon_len, self.num_envs, 1, device=self.device)
        # actor
        self.mu_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, device=self.device)
        self.sigma_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, device=self.device)
        self.logp_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.step = 0

    def store(self, obs, states, act, rew, log_prob, done, value, mu, sigma):
        if self.step >= self.horizon_len:
            raise AssertionError("Rollout buffer overflow")

        # sars + done + privileged information
        self.obs_buf[self.step].copy_(obs)
        self.states_buf[self.step].copy_(states)
        self.act_buf[self.step].copy_(act)
        self.rew_buf[self.step].copy_(rew.view(-1, 1))
        self.done_buf[self.step].copy_(done.view(-1, 1))

        # critic
        self.value_buf[self.step].copy_(value)

        # actor
        self.mu_buf[self.step].copy_(mu)
        self.sigma_buf[self.step].copy_(sigma)
        self.logp_buf[self.step].copy_(log_prob.view(-1, 1))

        self.step += 1

    def reset(self):
        # osar + done
        self.obs_buf = torch.zeros(self.horizon_len, self.num_envs, self.obs_len, self.obs_dim, dtype=torch.float32,
                                   device=self.device)
        self.states_buf = torch.zeros(self.horizon_len, self.num_envs, self.states_len, self.states_dim, dtype=torch.float32,
                                   device=self.device)
        self.act_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, dtype=torch.float32,
                                   device=self.device)
        self.rew_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)

        # critic
        self.ret_buf = torch.zeros(self.horizon_len, self.num_envs, 1, device=self.device)
        self.value_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.adv_buf = torch.zeros(self.horizon_len, self.num_envs, 1, device=self.device)

        # actor
        self.mu_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, device=self.device)
        self.sigma_buf = torch.zeros(self.horizon_len, self.num_envs, self.act_dim, device=self.device)
        self.logp_buf = torch.zeros(self.horizon_len, self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.step = 0

    def compute_returns_and_advantage(self, last_values):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # last_values = last_values.clone().reshape((self.num_envs, 1))
        # dones = dones.reshape((self.num_envs, 1))

        last_gae_lam = 0
        for step in reversed(range(self.horizon_len)):
            if step == self.horizon_len - 1:
                next_values = last_values
            else:
                next_values = self.value_buf[step + 1]
            next_non_terminal = 1.0 - self.done_buf[step].float()

            td_target = self.rew_buf[step] + next_non_terminal * self.gamma * next_values
            delta = td_target - self.value_buf[step]  # one step bootstrap
            last_gae_lam = delta + next_non_terminal * self.gamma * self.lam * last_gae_lam
            self.adv_buf[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.ret_buf = self.adv_buf + self.value_buf

        # normalize the advantages
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)

    def batch_idx_generator(self):
        # 将一个epoch中所有环境采集的所有交互共batch_size个，分成mini_batch_num次给出，进行训练
        original_list = torch.arange(self.num_envs * self.horizon_len)  # 创建原始列表
        random_indices = torch.randperm(self.num_envs * self.horizon_len)  # 生成随机索引
        data_idx = original_list[random_indices].reshape(self.mini_batch_num, -1).tolist()  # 重新排列列表
        return data_idx