"""
专门用于不对称actor critic的ppo，对称情况需要使用ppo.py
"""

import os
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    from algorithms.buffer_asymmetry import PPOReplayBuffer
    from algorithms.nets_asymmetry import PPO_ActorCritic

except:
    from algorithms.buffer_asymmetry import PPOReplayBuffer
    from algorithms.nets_asymmetry import PPO_ActorCritic


class PPO:
    def __init__(self, env, actor_critic=PPO_ActorCritic, actor_critic_para_dict=dict(),
                 clip=0.2, target_kl=0.03, lam=0.95, max_grad=0.5,
                 use_clipped_value_loss=False,
                 epochs=500, horizon_len=1024, train_iters=16, mini_batch_num=8, gamma=0.99, seed=42,
                 lr=3e-4, pi_coef=1, vf_coef=0.5, ent_coef=0, imit_coef=0, learning_rate_schedule=True, lr_ratio=0.3, lr_lp_index=0.7, lr_epoch_index=350,
                 use_lipschitz=False, lipschitz_para=5, lipschitz_schedule=True, lip_ratio=None, lip_lp_index=None, lip_epoch_index=None,
                 difficulty_schedule=True, diff_value=None, diff_lp_index=None, diff_epoch_index=None,
                 log_dir="", log_interval=100, is_testing=False, device="cuda:0"):

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ############################ save para ###############################
        # env
        self.device = device
        self.env = env
        self.num_envs = self.env.num_envs
        self.len_obs = env.observation_space.shape[0]
        self.num_obs = env.observation_space.shape[1]
        # critic的输入，只影响critic net输入、buffer保存、交互、更新
        self.len_states = env.state_space.shape[0]
        self.num_states = env.state_space.shape[1]
        self.num_act = env.action_space.shape[0]
        self.action_low_limits = torch.tensor(self.env.action_space.low, device=self.device,
                                              dtype=torch.float32, requires_grad=False)
        self.action_high_limits = torch.tensor(self.env.action_space.high, device=self.device,
                                               dtype=torch.float32, requires_grad=False)

        # ppo para
        self.clip = clip  # gradient clip range
        self.target_kl = target_kl
        self.lam = lam  # gae lambda
        self.max_grad = max_grad  # 计算得到梯度后进行截断
        self.use_clipped_value_loss = use_clipped_value_loss  # 是否对值函数损失进行截断。会降低表现，不要用

        # training para
        self.epochs = epochs
        self.horizon_len = horizon_len  # 在每个epoch，每个env收集horizon_len个交互，填满replay_buffer
        self.train_iters = train_iters  # 每个epoch所有交互的使用次数
        self.mini_batch_num = mini_batch_num  # 将replay buffer所有数据，分成多少次进行更新
        self.gamma = gamma
        # 学习率、调节
        # 通过调节各种coef来调节不同loss的占比，实现不同的训练方式
        # 经典强化学习训练/teacher-student中teacher训练：pi_coef = 1, vf_coef > 0, ent_coef > 0, imit_coef = 0
        # 模仿学习训练：pi_coef = 0, vf_coef = 0, ent_coef = 0, imit_coef = 1, imitation_schedule=False
        # teacher-student中student训练：pi_coef = 1, vf_coef > 0, ent_coef > 0, imit_coef = 1, imitation_schedule=True
        self.lr = lr
        self.pi_coef = pi_coef  # 策略梯度loss的权重
        self.vf_coef = vf_coef  # 值函数loss的权重
        self.ent_coef = ent_coef  # 熵loss的权重
        self.imit_coef = imit_coef  # 模仿loss的权重

        self.learning_rate_schedule = learning_rate_schedule  # 是否自动调节学习率，调节方式为在一定范围内线性下降，最后稳定
        self.lr_ratio = lr_ratio  # 最终学习率与初始学习率的比值
        self.lr_lp_index = lr_lp_index  # 学习率停止变化的learning_process
        self.lr_epoch_index = lr_epoch_index  # 学习率停止变化的epoch节点

        # lipschitz、调节
        #  WARNING：需要 lipschitz最后转折点 最晚也在 任务难度的最后转折点 的前面
        self.use_lipschitz = use_lipschitz  # 是否使能lipschitz约束，可以显著提升奖励
        self.lipschitz_para = lipschitz_para  # lipschitz常数初始值
        self.lipschitz_schedule = lipschitz_schedule  # 是否根据训练程度，调节lipschitz约束的参数
        self.lip_ratio = [1, 0.3] if lip_ratio is None else lip_ratio  # lipschitz约束的参数在两个节点的比例，实际参数 = lip_ratio × lipschitz常数初始值
        self.lip_lp_index = [0.3, 0.7] if lip_lp_index is None else lip_lp_index  # lipschitz约束的参数变化的两个learning_process节点
        self.lip_epoch_index = [100, 500] if lip_epoch_index is None else lip_epoch_index  # lipschitz约束的参数变化的两个epoch节点

        # 任务难度、调节
        self.difficulty_schedule = difficulty_schedule  # 是否根据训练程度，调节任务难度
        self.diff_value = [0.1, 1] if diff_value is None else diff_value  # 任务难度在两个节点的取值，取值范围为0-1
        self.diff_lp_index = [0.3, 0.7] if diff_lp_index is None else diff_lp_index  # 任务难度变化的两个learning_process节点
        self.diff_epoch_index = [100, 500] if diff_epoch_index is None else diff_epoch_index  # 任务难度变化的两个epoch节点

        self.is_testing = is_testing

        # ############################ replay buffer ###############################
        self.replay_buffer = PPOReplayBuffer(num_envs=self.num_envs,
                                             obs_dim=self.num_obs,
                                             obs_len=self.len_obs,
                                             states_dim=self.num_states,
                                             states_len=self.len_states,
                                             act_dim=self.num_act,
                                             horizon_len=self.horizon_len,
                                             mini_batch_num=self.mini_batch_num,
                                             gamma=self.gamma,
                                             lam=self.lam,
                                             device=self.device)

        # ############################ neural network ###############################
        if not self.is_testing:  # 如果是测试，就不创建模型结构了，直接加载保存的结构与参数。train与retrain都需要重新创建
            self.agent = actor_critic(actor_critic_para_dict)
            self.agent.to(self.device)
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.agent.parameters()), lr=lr, eps=1e-5)  # 只更新requires_grad的参数

        # ############################ logs ###############################
        if not self.is_testing:
            self.log_interval = log_interval
            self.log_dir = os.path.join(log_dir, 'nn')
            summary_dir = os.path.join(log_dir, 'summaries')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.writer = SummaryWriter(log_dir=summary_dir, flush_secs=10)
            print("log dir", log_dir)

        # other stuff
        self.mean_reward = 0
        self.optim_step = 0

    #####################################################################
    ### =======================train functions=======================###
    #####################################################################
    def update(self, epoch):
        self.agent.train()  # 训练开始前切换到train
        learning_process = epoch / self.epochs  # 学习进程，0-1之间

        if self.learning_rate_schedule:
            # 更新learning rate，线性递减，最小0.3
            lr_ratio0 = (self.lr_ratio - 1) / self.lr_lp_index * learning_process + 1 \
                if learning_process < self.lr_lp_index else self.lr_ratio
            lr_ratio1 = (self.lr_ratio - 1) / self.lr_epoch_index * epoch + 1 \
                if epoch < self.lr_epoch_index else self.lr_ratio
            self.optimizer.param_groups[0]['lr'] = learning_rate = min(lr_ratio0, lr_ratio1) * self.lr
        else:
            self.optimizer.param_groups[0]['lr'] = learning_rate = self.lr_ratio * self.lr

        if self.lipschitz_schedule:
            # 更新lipschitz常数，线性递减，最小1.5
            lipschitz_para_0 = self.lip_ratio[0] if learning_process < self.lip_lp_index[0] \
                else self.lip_ratio[1] if learning_process > self.lip_lp_index[1] \
                else (self.lip_ratio[1] - self.lip_ratio[0]) / (self.lip_lp_index[1] - self.lip_lp_index[0]) * (learning_process - self.lip_lp_index[0]) + self.lip_ratio[0]
            lipschitz_para_1 = self.lip_ratio[0] if epoch < self.lip_epoch_index[0] \
                else self.lip_ratio[1] if epoch > self.lip_epoch_index[1] \
                else (self.lip_ratio[1] - self.lip_ratio[0]) / (self.lip_epoch_index[1] - self.lip_epoch_index[0]) * (epoch - self.lip_epoch_index[0]) + self.lip_ratio[0]
            lipschitz_para = min(lipschitz_para_0, lipschitz_para_1) * self.lipschitz_para
        else:
            lipschitz_para = self.lip_ratio[1] * self.lipschitz_para  # 直接取最后值

        if self.difficulty_schedule:
            # 更新任务难度，前40最简难度，后40最高难度，中间逐渐提高
            difficulty_0 = self.diff_value[0] if learning_process < self.diff_lp_index[0] \
                else self.diff_value[1] if learning_process > self.diff_lp_index[1] \
                else (self.diff_value[1] - self.diff_value[0]) / (self.diff_lp_index[1] - self.diff_lp_index[0]) * (learning_process - self.diff_lp_index[0]) + self.diff_value[0]
            difficulty_1 = self.diff_value[0] if epoch < self.diff_epoch_index[0] \
                else self.diff_value[1] if epoch > self.diff_epoch_index[1] \
                else (self.diff_value[1] - self.diff_value[0]) / (self.diff_epoch_index[1] - self.diff_epoch_index[0]) * (epoch - self.diff_epoch_index[0]) + self.diff_value[0]
            # difficulty = self.env.difficulty = round(max(difficulty_0, difficulty_1), 2)
            difficulty = self.env.difficulty = max(difficulty_0, difficulty_1)
        else:
            difficulty = self.env.difficulty = self.diff_value[1]  # 直接取最后值

        # epoch iteration
        batch_idx = self.replay_buffer.batch_idx_generator()
        pg_losses, value_losses, entropy_losses, imitation_losses, sum_losses = [], [], [], [], []
        approx_kl_divs = []
        continue_training = True
        for i in range(self.train_iters):
            for indices in batch_idx:
                obs_batch = self.replay_buffer.obs_buf.view(-1, *self.replay_buffer.obs_buf.size()[2:])[indices]
                states_batch = self.replay_buffer.states_buf.view(-1, *self.replay_buffer.states_buf.size()[2:])[indices]
                actions_batch = self.replay_buffer.act_buf.view(-1, self.replay_buffer.act_buf.size(-1))[indices]
                old_value_batch = self.replay_buffer.value_buf.view(-1, 1)[indices]
                returns_batch = self.replay_buffer.ret_buf.view(-1, 1)[indices]
                done_batch = self.replay_buffer.done_buf.view(-1, 1)[indices]

                old_actions_log_prob_batch = self.replay_buffer.logp_buf.view(-1, 1)[indices]
                advantages_batch = self.replay_buffer.adv_buf.view(-1, 1)[indices]
                old_mu_batch = self.replay_buffer.mu_buf.view(-1, self.replay_buffer.act_buf.size(-1))[indices]
                old_sigma_batch = self.replay_buffer.sigma_buf.view(-1, self.replay_buffer.act_buf.size(-1))[indices]

                # 前向传播过程，消耗巨量显存
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.agent.evaluate(obs_batch, states_batch, actions_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
                surrogate_loss = - torch.min(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = old_value_batch + (value_batch - old_value_batch).clamp(-0.2, 0.2)
                    value_losses = F.mse_loss(value_batch, returns_batch)
                    value_losses_clipped = F.mse_loss(returns_batch, value_batch)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = F.mse_loss(returns_batch, value_batch)

                # Entropy loss
                entropy_loss = - torch.mean(entropy_batch)

                loss = self.pi_coef * surrogate_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                # print("loss", loss)

                # record
                pg_losses.append(surrogate_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                sum_losses.append(loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = actions_log_prob_batch - old_actions_log_prob_batch.view(-1)
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if approx_kl_div > 1.5 * self.target_kl and self.pi_coef > 0:  # 假如不是纯模仿学习模式，则需要限制更新的kl距离
                    continue_training = False
                    print(f"Early stopping at {i} train_iters of epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad)  # 梯度截断
                self.optimizer.step()
                self.optim_step += 1

                if self.use_lipschitz:
                    self.spectral_normalize_actors(lipschitz_para)

            if not continue_training:
                break

        mean_value = old_value_batch.cpu().numpy().mean()
        explained_var = self.explained_variance(old_value_batch.cpu().numpy().flatten(),
                                                returns_batch.cpu().numpy().flatten())

        self.log_update(locals())
        self.agent.eval()  # 训练结束后切换到eval

    def run(self):
        if self.is_testing:
            print("testing")
            self.agent.eval()
            t = 0
            obs_dict = self.env.reset()
            obs = obs_dict['obs']
            states = obs_dict['states']
            action = self.agent.act(obs, states, deterministic=True, action_only=True)
            print("all zero observation action: ", action[0])

            ret = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            while True:
                with torch.no_grad():
                    t = t + 1
                    action = self.agent.act(obs, states, deterministic=True, action_only=True)
                    # print((action[:, 0] + 1) / 2 * 1000)
                    next_obs_states_dict, rewards, dones, _ = self.env.step(action)
                    ret += rewards
                    done_envs = torch.nonzero(dones).squeeze().tolist()
                    if done_envs:
                        # print("mean return: ", ret[done_envs].mean(), "std return: ", ret[done_envs].std(unbiased=False))
                        print("return 0: ", ret[0], "return -1: ", ret[-1])
                        ret[done_envs] = 0
                    obs.copy_(next_obs_states_dict['obs'])
                    states.copy_(next_obs_states_dict['states'])
        else:
            print("trainning")
            cur_return = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 各环境当前episode的回报
            cur_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 各环境当前episode的长度
            final_return = []  # 各环境当前episode结束时的回报
            final_length = []  # 各环境当前episode结束时的长度
            latest_return_buffer = deque(maxlen=100)  # 最近结束的100个episode的reward
            latest_length_buffer = deque(maxlen=100)  # 最近结束的100个episode的reward
            highest_return = - np.inf
            highest_return_final = - np.inf

            obs_dict = self.env.reset()
            obs = obs_dict['obs']
            states = obs_dict['states']
            for epoch in range(self.epochs):
                # 收集数据
                self.replay_buffer.reset()
                done_env_num = 0  # 当前epoch中reset的环境数量
                truncated_env_num = 0  # 当前epoch中因为truncated而reset的环境数量
                time_0 = time.time()
                self.agent.eval()  # 收集数据的时候用eval
                with torch.no_grad():
                    for _ in range(self.horizon_len):
                        action, log_prob, value, mu, sigma = self.agent.act(obs, states)
                        clipped_act = torch.clip(action, self.action_low_limits, self.action_high_limits)
                        next_obs_states_dict, rewards, dones, infos = self.env.step(clipped_act)
                        next_obs = next_obs_states_dict['obs']
                        next_states = next_obs_states_dict['states']

                        truncated_env_ids = (infos["time_outs"] * dones).nonzero(as_tuple=False).squeeze(-1).tolist()
                        done_env_num += len(dones.nonzero(as_tuple=False))  # 当前epoch中done的所有env数量
                        truncated_env_num += len(truncated_env_ids)  # 当前epoch中truncated的所有env数量

                        # bootstrapping, copied from sb3, see GitHub sb3 issue #633，与gae搭配使用
                        # 没法加到buffer内处理，因为需要使用调用self.agent.act
                        rewards_augmented = rewards.clone()
                        if truncated_env_ids:
                            _, _, time_out_value, _, _ = self.agent.act(obs[truncated_env_ids, :], states[truncated_env_ids, :])
                            rewards_augmented[truncated_env_ids] += self.gamma * time_out_value.squeeze()

                        self.replay_buffer.store(obs, states, action, rewards_augmented, log_prob, dones, value, mu, sigma)

                        obs.copy_(next_obs)
                        states.copy_(next_states)

                        cur_return[:] += rewards  # 环境中各子环境的当前总回报：奖励之和
                        cur_length[:] += 1  # 环境中各子环境的当前步长
                        self.mean_reward += torch.mean(rewards).item()  # 当前epoch中每一步的（各子环境当前奖励的平均）之和

                        reset_idx = dones.nonzero()
                        final_return += cur_return[reset_idx].tolist()  # 当前epoch中done的所有env的return
                        final_length += cur_length[reset_idx].tolist()  # 当前epoch中done的所有env的length
                        cur_return[reset_idx] = 0
                        cur_length[reset_idx] = 0

                    _, _, value, _, _ = self.agent.act(obs, states)  # Compute value for the last timestep
                    self.replay_buffer.compute_returns_and_advantage(value)

                # 网络更新
                time_1 = time.time()
                self.agent.train()  # 收集完切回train
                self.update(epoch)  # update
                time_2 = time.time()

                # 其他计算
                latest_return_buffer.extend(final_return)
                latest_length_buffer.extend(final_length)
                final_return = []
                final_length = []
                sim_time = time_1 - time_0
                train_time = time_2 - time_1
                mean_return = np.array(latest_return_buffer).mean()  # 有时候一个epoch中一个死掉的都没有，就求不了值，输出个nan，是可以忽略的
                std_return = np.array(latest_return_buffer).std()  # 有时候一个epoch中一个死掉的都没有，就求不了值，输出个nan，是可以忽略的
                mean_length = np.array(latest_length_buffer).mean()  # 有时候一个epoch中一个死掉的都没有，就求不了值，输出个nan，是可以忽略的
                mean_reward = self.mean_reward / self.horizon_len
                self.mean_reward = 0
                if epoch % 10 == 0:
                    # 输出当前epoch、已经进行的网络优化次数、策略分布的方差、episode平均长度、episode平均回报、step平均奖励
                    print('Epoch: {:04d} / {:04d}| Opt Step: {:04d}'.format(epoch + 1, self.epochs, self.optim_step))
                    print('Sim time: {:04f} | Train time: {:04f}'.format(sim_time, train_time))
                    print('Action Var {:.04f} | Mean length {:.2f} | Mean return {:.2f} | Mean reward {:.2f}'.format(self.agent.log_std.exp().mean().item(), mean_length, mean_return, mean_reward))
                    print('#################################')

                # 最优模型保存文件名为'model_0.pt'
                if mean_return > highest_return:
                    highest_return = mean_return
                    self.save(os.path.join(self.log_dir, 'model_0.pt'), para_only=False)
                    print('The current best network is', epoch, ', the return is ', round(highest_return, 2))

                # 最高难度最优模型保存文件名为'model_1.pt'
                if mean_return > highest_return_final and self.env.difficulty == 1:
                    highest_return_final = mean_return
                    self.save(os.path.join(self.log_dir, 'model_1.pt'), para_only=False)
                    print('The current best network is', epoch, ', the return is ', round(highest_return_final, 2))

                # 中间模型只保存参数
                if ((epoch % self.log_interval == 0) and epoch != 0) or (epoch == self.epochs):
                    self.save(os.path.join(self.log_dir, 'model_{}_{}.pt'.format(epoch, round(mean_return, 2))), para_only=False)

                self.log_interact(locals())

            # 将最优模型的actor保存为不同的格式
            self.agent = torch.load(os.path.join(self.log_dir, 'model_0.pt'))
            self.save_actor_as_pt(os.path.join(self.log_dir, 'actor_0.pt'))

            # 将最高难度最优模型的actor保存为不同的格式
            self.agent = torch.load(os.path.join(self.log_dir, 'model_1.pt'))
            self.save_actor_as_pt(os.path.join(self.log_dir, 'actor_1.pt'))

    #####################################################################
    ### =======================advanced functions=======================###
    #####################################################################
    def spectral_normalize_actors(self, lipschitz_const=5):
        # lipschitz约束
        for param in self.agent.actor_mlp.parameters():
            if param.data.ndim > 1:
                spectral_norm = torch.linalg.matrix_norm(param.data, ord=2)  # 矩阵2范数，即矩阵最大奇异值
                if spectral_norm > lipschitz_const:
                    param.data *= lipschitz_const / spectral_norm
                    # print("spectral_norm")

    def explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]

        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero

        :param y_pred: the prediction
        :param y_true: the expected value
        :return: explained variance of ypred and y
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    #####################################################################
    ### =======================save functions=======================###
    #####################################################################
    def log_interact(self, locs):
        # 保存交互时的数据到tensorboard
        self.writer.add_scalar('Interact/Return:', locs['mean_return'], locs['epoch'])
        self.writer.add_scalar('Interact/Length:', locs['mean_length'], locs['epoch'])
        self.writer.add_scalar('Interact/Reward:', locs['mean_reward'], locs['epoch'])
        self.writer.add_scalar('Interact/Reward Var:', locs['std_return'], locs['epoch'])
        self.writer.add_scalar('Interact/Var:', self.agent.log_std.exp().mean().item(), locs['epoch'])
        self.writer.add_scalar('Interact/done_env_num:', locs['done_env_num'], locs['epoch'])
        self.writer.add_scalar('Interact/truncated_env_num:', locs['truncated_env_num'], locs['epoch'])

    def log_update(self, locs):
        # 保存更新时的数据到tensorboard
        self.writer.add_scalar('Update/policy_gradient_loss:', np.mean(locs['pg_losses']), locs['epoch'])
        self.writer.add_scalar('Update/value_loss:', np.mean(locs['value_losses']), locs['epoch'])
        self.writer.add_scalar('Update/entropy_loss:', np.mean(locs['entropy_losses']), locs['epoch'])
        self.writer.add_scalar('Update/sum_loss:', np.mean(locs['sum_losses']), locs['epoch'])
        self.writer.add_scalar('Update/mean_value:', locs['mean_value'], locs['epoch'])
        self.writer.add_scalar('Update/explained_variance:', locs['explained_var'], locs['epoch'])
        self.writer.add_scalar('Update/learning_rate:', locs['learning_rate'], locs['epoch'])
        self.writer.add_scalar('Update/lipschitz_para:', locs['lipschitz_para'], locs['epoch'])
        self.writer.add_scalar('Update/learning_process:', locs['learning_process'], locs['epoch'])
        self.writer.add_scalar('Update/difficulty:', locs['difficulty'], locs['epoch'])
        self.writer.add_scalar("Update/approx_kl", np.mean(locs['approx_kl_divs']), locs['epoch'])

    def save(self, path, para_only=True):
        if para_only:
            torch.save(self.agent.state_dict(), path)
        else:
            torch.save(self.agent, path)

    def save_actor_as_pt(self, path):
        # save policy network only
        self.agent.eval()
        obs = torch.zeros((1, self.len_obs, self.num_obs), device='cuda:0')
        traced_script_module = torch.jit.trace(self.agent, obs)
        traced_script_module.save(path)
        # test output
        torch_out = self.agent(obs)
        pt_in = torch.zeros((1, self.len_obs, self.num_obs), device='cuda:0')
        pt_out = traced_script_module(pt_in)
        print("touch_out", torch_out, "pt_out", pt_out)

    #####################################################################
    ### =======================test functions=======================###
    #####################################################################
    def run_model(self):
        obs = self.env.obs_buf.clone()
        action, _, _, _, _ = self.agent(obs)
        self.env.step(action)
        next_obs, reward, dones = self.env.obs_buf.clone(), self.env.rew_buf.clone(), self.env.reset_buf.clone()
        return obs, action, next_obs, dones
