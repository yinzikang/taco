import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import MultivariateNormal

"""utils"""


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪的模块，裁剪一侧多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU, output_activation=nn.Identity):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size

        sizes = [input_size] + hidden_size + [output_size]
        layers = []
        for j in range(len(sizes) - 2):
            # layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.BatchNorm1d(sizes[j + 1]), activation()]  # 不要加BatchNorm1d，负面影响很大
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x.contiguous().view(x.size(0), -1))
        return x

    def para_init(self, last_layer_only=False):
        weights = [np.sqrt(2)] * len(self.hidden_size)
        weights.append(0.01)
        if last_layer_only:
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    self.init_weights([layer], weights)
                    break
        else:
            self.init_weights(self.layers, weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def para_freeze(self, last_layer=False):
        for name, param in self.named_parameters():
            param.requires_grad = False
        if last_layer:
            for name, param in self.layers[-2].named_parameters():
                param.requires_grad = True


class CNNEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size, stride):
        super(CNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(input_size, output_size, kernel_size, stride))
            # self.layers.append(nn.BatchNorm1d(output_size))
            self.layers.append(nn.ReLU())

            input_size = output_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将batch * timestep * feature变成batch * feature * timestep
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)

        return x[:, -1, :]

    def para_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def para_freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


class TCNEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size=3, stride=1, dilation_base=2):
        super(TCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = dilation * (kernel_size - 1)
            self.layers.append(nn.Conv1d(input_size, output_size, kernel_size, stride, dilation=dilation, padding=padding))
            self.layers.append(Chomp1d(padding))
            # self.layers.append(nn.BatchNorm1d(output_size))
            self.layers.append(nn.ReLU())

            input_size = output_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将batch * timestep * feature变成batch * feature * timestep
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)

        return x[:, -1, :]

    def para_init(self):
        for i in range(0, len(self.layers), 4):
            if isinstance(self.layers[i], nn.Conv1d):
                nn.init.kaiming_uniform_(self.layers[i].weight)
                nn.init.constant_(self.layers[i].bias, 0.0)

    def para_freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.layers = nn.LSTM(input_size, output_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.layers(x)

        return x[:, -1, :]

    def para_init(self):
        for param in self.layers.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)
            else:
                init.constant_(param, 0)

    def para_freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class AttentionEncoder(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, dropout=0.1, max_len=5000):
        super(AttentionEncoder, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Linear(input_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward=embed_size * 4, dropout=dropout, batch_first=True))

    def forward(self, x):  # batch * timestep * feature
        x = self.embedding(x) * math.sqrt(self.embed_size)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)  # query、key、value 都是输入 x
        return x

    def para_init(self):
        """
        参数初始化函数，使用正态分布初始化参数
        """
        # Initialize the embedding layer
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.constant_(self.embedding.bias, 0)

        # Initialize the transformer encoder layers
        for layer in self.layers:
            # Initialize weights for the self-attention layer
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.constant_(layer.self_attn.out_proj.bias, 0)

            # Initialize weights for the feedforward layers
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.constant_(layer.linear1.bias, 0)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.constant_(layer.linear2.bias, 0)

    def para_freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


def create_encoder(para_dict=dict()):
    """
    创建各种encoder
    原始数据：batch * timestep * feature
    CNN：输入原为batch * feature * timestep，现已换顺序；输出batch * 1 * hidden；将全部计算结果送入mlp
    TCN：输入原为batch * feature * timestep，现已换顺序；输出batch * timestep * hidden；将计算结果的最后一个timestep送入mlp
    LSTM：输入batch * timestep * feature；输出batch * timestep * hidden；将计算结果的最后一个timestep送入mlp
    ATTENTION：输入batch * timestep * feature；输出batch * timestep * feature；将全部计算结果送入mlp
    """
    if para_dict['encoder_type'] == 'CNN':
        encoder = CNNEncoder(input_size=para_dict['input_size'],
                             output_size=para_dict['output_size'],
                             num_layers=para_dict['num_layers'],
                             kernel_size=para_dict['kernel_size'],
                             stride=para_dict['stride'])
        mlp_input_dim = para_dict['output_size']
    elif para_dict['encoder_type'] == 'TCN':
        encoder = TCNEncoder(input_size=para_dict['input_size'],
                             output_size=para_dict['output_size'],
                             num_layers=para_dict['num_layers'],
                             kernel_size=para_dict['kernel_size'],
                             dilation_base=para_dict['dilation_base'])
        mlp_input_dim = para_dict['output_size']
    elif para_dict['encoder_type'] == 'LSTM':
        encoder = LSTMEncoder(input_size=para_dict['input_size'],
                              output_size=para_dict['output_size'],
                              num_layers=para_dict['num_layers'],
                              bidirectional=para_dict['bidirectional'])
        mlp_input_dim = para_dict['output_size'] if not para_dict['bidirectional'] else para_dict['output_size'] * 2
    elif para_dict['encoder_type'] == 'ATTENTION':
        encoder = AttentionEncoder(input_size=para_dict['input_size'],
                                   embed_size=para_dict['embed_size'],
                                   num_heads=para_dict['num_heads'],
                                   num_layers=para_dict['num_layers'],
                                   dropout=para_dict['dropout'])
        mlp_input_dim = para_dict['embed_size'] * para_dict['time_len']
    else:
        print('error encoder type')

    return encoder, mlp_input_dim


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# try to get a feel for how different size networks behave!
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


"""PPO ActorCritic"""


class PPO_ActorCritic(nn.Module):
    def __init__(self, para_dict):
        super().__init__()

        self.actor_input_dim = para_dict['actor_critic_mlp_dict']['actor_input_dim']
        self.actor_output_dim = para_dict['actor_critic_mlp_dict']['actor_output_dim']
        self.critic_input_dim = para_dict['actor_critic_mlp_dict']['critic_input_dim']
        self.critic_output_dim = para_dict['actor_critic_mlp_dict']['critic_output_dim']
        self.actor_hidden_dim = list(para_dict['actor_critic_mlp_dict']['actor_hidden_sizes'])
        self.critic_hidden_dim = list(para_dict['actor_critic_mlp_dict']['critic_hidden_sizes'])
        self.activation = para_dict['actor_critic_mlp_dict']['activation']

        self.use_actor_encoder = para_dict['use_actor_encoder']
        self.use_critic_encoder = para_dict['use_critic_encoder']
        self.share_encoder = para_dict['share_encoder']

        # encoder
        if self.use_actor_encoder:
            self.actor_encoder_type = para_dict['actor_encoder_type']
            print('actor using ' + self.actor_encoder_type + ' encoder')
            self.actor_encoder, self.actor_input_dim = create_encoder(para_dict['actor_encoder_dict'])
            self.actor_encoder.para_init()

        if self.use_critic_encoder:
            self.critic_encoder_type = para_dict['critic_encoder_type']
            print('critic using ' + self.critic_encoder_type + ' encoder')
            self.critic_encoder, self.critic_input_dim = create_encoder(para_dict['critic_encoder_dict'])
            self.critic_encoder.para_init()

        if self.share_encoder:
            print('critic using shared encoder')
            self.use_critic_encoder = self.use_actor_encoder
            self.critic_encoder_type = self.actor_encoder_type
            self.critic_encoder, self.critic_input_dim = self.actor_encoder, self.actor_input_dim

        if self.use_actor_encoder:
            print("actor encoder net structure", self.actor_encoder)
            print("actor encoder net para count", count_vars(self.actor_encoder))
        if self.use_critic_encoder:
            print("critic encoder net structure", self.critic_encoder)
            print("critic encoder net para count", count_vars(self.critic_encoder))

        # actor mlp net
        self.actor_mlp = MLP(self.actor_input_dim, self.actor_hidden_dim, self.actor_output_dim, self.activation, nn.Tanh)
        self.actor_mlp.para_init()
        self.log_std = nn.Parameter(np.log(1.0) * torch.ones(self.actor_output_dim))  # Action noise

        # critic mlp net
        self.critic_mlp = MLP(self.critic_input_dim, self.critic_hidden_dim, self.critic_output_dim, self.activation, nn.Identity)
        self.critic_mlp.para_init()

        print("actor mlp net structure", self.actor_mlp)
        print("actor mlp net para count", count_vars(self.actor_mlp))
        print("critic mlp net structure", self.critic_mlp)
        print("critic mlp para count", count_vars(self.critic_mlp))

    def act(self, actor_input, critic_input, deterministic=False, action_only=False):
        """
        在run中使用
        actor_input: actor的输入。对于stack frame，输入为n-T到n的长度为T的序列；对于recurrence，输入为0-n的长度为n的序列
        critic_input: critic的输入。对于stack frame，输入为n-T到n的长度为T的序列；对于recurrence，输入为0-n的长度为n的序列
        deterministic: 输出为action均值还是采样
        action_only: 是否只输出动作
        """

        # actor
        actor_mlp_input = self.actor_encoder(actor_input) if self.use_actor_encoder else actor_input
        action_mean = self.actor_mlp(actor_mlp_input)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(action_mean, scale_tril=covariance)
        if deterministic:
            action = action_mean
        else:
            action = distribution.sample()

        if action_only:
            return action

        log_p = distribution.log_prob(action)

        # critic, 哪怕share也再计算一次，因为可能actor_input和critic_input不一样
        critic_mlp_input = self.critic_encoder(critic_input) if self.use_critic_encoder else critic_input
        value = self.critic_mlp(critic_mlp_input)

        return action.detach(), log_p.detach(), value.detach(), action_mean.detach(), self.log_std.repeat(
            action_mean.shape[0], 1).detach()

    def evaluate(self, actor_input, critic_input, actor_output):
        """
        在update中使用
        actor_input: actor的输入。对于stack frame，输入为n-T到n的长度为T的序列；对于recurrence，输入为0-n的长度为n的序列
        critic_input: critic的输入。对于stack frame，输入为n-T到n的长度为T的序列；对于recurrence，输入为0-n的长度为n的序列
        actor_output: 旧网络在actor_input采样得到的action
        """

        # actor
        actor_mlp_input = self.actor_encoder(actor_input) if self.use_actor_encoder else actor_input
        action_mean = self.actor_mlp(actor_mlp_input)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(action_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actor_output)
        entropy = distribution.entropy()

        # critic, 哪怕share也再计算一次，因为可能actor_input和critic_input不一样
        critic_mlp_input = self.critic_encoder(critic_input) if self.use_critic_encoder else critic_input
        value = self.critic_mlp(critic_mlp_input)

        return actions_log_prob, entropy, value, action_mean, self.log_std.repeat(action_mean.shape[0], 1)

    def forward(self, actor_input):
        """
        用来结合torch.jit.trace
        """
        actor_mlp_input = self.actor_encoder(actor_input) if self.use_actor_encoder else actor_input
        action_mean = self.actor_mlp(actor_mlp_input)

        return action_mean

    def reset_parameters(self, reset_actor_encoder=False, reset_actor_mlp=False, reset_actor_last_layer_only=False, reset_actor_std=False,
                         reset_critic_encoder=False, reset_critic_mlp=False, reset_critic_last_layer_only=False):
        if self.use_actor_encoder and reset_actor_encoder:
            self.actor_encoder.para_init()
        if reset_actor_mlp:
            self.actor_mlp.para_init(reset_actor_last_layer_only)
        if reset_actor_std:
            self.log_std = nn.Parameter(np.log(1.0) * torch.ones(self.actor_output_dim))
            # self.log_std = nn.Parameter(np.log(1.0) * torch.ones(4))

        if self.use_critic_encoder and reset_critic_encoder:
            self.critic_encoder.para_init()
        if reset_critic_mlp:
            self.critic_mlp.para_init(reset_critic_last_layer_only)

    def freeze_para(self, actor_encoder=False, actor_mlp=False, actor_last_layer=False,
                    critic_encoder=False, critic_mlp=False, critic_last_layer=False):
        if self.use_actor_encoder and actor_encoder:
            self.actor_encoder.para_freeze()
        if actor_mlp:
            self.actor_mlp.para_freeze(actor_last_layer)

        if self.use_critic_encoder and critic_encoder:
            self.critic_encoder.para_freeze()
        if critic_mlp:
            self.actor_mlp.para_freeze(critic_last_layer)
