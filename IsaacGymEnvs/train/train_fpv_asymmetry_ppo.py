#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Custom PPO training script

All default parameters are loaded via yaml file and passed into cfg.
Args can override cfg.
Create environment and algorithm instances using cfg.

base train: python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=rotating --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM
eval: python train_fpv_asymmetry_ppo.py --train_mode=testmodel --load_task_mode=pos --load_time=05-23-02-57 --num_episodes=1000

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/2/21 下午16:30   yzk、ikun      1.0         None
"""

import os
import gym
import isaacgym
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
import numpy as np
import random
import torch
import torch.nn as nn
import yaml
import argparse

from algorithms.ppo_asymmetry import PPO, PPO_ActorCritic
import datetime


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print("Current seed is:", seed)

    return seed


def update_yaml(k, value, yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        old_data = yaml.load(f, Loader=yaml.FullLoader)
    old_data[k] = value  # 修改读取的数据（k存在就修改对应值，k不存在就新增一组键值对）
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(old_data, f, default_flow_style=False)


def find_closest_file(folder_path):
    current_time = datetime.datetime.now()
    closest_file = None
    min_time_difference = float('inf')

    for file_name in os.listdir(folder_path):
        try:
            file_time = datetime.datetime.strptime(file_name, "%m-%d-%H-%M")
            time_difference = abs((current_time - file_time).total_seconds())
            if time_difference < min_time_difference:
                min_time_difference = time_difference
                closest_file = file_name
        except ValueError:
            # Skip invalid file names
            continue
    return closest_file


def get_args():
    parser = argparse.ArgumentParser(description="RL Policy")

    # load and save dir
    '''
    mode \ operation | create model | load model | save model
    train 训练        | true         | false      | true       
    testmodel 测试模型| flase         | true        | false
    '''
    parser.add_argument("--train_mode", type=str,
                        default="train", help="Can be train, testmodel")
    # load
    parser.add_argument("--load_folder", type=str, default="", help="Additional load folder")
    parser.add_argument("--load_task_mode", type=str,
                        default="pos", help="Load model's task_mode")
    parser.add_argument("--load_exp", type=str, default="", help="Load model's exp suffix")
    parser.add_argument("--load_time", type=str, default="", help="Load model's time")
    parser.add_argument("--load_model_name", type=str, default='model_0.pt',
                        help="Load model's file name, model_0 is the global optimal model, model_1 is the optimal model when difficulty is 1")
    # save
    parser.add_argument("--save_folder", type=str, default="", help="Additional save folder")
    parser.add_argument("--save_exp", type=str, default="", help="Save model's exp suffix")

    # Task
    parser.add_argument("--task", type=str, default="Fpv",
                        help="Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, Quadcopter, ShadowHand, Ingenuity")
    # 以下参数默认值由yaml文件确定
    parser.add_argument("--num_envs", type=int, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes per environment run")
    parser.add_argument("--lenObservations", type=int, help="Number of frames for observation")
    parser.add_argument("--lenStates", type=int, help="Number of frames for state")
    parser.add_argument("--task_mode", type=str,
                        help="Can be pos, rotate, roll, flip, tictoc")

    # Fpv params
    parser.add_argument("--random_copter_pos", type=str, help="Whether to randomize the initial position of the UAV")
    parser.add_argument("--random_copter_quat", type=str, help="Whether to randomize the initial attitude of the UAV")
    parser.add_argument("--random_copter_vel", type=str, help="Whether to randomize the initial velocity of the UAV")

    # Target params
    parser.add_argument("--random_target_pos", type=str, help="Whether to randomize the initial position of the target")
    parser.add_argument("--random_target_yaw", type=str, help="Whether to randomize the initial yaw of the target")

    # Battery params
    parser.add_argument("--battery_consumption", type=str, help="Whether to consider battery voltage drop")
    parser.add_argument("--random_voltage", type=str, help="Whether to randomize the initial battery voltage")

    # rotor params
    parser.add_argument("--rotor_response_time", type=float, help="Motor response time")
    parser.add_argument("--rotor_noise", type=str, help="Whether to consider motor noise")
    parser.add_argument("--rotor_delay", type=str,
                        help="Whether to consider motor delay. Do not use this, use delay_time")
    parser.add_argument("--rotor_response", type=str, help="Whether to consider motor response")
    parser.add_argument("--random_rotordynamic_coe",
                        type=str, help="Whether to randomize motor response fitting parameters")
    parser.add_argument("--random_rotor_delay", type=str,
                        help="Whether to randomize motor delay time, now it's not used, there are other delay methods")
    parser.add_argument("--random_rotor_response",
                        type=str, help="Whether to randomize motor response time")
    parser.add_argument("--random_rotor_speed", type=str, help="Whether to consider randomizing the initial rotor speed")
    parser.add_argument("--random_aerodynamic_coe",
                        type=str, help="Whether to randomize aerodynamic fitting parameters")

    # Delay params
    parser.add_argument("--delay_time_max", type=int, help="Maximum simulation time for supported delay, in ms")
    parser.add_argument("--delay_time", type=int,
                        help="From the real time of the state, to the delay of the action that actually produces thrust, in ms")
    parser.add_argument("--ramdom_delay_time", type=str, help="Whether to randomize delay time")
    parser.add_argument("--ramdom_deploy_time", type=str,
                        help="Whether to randomize action execution time, default is 10ms for each action, but there is a certain deviation")

    # Command params
    parser.add_argument("--random_command", type=str, help="Whether to randomize the task")

    parser.add_argument("--observation_noise", type=str, help="Whether to consider observation noise")

    # net params
    parser.add_argument("--actor_hidden_sizes", nargs='+',
                        type=int, help="actor_hidden_sizes")
    parser.add_argument("--critic_hidden_sizes", nargs='+',
                        type=int, help="critic_hidden_sizes")
    parser.add_argument("--use_actor_encoder", type=str,
                        help="Whether actor uses encoder")
    parser.add_argument("--use_critic_encoder", type=str,
                        help="Whether critic uses encoder")
    parser.add_argument("--share_encoder", type=str,
                        help="Whether actor and critic share encoder, shared means everything is based on actor_encoder")
    parser.add_argument("--actor_encoder_type", type=str,
                        help="Can be LSTM，CNN，TCN，ATTENTION")
    parser.add_argument("--critic_encoder_type", type=str,
                        help="Can be LSTM，CNN，TCN，ATTENTION")

    # Learning params
    parser.add_argument("--clip", type=float)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--max_grad", type=float)

    parser.add_argument("--use_clipped_value_loss", type=str)
    parser.add_argument("--use_critic_tar", type=str)
    parser.add_argument("--polyak", type=float)

    parser.add_argument("--epochs", type=int,
                        help="Set a maximum number of training iterations")
    parser.add_argument("--horizon_len", type=int)
    parser.add_argument("--train_iters", type=float)
    # batch_size = horizon_len * num_Envs / mini_batch_num
    parser.add_argument("--mini_batch_num", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--pi_coef", type=float)
    parser.add_argument("--vf_coef", type=float)
    parser.add_argument("--ent_coef", type=float,
                        help="entropy loss parameter. The smaller std, the larger entropy loss. The larger the parameter, the more it encourages exploration")

    parser.add_argument("--learning_rate_schedule",
                        type=str, help="Whether to enable learning rate self-adjustment")
    parser.add_argument("--lr_ratio", type=float, help="Final learning rate to initial learning rate ratio")
    parser.add_argument("--lr_lp_index", type=float,
                        help="Learning process size when learning rate stops changing")
    parser.add_argument("--lr_epoch_index", type=float,
                        help="Epoch size when learning rate stops changing")

    parser.add_argument("--use_lipschitz", type=str,
                        help="Whether to adjust the parameter of lipschitz constraint based on training degree")
    parser.add_argument("--lipschitz_para", type=float, help="Initial value of lipschitz constant")
    parser.add_argument("--lipschitz_schedule", type=str, help="If closed, always take the final value")
    parser.add_argument("--lip_ratio", nargs='+', type=float,
                        help="Initial proportion and final proportion of lipschitz constant, actual parameter = lip_ratio × initial lipschitz constant")
    parser.add_argument("--lip_lp_index", nargs='+', type=float,
                        help="Two learning_process node values of lipschitz constraint parameter change")
    parser.add_argument("--lip_epoch_index", nargs='+',
                        type=float, help="Two epoch node values of lipschitz constraint parameter change")

    parser.add_argument("--difficulty_schedule",
                        type=str, help="Whether to adjust task difficulty based on training degree")
    parser.add_argument("--diff_value", nargs='+',
                        type=float, help="Initial task difficulty, final task difficulty")
    parser.add_argument("--diff_lp_index", nargs='+',
                        type=float, help="Two learning_process node values of task difficulty change")
    parser.add_argument("--diff_epoch_index", nargs='+',
                        type=float, help="Two epoch node values of task difficulty change")

    parser.add_argument("--device", type=str)

    args = parser.parse_args()

    return args


def process(args):
    # Set load and save paths
    absolute_path = os.path.abspath(os.path.dirname(__file__))
    load_path = ''  # Path for loading model during test or retrain
    save_path = ''  # Path for saving model during train or retrain
    record_path = ''  # Path for env, used to record UAV state during test
    if args.train_mode == "testmodel":
        temp = os.path.join(absolute_path, '../isaacgymenvs/runs',
                            args.load_folder, args.task + '_' + args.load_task_mode)
        temp = temp if args.load_exp == '' else temp + '_' + args.load_exp
        load_time_dir = find_closest_file(
            temp) if args.load_time == '' else args.load_time
        load_path = os.path.join(
            temp, load_time_dir, 'nn', args.load_model_name)
        record_path = os.path.join(temp, load_time_dir, 'records')

        # Load yaml file
        cfg_path = os.path.join(absolute_path, '../isaacgymenvs/cfg')
        with open(cfg_path + '/Fpv_asymmetry_PPO_' + args.load_task_mode + '.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.train_mode == "train":
        temp = os.path.join(absolute_path, '../isaacgymenvs/runs',
                            args.save_folder, args.task + '_' + args.task_mode)
        temp = temp if args.save_exp == '' else temp + '_' + args.save_exp
        save_time_dir = datetime.datetime.now().strftime("%m-%d-%H-%M")
        save_path = os.path.join(temp, save_time_dir)

        # Load yaml file
        cfg_path = os.path.join(absolute_path, '../isaacgymenvs/cfg')
        with open(cfg_path + '/Fpv_asymmetry_PPO_' + args.task_mode + '.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Override default parameters in yaml with args
    cfg['Task']['seed'] = args.seed if args.seed is not None else cfg['Task']['seed']
    cfg['Task']['seed'] = set_seed(cfg['Task']['seed'], False)

    cfg['Task']['env']['numEnvs'] = args.num_envs if args.num_envs is not None else cfg['Task']['env']['numEnvs']
    cfg['Task']['env']['maxEpisodeLength'] = args.num_episodes if args.num_episodes is not None else cfg['Task']['env']['maxEpisodeLength']
    cfg['Task']['env']['lenObservations'] = args.lenObservations if args.lenObservations is not None else cfg['Task']['env']['lenObservations']
    cfg['Task']['env']['lenStates'] = args.lenStates if args.lenStates is not None else cfg['Task']['env']['lenStates']
    cfg['Task']['observation_noise'] = True if args.observation_noise == 'True' else False if args.observation_noise == 'False' else cfg['Task']['observation_noise']

    if args.train_mode == "testmodel":
        cfg['Task']['task_mode'] = args.load_task_mode if args.load_task_mode is not None else cfg['Task']['task_mode']
    else:
        cfg['Task']['task_mode'] = args.task_mode if args.task_mode is not None else cfg['Task']['task_mode']
    cfg['Task']['name'] = cfg['Task']['name'] + '_' + cfg['Task']['task_mode']

    # Fpv params
    cfg['Task']['random_copter_pos'] = True if args.random_copter_pos == 'True' else False if args.random_copter_pos == 'False' else cfg['Task']['random_copter_pos']
    cfg['Task']['random_copter_quat'] = True if args.random_copter_quat == 'True' else False if args.random_copter_quat == 'False' else cfg['Task']['random_copter_quat']
    cfg['Task']['random_copter_vel'] = True if args.random_copter_vel == 'True' else False if args.random_copter_vel == 'False' else cfg['Task']['random_copter_vel']

    # Target params
    cfg['Task']['random_target_pos'] = True if args.random_target_pos == 'True' else False if args.random_target_pos == 'False' else cfg['Task']['random_target_pos']
    cfg['Task']['random_target_yaw'] = True if args.random_target_yaw == 'True' else False if args.random_target_yaw == 'False' else cfg['Task']['random_target_yaw']

    # Battery params
    cfg['Task']['battery_consumption'] = True if args.battery_consumption == 'True' else False if args.battery_consumption == 'False' else cfg['Task']['battery_consumption']
    cfg['Task']['random_voltage'] = True if args.random_voltage == 'True' else False if args.random_voltage == 'False' else cfg['Task']['random_voltage']

    # Motor params
    cfg['Task']['rotor_response_time'] = args.rotor_response_time if args.rotor_response_time is not None else cfg['Task']['rotor_response_time']
    cfg['Task']['rotor_noise'] = True if args.rotor_noise == 'True' else False if args.rotor_noise == 'False' else cfg['Task']['rotor_noise']
    cfg['Task']['rotor_delay'] = True if args.rotor_delay == 'True' else False if args.rotor_delay == 'False' else cfg['Task']['rotor_delay']
    cfg['Task']['rotor_response'] = True if args.rotor_response == 'True' else False if args.rotor_response == 'False' else cfg['Task']['rotor_response']
    cfg['Task']['random_rotordynamic_coe'] = True if args.random_rotordynamic_coe == 'True' else False if args.random_rotordynamic_coe == 'False' else cfg['Task']['random_rotordynamic_coe']
    cfg['Task']['random_rotor_delay'] = True if args.random_rotor_delay == 'True' else False if args.random_rotor_delay == 'False' else cfg['Task']['random_rotor_delay']
    cfg['Task']['random_rotor_response'] = True if args.random_rotor_response == 'True' else False if args.random_rotor_response == 'False' else cfg['Task']['random_rotor_response']
    cfg['Task']['random_rotor_speed'] = True if args.random_rotor_speed == 'True' else False if args.random_rotor_speed == 'False' else cfg['Task']['random_rotor_speed']
    cfg['Task']['random_aerodynamic_coe'] = True if args.random_aerodynamic_coe == 'True' else False if args.random_aerodynamic_coe == 'False' else cfg['Task']['random_aerodynamic_coe']

    # Delay params
    cfg['Task']['delay_time_max'] = args.delay_time_max if args.delay_time_max is not None else cfg['Task']['delay_time_max']
    cfg['Task']['delay_time'] = args.delay_time if args.delay_time is not None else cfg['Task']['delay_time']
    cfg['Task']['ramdom_delay_time'] = True if args.ramdom_delay_time == 'True' else False if args.ramdom_delay_time == 'False' else cfg['Task']['ramdom_delay_time']
    cfg['Task']['ramdom_deploy_time'] = True if args.ramdom_deploy_time == 'True' else False if args.ramdom_deploy_time == 'False' else cfg['Task']['ramdom_deploy_time']

    # Command params
    cfg['Task']['random_command'] = True if args.random_command == 'True' else False if args.random_command == 'False' else cfg['Task']['random_command']

    if args.train_mode == "train":  # Training
        cfg['IsaacGym']['headless'] = True
        cfg['Task']['difficulty'] = cfg['PPO']['diff_value'][0]
        cfg['Task']['record_flag'] = False
        cfg['Task']['record_path'] = ''

    elif args.train_mode == "testmodel":  # Test model
        cfg['IsaacGym']['headless'] = False
        cfg['Task']['env']['numEnvs'] = 12
        cfg['Task']['difficulty'] = 1
        cfg['Task']['record_flag'] = True
        cfg['Task']['record_path'] = record_path
        cfg['Task']['random_rotor_speed'] = True
        cfg['Task']['observation_noise'] = True
        cfg['Task']['rotor_noise'] = False

        # Uncomment the following code for strict model comparison
        # cfg['Task']['env']['numEnvs'] = 200
        cfg['Task']['random_rotor_delay'] = False
        cfg['Task']['random_voltage'] = True
        cfg['Task']['env']['maxEpisodeLength'] = 1000

        cfg['Task']['random_copter_pos'] = False
        cfg['Task']['random_copter_quat'] = True
        cfg['Task']['random_copter_vel'] = True
        cfg['Task']['random_copter_dynamics'] = False
        cfg['Task']['random_target_pos'] = False
        cfg['Task']['random_target_yaw'] = False
        cfg['Task']['random_command'] = True
        cfg['Task']['random_rotordynamic_coe'] = False
        cfg['Task']['random_rotor_delay'] = False
        cfg['Task']['random_rotor_response'] = False
        cfg['Task']['random_rotor_speed'] = False
        cfg['Task']['random_aerodynamic_coe'] = False
        cfg['Task']['battery_consumption'] = False
        # cfg['Task']['aerodynamic_mode'] = 'lift only'

    else:
        print("Invalid train mode")

    # Create environment
    env = isaacgym_task_map[cfg['Task']['name']](
        cfg=cfg['Task'],
        rl_device=cfg['IsaacGym']['rl_device'],
        sim_device=cfg['IsaacGym']['sim_device'],
        graphics_device_id=cfg['IsaacGym']['graphics_device_id'],
        headless=cfg['IsaacGym']['headless'],
        virtual_screen_capture=cfg['IsaacGym']['capture_video'],
        force_render=cfg['IsaacGym']['force_render'],
    )
    print("Observation space is", env.num_obs)
    print("State space is", env.num_states)
    print("Action space is", env.num_acts)

    cfg['PPO']['actor_critic_dict']["actor_critic_mlp_dict"]['actor_hidden_sizes'] = args.actor_hidden_sizes if args.actor_hidden_sizes is not None else \
        cfg['PPO']['actor_critic_dict']["actor_critic_mlp_dict"]['actor_hidden_sizes']
    cfg['PPO']['actor_critic_dict']["actor_critic_mlp_dict"]['critic_hidden_sizes'] = args.critic_hidden_sizes if args.critic_hidden_sizes is not None else \
        cfg['PPO']['actor_critic_dict']["actor_critic_mlp_dict"]['critic_hidden_sizes']
    cfg['PPO']['actor_critic_dict']['use_actor_encoder'] = True if args.use_actor_encoder == 'True' else False if args.use_actor_encoder == 'False' else cfg['PPO']['actor_critic_dict'][
        'use_actor_encoder']
    cfg['PPO']['actor_critic_dict']['use_critic_encoder'] = True if args.use_critic_encoder == 'True' else False if args.use_critic_encoder == 'False' else cfg['PPO']['actor_critic_dict'][
        'use_critic_encoder']
    cfg['PPO']['actor_critic_dict']['share_encoder'] = True if args.share_encoder == 'True' else False if args.share_encoder == 'False' else cfg['PPO']['actor_critic_dict']['share_encoder']
    cfg['PPO']['actor_critic_dict']['actor_encoder_type'] = args.actor_encoder_type if args.actor_encoder_type is not None else cfg['PPO']['actor_critic_dict']['actor_encoder_type']
    cfg['PPO']['actor_critic_dict']['critic_encoder_type'] = args.critic_encoder_type if args.critic_encoder_type is not None else cfg['PPO']['actor_critic_dict']['critic_encoder_type']

    cfg['PPO']['actor_critic_dict']['actor_critic_mlp_dict'].update({
        # actor's input like task type, command are regarded as observation, flatten multi-frame into network
        'actor_input_dim': env.num_obs * env.len_obs,
        'actor_output_dim': env.num_acts,  # actor's output is regarded as action, so output dimension equals action dimension
        # critic's input like task type, command, privileged information are regarded as state, flatten multi-frame into network
        'critic_input_dim': env.num_states * env.len_states,
        'critic_output_dim': 1,  # critic's output dimension equals 1. Unless q is decomposed into v and a, it will not change
        'activation': nn.ReLU,
    })

    if cfg['PPO']['actor_critic_dict']['use_actor_encoder']:
        if cfg['PPO']['actor_critic_dict']['actor_encoder_type'] == 'CNN':
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'] = cfg['PPO']['actor_critic_dict']['actor_CNN_dict']
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['actor_encoder_type'],
                'input_size': env.num_obs,
                'output_size': env.num_obs,
            })
        elif cfg['PPO']['actor_critic_dict']['actor_encoder_type'] == 'TCN':
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'] = cfg['PPO']['actor_critic_dict']['actor_TCN_dict']
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['actor_encoder_type'],
                'input_size': env.num_obs,
                'output_size': env.num_obs,
            })
        elif cfg['PPO']['actor_critic_dict']['actor_encoder_type'] == 'LSTM':
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'] = cfg['PPO']['actor_critic_dict']['actor_LSTM_dict']
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['actor_encoder_type'],
                'input_size': env.num_obs,
            })
        elif cfg['PPO']['actor_critic_dict']['actor_encoder_type'] == 'ATTENTION':
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'] = cfg['PPO']['actor_critic_dict']['actor_ATTENTION_dict']
            cfg['PPO']['actor_critic_dict']['actor_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['actor_encoder_type'],
                'input_size': env.num_obs,
                'time_len': env.len_obs,
            })

    if cfg['PPO']['actor_critic_dict']['use_critic_encoder']:
        if cfg['PPO']['actor_critic_dict']['critic_encoder_type'] == 'CNN':
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'] = cfg['PPO']['actor_critic_dict']['critic_CNN_dict']
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['critic_encoder_type'],
                'input_size': env.num_states,
                'output_size': env.num_states,
            })
        elif cfg['PPO']['actor_critic_dict']['critic_encoder_type'] == 'TCN':
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'] = cfg['PPO']['actor_critic_dict']['critic_TCN_dict']
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['critic_encoder_type'],
                'input_size': env.num_states,
                'output_size': env.num_states,
            })
        elif cfg['PPO']['actor_critic_dict']['critic_encoder_type'] == 'LSTM':
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'] = cfg['PPO']['actor_critic_dict']['critic_LSTM_dict']
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['critic_encoder_type'],
                'input_size': env.num_states,
            })
        elif cfg['PPO']['actor_critic_dict']['critic_encoder_type'] == 'ATTENTION':
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'] = cfg['PPO']['actor_critic_dict']['critic_ATTENTION_dict']
            cfg['PPO']['actor_critic_dict']['critic_encoder_dict'].update({
                'encoder_type': cfg['PPO']['actor_critic_dict']['critic_encoder_type'],
                'input_size': env.num_states,
                'time_len': env.len_states,
            })

    cfg['PPO']['clip'] = args.clip if args.clip is not None else cfg['PPO']['clip']
    cfg['PPO']['target_kl'] = args.target_kl if args.target_kl is not None else cfg['PPO']['target_kl']
    cfg['PPO']['lam'] = args.lam if args.lam is not None else cfg['PPO']['lam']
    cfg['PPO']['max_grad'] = args.max_grad if args.max_grad is not None else cfg['PPO']['max_grad']

    cfg['PPO']['use_clipped_value_loss'] = True if args.use_clipped_value_loss == 'True' else False if args.use_clipped_value_loss == 'False' else cfg['PPO']['use_clipped_value_loss']

    cfg['PPO']['epochs'] = args.epochs if args.epochs is not None else cfg['PPO']['epochs']
    cfg['PPO']['horizon_len'] = args.horizon_len if args.horizon_len is not None else cfg['PPO']['horizon_len']
    cfg['PPO']['train_iters'] = args.train_iters if args.train_iters is not None else cfg['PPO']['train_iters']
    cfg['PPO']['mini_batch_num'] = args.mini_batch_num if args.mini_batch_num is not None else cfg['PPO']['mini_batch_num']
    cfg['PPO']['gamma'] = args.gamma if args.gamma is not None else cfg['PPO']['gamma']

    cfg['PPO']['lr'] = args.lr if args.lr is not None else cfg['PPO']['lr']
    cfg['PPO']['pi_coef'] = args.pi_coef if args.pi_coef is not None else cfg['PPO']['pi_coef']
    cfg['PPO']['vf_coef'] = args.vf_coef if args.vf_coef is not None else cfg['PPO']['vf_coef']
    cfg['PPO']['ent_coef'] = args.ent_coef if args.ent_coef is not None else cfg['PPO']['ent_coef']

    cfg['PPO']['learning_rate_schedule'] = True if args.learning_rate_schedule == 'True' else False if args.learning_rate_schedule == 'False' else cfg['PPO']['learning_rate_schedule']
    cfg['PPO']['lr_ratio'] = args.lr_ratio if args.lr_ratio is not None else cfg['PPO']['lr_ratio']
    cfg['PPO']['lr_lp_index'] = args.lr_lp_index if args.lr_lp_index is not None else cfg['PPO']['lr_lp_index']
    cfg['PPO']['lr_epoch_index'] = args.lr_epoch_index if args.lr_epoch_index is not None else cfg['PPO']['lr_epoch_index']

    cfg['PPO']['use_lipschitz'] = True if args.use_lipschitz == 'True' else False if args.use_lipschitz == 'False' else cfg['PPO']['use_lipschitz']
    cfg['PPO']['lipschitz_para'] = args.lipschitz_para if args.lipschitz_para is not None else cfg['PPO']['lipschitz_para']
    cfg['PPO']['lipschitz_schedule'] = True if args.lipschitz_schedule == 'True' else False if args.lipschitz_schedule == 'False' else cfg['PPO']['lipschitz_schedule']
    cfg['PPO']['lip_ratio'] = args.lip_ratio if args.lip_ratio is not None else cfg['PPO']['lip_ratio']
    cfg['PPO']['lip_lp_index'] = args.lip_lp_index if args.lip_lp_index is not None else cfg['PPO']['lip_lp_index']
    cfg['PPO']['lip_epoch_index'] = args.lip_epoch_index if args.lip_epoch_index is not None else cfg['PPO']['lip_epoch_index']

    cfg['PPO']['difficulty_schedule'] = True if args.difficulty_schedule == 'True' else False if args.difficulty_schedule == 'False' else cfg['PPO']['difficulty_schedule']
    cfg['PPO']['diff_value'] = args.diff_value if args.diff_value is not None else cfg['PPO']['diff_value']
    cfg['PPO']['diff_lp_index'] = args.diff_lp_index if args.diff_lp_index is not None else cfg['PPO']['diff_lp_index']
    cfg['PPO']['diff_epoch_index'] = args.diff_epoch_index if args.diff_epoch_index is not None else cfg['PPO']['diff_epoch_index']

    cfg['PPO']['device'] = args.device if args.device is not None else cfg['PPO']['device']

    algorithm = PPO(
        env=env,
        actor_critic=PPO_ActorCritic,
        actor_critic_para_dict=cfg['PPO']['actor_critic_dict'],
        clip=cfg['PPO']['clip'],
        target_kl=cfg['PPO']['target_kl'],
        lam=cfg['PPO']['lam'],
        max_grad=cfg['PPO']['max_grad'],

        use_clipped_value_loss=cfg['PPO']['use_clipped_value_loss'],

        epochs=cfg['PPO']['epochs'],
        horizon_len=cfg['PPO']['horizon_len'],
        train_iters=cfg['PPO']['train_iters'],
        mini_batch_num=cfg['PPO']['mini_batch_num'],
        gamma=cfg['PPO']['gamma'],
        seed=cfg['Task']['seed'],

        lr=cfg['PPO']['lr'],
        pi_coef=cfg['PPO']['pi_coef'],
        vf_coef=cfg['PPO']['vf_coef'],
        ent_coef=cfg['PPO']['ent_coef'],
        imit_coef=cfg['PPO']['imit_coef'],

        learning_rate_schedule=cfg['PPO']['learning_rate_schedule'],
        lr_ratio=cfg['PPO']['lr_ratio'],
        lr_lp_index=cfg['PPO']['lr_lp_index'],
        lr_epoch_index=cfg['PPO']['lr_epoch_index'],

        use_lipschitz=cfg['PPO']['use_lipschitz'],
        lipschitz_para=cfg['PPO']['lipschitz_para'],
        lipschitz_schedule=cfg['PPO']['lipschitz_schedule'],
        lip_ratio=cfg['PPO']['lip_ratio'],
        lip_lp_index=cfg['PPO']['lip_lp_index'],
        lip_epoch_index=cfg['PPO']['lip_epoch_index'],

        difficulty_schedule=cfg['PPO']['difficulty_schedule'],
        diff_value=cfg['PPO']['diff_value'],
        diff_lp_index=cfg['PPO']['diff_lp_index'],
        diff_epoch_index=cfg['PPO']['diff_epoch_index'],

        log_dir=save_path,
        log_interval=cfg['PPO']['epochs'] / 20,
        is_testing=args.train_mode == "testmodel",
        device=cfg['PPO']['device']
    )

    # As long as it's not testing, save training parameters (for training or retraining)
    if args.train_mode == "train" or args.train_mode == "retrain":
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/training_params.yaml', 'w') as f:
            yaml.dump(cfg, f)

    if args.train_mode == "testmodel":
        print('load model from {}'.format(load_path))
        algorithm.agent = torch.load(load_path)
        print(algorithm.agent)
        algorithm.agent.to(algorithm.device)

    if args.train_mode == "retrain":
        print('load model from {}'.format(load_path))
        algorithm.agent.load_state_dict(torch.load(load_path))
        print(algorithm.agent)
        algorithm.agent.to(algorithm.device)
        algorithm.agent.reset_parameters(reset_actor_encoder=False, reset_actor_mlp=True, reset_actor_last_layer_only=True, reset_actor_std=True,
                                         reset_critic_encoder=False, reset_critic_mlp=True, reset_critic_last_layer_only=True)
        algorithm.agent.freeze_para(actor_encoder=True, actor_mlp=True, actor_last_layer=False,
                                    critic_encoder=True, critic_mlp=True, critic_last_layer=False)

    return env, algorithm


if __name__ == "__main__":
    args = get_args()
    env, algorithm = process(args)
    algorithm.run()
