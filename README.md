# TACO: General Acrobatic Flight Control via Target-and-Command-Oriented Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2503.01125-b31b1b.svg)](https://arxiv.org/abs/2503.01125)
[![Conference](https://img.shields.io/badge/Conference-IROS%202025-blue.svg)](https://iros2025.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the implementation of **TACO (Target-and-Command-Oriented Reinforcement Learning)**, a novel framework for achieving general acrobatic flight control of Micro Aerial Vehicles (MAVs). The framework enables high-speed, high-accuracy circular flights and continuous multi-flips through a unified reinforcement learning approach.

## 🎯 Overview

TACO addresses two critical limitations in existing aerobatic flight control methods:
1. **Task Flexibility**: Unlike traditional methods restricted to specific maneuver trajectories, TACO supports online parameter adjustments and handles diverse aerobatic tasks within a unified framework.
2. **Sim-to-Real Transfer**: Our spectral normalization method with input-output rescaling enhances policy smoothness, independence, and symmetry, enabling zero-shot sim-to-real deployment.

## ✨ Key Features

- **Unified Framework**: Single framework handles multiple aerobatic maneuvers (hover, circle, flip)
- **Online Parameter Adjustment**: Real-time modification of flight parameters during execution
- **Zero-shot Sim-to-Real**: Advanced training techniques eliminate the sim-to-real gap
- **High Performance**: Achieves 4.2 rad/s angular velocity (1.6× faster than previous work) with 70° tilt angle
- **Continuous Multi-flips**: Stable 14+ continuous flips without altitude loss or stabilization pauses

## 🏗️ System Architecture

The TACO framework consists of three main components:

1. **TACO RL Framework**: Unified state design and reward functions for different maneuvers
2. **Simulation Environment**: High-fidelity MAV model with motor dynamics and aerodynamics
3. **Real MAV Platform**: Hardware implementation with onboard inference

### State Design

The unified state representation includes:
- **Task-oriented state** (14D): Relative position/orientation to target, task flags, commands
- **MAV-oriented state** (8D): Body velocity, angular velocity, altitude, battery voltage
- **Context-oriented state** (4D): Previous action for temporal consistency

### Training Method

- **Network**: L-layer fully connected network with spectral normalization
- **Algorithm**: PPO with 4096 parallel environments
- **Properties**: Temporal/spatial smoothness, independence, symmetry

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch
- IsaacGym
- CUDA (for GPU acceleration)

### Setup

please refer to the installation of IsaacGym

### Training

```python
python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=pos --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=20 --lipschitz_para=4&

python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=rotate --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=20 --lipschitz_para=4&

python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=flip --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time==20 --lipschitz_para=4&

python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=mix --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=20 --lipschitz_para=4&
```

### Evaluation

```python
python train_fpv_asymmetry_ppo.py --train_mode=testmodel --load_task_mode=pos --load_time=05-23-02-57
```

## 🎮 Supported Tasks

### 1. POS (Hover) Task
- **Objective**: Fly to and hover at desired position with specified yaw
- **Command**: None
- **Performance**: Precise position and attitude control

### 2. CIRCLE Task
- **Objective**: Rotate around center point with specified speed and radius
- **Command**: Tangential velocity (adjustable online)
- **Performance**: 1.2m radius at 5m/s, 4.2 rad/s angular velocity

### 3. FLIP Task
- **Objective**: Perform continuous flips around x-axis
- **Command**: flip radian remains to complete
- **Performance**: 14+ continuous flips, stable fix-point execution

## 📊 Experimental Results

### Real-world Performance

- **CIRCLE Task**: Achieved 1.2m radius at 5m/s with 70° tilt angle
- **FLIP Task**: Completed 14 continuous flips in 6.6s
- **Command Tracking**: Superior performance compared to MPC controllers

### Sim-to-Real Transfer

Our spectral normalization method demonstrates:
- **Spatial Smoothness**: Smooth action transitions across state space
- **Independence**: Unrelated actions remain unchanged during task execution
- **Symmetry**: Symmetric states produce symmetric actions
- **Temporal Smoothness**: Continuous action sequences over time

## 📁 Project Structure

```
IsaacGymEnvs/
├── isaacgymenvs/                    # Main package directory
│   ├── tasks/                       # Task implementations
│   │   ├── control/                 # Control-related modules
│   │   │   ├── task_reward.py       # Reward function implementations
│   │   │   ├── thrust_dynamics.py   # Thrust and motor dynamics
│   │   │   ├── angvel_control.py    # Angular velocity controller
│   │   │   ├── battery_dynamics.py  # Battery model
│   │   │   ├── fpv_dynamics.py      # FPV drone dynamics
│   │   │   └── logger.py            # Logging utilities
│   │   ├── base/                    # Base task classes
│   │   └── fpv_asymmetry.py         # Main TACO environment implementation
│   ├── cfg/                         # Configuration files
│   │   ├── Fpv_asymmetry_PPO_pos.yaml      # POS task configuration
│   │   ├── Fpv_asymmetry_PPO_rotate.yaml   # CIRCLE task configuration
│   │   └── Fpv_asymmetry_PPO_flip.yaml     # FLIP task configuration
│   └── utils/                       # Utility functions
│       ├── torch_jit_utils.py       # PyTorch JIT utilities
│       ├── utils.py                 # General utilities
│       └── dr_utils.py              # Domain randomization utilities
├── train/                           # Training scripts
│   ├── train_fpv_asymmetry_ppo.py   # Main training script
│   ├── start_train.sh               # Training start script
│   └── stop_train.sh                # Training stop script
├── algorithms/                      # RL algorithm implementations
│   ├── ppo_asymmetry.py             # PPO algorithm with asymmetry handling
│   ├── nets_asymmetry.py            # Neural network architectures
│   └── buffer_asymmetry.py          # Experience buffer implementation
├── assets/                          # 3D models and assets
├── docs/                            # Documentation
└── setup.py                         # Package setup file
```

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yin2025taco,
  title={TACO: General Acrobatic Flight Control via Target-and-Command-Oriented Reinforcement Learning},
  author={Yin, Zikang and Zheng, Canlun and Guo, Shiliang and Wang, Zhikun and Zhao, Shiyu},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  note={Accepted}
}
```

## 🔗 Related Links

- [Paper on arXiv](https://arxiv.org/abs/2503.01125)
- [Demo Videos](https://www.youtube.com/watch?v=x1v7nD2iHIk&ab_channel=WINDYLab)

---

