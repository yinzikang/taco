"""
This function show the fpv power dynamics
real dyanmics:
            ^X
    2(ccw)   |   1(cw)
      Y<-----Z up-----
    3(cw)    |   0(ccw)
T  = f0 + f1 + f2 + f3
tx = (-f0 - f1 + f2 + f3) * sqrt(2)/2 * d
ty = (f0 - f1 - f2 + f3) * sqrt(2)/2 * d
fpv： tz = (-f0 + f1 - f2 + f3) * k
vpp： tz = (-|f0| + |f1| - |f2| + |f3|) * k
d：螺旋桨到中心直线距离
k：力矩与力比例

visual dynamics:
             ^X
    0(ccw)   |   3(cw)
      Y<---- Z up-----
    1(cw)    |   2(ccw)
"""

import torch


class FpvDynamicsReal2Sim():
    def __init__(self):
        self.weight = torch.tensor([
            [1, -1, 1, -1],
            [1, -1, -1, 1],
            [1, 1, -1, -1],
            [1, 1, 1, 1]
        ], dtype=torch.float32, device=torch.device('cuda:0'))

    def control_allocator(self, u):
        # 基于经验的fpv效率矩阵，没有考虑轴距以及力矩到力的系数
        # 截断
        u[:, 3] = torch.clip(u[:, 3], - u[:, 0] / 2, u[:, 0] / 2)
        # 分配
        f = torch.matmul(u, self.weight.T)
        # 饱和，如果某一个电机的指令大于1000,则所有电机均减去超出量
        f -= torch.clamp((f - 1000).max(dim=1, keepdim=True)[0], 0)
        # 放缩，感觉后面的1000并没有什么意义，因为上面已经限制了最大值
        f = torch.clip(f, 100, 1000)

        return f

    def sim_process(self, forces, torques):
        # 交换0和2列，以及1和3列
        forces[:, [0, 2, 1, 3]] = forces[:, [2, 0, 3, 1]]
        torques[:, [0, 2, 1, 3]] = torques[:, [2, 0, 3, 1]]
        # 根据电机旋转方向对力矩取负
        torques[:, 0] = - torques[:, 0]
        torques[:, 2] = - torques[:, 2]

        return forces, torques
