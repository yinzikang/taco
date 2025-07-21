#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""保存各种数据的logger

来源自spinup/utils/logx，负责保存各种超参数文件、训练指标、环境状态变化
嫌弃针对一个episode的env内的信息保存不好，自己改

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/26/23 12:07 PM   yinzikang      1.0         None
"""
import os
import json
import numpy as np


class Logger():
    """
    仿照EpochLogger的自定义EpisodeLogger，用于保存一个episode中的状态变化
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.episode_dict = dict()
        self.dump_index = 0

    def store_buffer(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.episode_dict.keys()):
                self.episode_dict[k] = []
            self.episode_dict[k].append(v)

    def dump_buffer(self):
        if self.dump_index < 5:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if bool(self.episode_dict):
                for k, v in self.episode_dict.items():
                    np.save(self.output_dir + '/' + k + str(self.dump_index), np.array(v))
                    np.savetxt(self.output_dir + '/' + k + str(self.dump_index) + ".csv", np.array(v), delimiter=',')  # 记录数据只能是二维的，observation容易出错
                print('log has been dumped to ' + self.output_dir)
                self.dump_index += 1

    def reset_buffer(self):
        self.episode_dict = dict()


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    """复杂的类型不会经过任何支路，直接输出str(obj)
        很多类型如类、方法都是有默认的str的，所以不用操心，只需要把自己想特别处理的加上即可
    """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif isinstance(obj, np.ndarray):
            return convert_json(obj.tolist())

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
