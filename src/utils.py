from src.launchers import *
import numpy as np
import os
import torch
import random


def stack_tensor_dict_list(tensor_dict_list):
    """Stack a list of dictionaries of {tensors or dictionary of tensors}.
    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.
    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = stack_tensor_dict_list(dict_list)
        else:
            v = np.array(dict_list)

        ret[k] = v

    return ret


def seed_everything(seed=0):
    """Set random seed"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_benchmark_by_name(algo_name, env_name):
    algo = eval(f"train_{algo_name}_{env_name}")
    return algo
