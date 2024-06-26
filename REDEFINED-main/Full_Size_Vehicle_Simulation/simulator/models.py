import torch
from redefined_opt import distance, sign, distance_gradient

def make_compiled_sdf_net():
    sdf_net = torch.compile(distance(), dynamic=True, mode="max-autotune")
    return sdf_net

def make_compiled_sign_net():
    sign_net = torch.compile(sign(), dynamic=True, mode="max-autotune")
    return sign_net

def make_compiled_gradient_net():
    gradient_net = torch.compile(distance_gradient(), dynamic=True, mode="max-autotune")
    return gradient_net

distance_net = make_compiled_sdf_net()
sign_net = make_compiled_sign_net()
gradient_net = make_compiled_gradient_net()
