from collections import defaultdict
import torch
from typing import Any, Callable

class LossFn():
    """
    Keeps track of a list of target values and a number of hooks. 
    Has a loss function, which should take the target values, and a 
    list of feature maps and return a scalar loss value.
    """
    def __init__(self, target: list, hooks: list, loss_fn: Callable, scale: int=1):
        self.__scale = scale
        self.__target = target
        self.__hooks = hooks
        self.__loss_fn = loss_fn
    
    def loss(self) -> torch.Tensor:
        feature_maps = [hook.features for hook in self.__hooks]
        return self.__loss_fn(self.__target, feature_maps) * self.__scale

class LossCollector():
    """
    Holds a bunch of loss functions, each under at group. Can execute
    all loss functions or a group at a time. Each group should correspond
    to and be named after a partricular input Each time adding their 
    losses to a running tally. Can return that running tally. 
    """
    def __init__(self):
        self.loss_fns = defaultdict(lambda: [])
        self.reset()
        self.losses = []
    
    def add_loss_fn(self, name: str, fn: LossFn):
        self.loss_fns[name] += [fn]
    
    def collect_losses_for(self, name):
        if self.loss_fns[name] == []:
            raise ValueError("Name does not match any loss group.")

        for loss_fn in self.loss_fns[name]:
            self.losses.append(loss_fn.loss())
    
    def collect_total_loss(self):
        for name in self.loss_fns:
            self.collect_losses_for(name)

    def get_loss(self, reset: bool=False) -> torch.Tensor:
        loss = sum(self.losses)
        if reset:
            self.reset()
        return loss

    def reset(self):
        self.losses = []