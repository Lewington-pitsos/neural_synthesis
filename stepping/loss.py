from collections import defaultdict
from typing import Any, Callable

class LossFn():
    """
    Keeps track of a list of target values and a number of hooks. 
    Has a loss function, which should take the target values, and a 
    list of feature maps and return a scalar loss value.
    """
    def __init__(self, target: Any, hooks: list, loss_fn: Callable):
        self.target = target
        self.hooks = hooks
        self.loss_fn = loss_fn
    
    def loss(self) -> int:
        feature_maps = [hook.features for hook in self.hooks]
        return self.loss_fn(self.target, feature_maps)

class LossCollector():
    """
    Holds a bunch of loss functions, each under at group. Can execute
    all loss functions or a group at a time. Each time adding their 
    losses to a running tally. Can return that running tally
    """
    def __init__(self):
        self.total_loss = 0
        self.loss_fns = defaultdict(lambda: [])
    
    def add_loss_fn(self, name: str, fn: LossFn):
        self.loss_fns[name] += [fn]
    
    def collect_losses_for(self, name) -> int:
        current_loss = 0

        for loss_fn in self.loss_fns[name]:
            current_loss += loss_fn.loss()
        
        self.total_loss += current_loss
        return current_loss
    
    def collect_total_loss(self) -> int:
        for name in self.loss_fns:
            self.collect_losses_for(name)
        
        return self.total_loss

    def get_loss(self) -> int:
        return self.total_loss
    
    def reset(self):
        self.total_loss = 0