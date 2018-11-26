from typing import Any
from collections import defaultdict

class Hook():
    features=None
    def __init__(self, layer): 
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def close(self): 
        self.hook.remove()

class HookBag():
    """
    Contains all the hooks from a given model. Stores groups of hooks.
    """
    def __init__(self):
        self.hooks = defaultdict(lambda: [])
    
    def get_hooks(self, hook_name: str) -> list:
        return self.hooks[hook_name]
    
    def get_hook(self, hook_name: str) -> Hook:
        return self.hooks[hook_name]

    def add_hooks(self, hook_name: str, hooks: list):
        self.hooks[hook_name] += hooks
    
    def add_hook(self, hook_name: str, hook: Hook):
        self.hooks[hook_name] += [hook]

class Hooker():
    """
    Takes in a model. Attaches hooks to that model and stores
    them in a HookBag. Keeps track of the highest layer hooked
    so we can clip the model easily. 
    """
    def __init__(self, model: Any):
        self.model = model
        self.layers = list(model.children())
        self.hook_bag = HookBag()
        self.highest_hook = 0

    def attach_hooks(self, hook_group: str, layer_indices: list):
        self.hook_bag.add_hooks(hook_group, self.attached_hooks(layer_indices))

    def attached_hooks(self, layer_indices: list) -> list:
        hooks = []

        for index in layer_indices:
            if index > self.highest_hook:
                self.highest_hook = index
            
            hooks.append(Hook(self.layers[index]))

        return hooks
    
    def get_bag(self) -> HookBag:
        return self.hook_bag

    def last_hooked_index(self) -> int:
        return self.highest_hook