from collections import defaultdict

class Hook():
    features=None
    def __init__(self, layer): 
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def close(self): 
        self.hook.remove()


def attach_hooks(model, layer_indices):
  all_layers = list(model.children())

  return [Hook(all_layers[index]) for index in layer_indices]

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
        self.hooks["hook_name"] += hooks
    
    def add_hook(self, hook_name: str, hook: Hook):
        self.hooks["hook_name"] += [hook]

        
