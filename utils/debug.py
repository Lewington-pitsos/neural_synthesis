import torch
import gc

def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def tensor_count():
    count = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            count += 1
    print("number of tensors stored in memory: {}".format(count))