from typing import Callable
import torch
from loss import LossCollector

class Stepper():
    """
    Takes a dictionary of named inputs, a neural network and a loss collector.
    Provides a closure that can be used by a pytorch optimizer.
    """
    def __init__(self, inputs: dict, model, loss_collector: LossCollector):
        self.inputs = inputs
        self.model = model
        self.loss_collector = loss_collector

    def update_inputs(self, new_inputs):
        self.inputs = new_inputs

    def loss_fn(self) -> torch.Tensor:
        """
        For each input, runs that input through the model, collects the
        appropriate losses, calls backwards on that loss, and returns it.
        """
        for input_name in self.inputs:
            self.model(self.inputs[input_name])
            self.loss_collector.collect_losses_for(input_name)
        
        loss = self.loss_collector.get_loss(reset=True)
        loss.backwards()
        return loss

