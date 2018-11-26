from typing import Any, Callable
import torch
from stepping.loss import LossCollector

class Stepper():
    """
    Takes a dictionary of named inputs, a neural network and a loss collector.
    Provides a closure that can be used by a pytorch optimizer.
    """
    def __init__(self, inputs: dict, model, loss_collector: LossCollector, optimizer: Any):
        self.inputs = inputs
        self.model = model
        self.loss_collector = loss_collector
        self.optimizer = optimizer

    def update_inputs(self, new_inputs):
        self.inputs = new_inputs

    def loss_fn(self) -> torch.Tensor:
        """
        For each input, runs that input through the model, collects the
        appropriate losses, calls backwards on that loss, and returns it.
        """
        self.optimizer.zero_grad()

        for input_name in self.inputs:
            inp, loss_name = self.inputs[input_name]
            self.model(inp)
            self.loss_collector.collect_losses_for(loss_name)
        
        loss = self.loss_collector.get_loss(reset=True)
        loss.backward()
        return loss

