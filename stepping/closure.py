from typing import Any, Callable
import torch
from stepping.loss import LossCollector
from utils import pyramid

class Stepper():
    """
    Takes a dictionary of named inputs, a neural network and a loss collector.
    Provides a closure that can be used by a pytorch optimizer.
    """
    def __init__(self, sample: dict, model, loss_collector: LossCollector, optimizer: Any):
        self.sample = sample
        self.model = model
        self.loss_collector = loss_collector
        self.optimizer = optimizer

    def prepair_inputs(self):
        """
        Performs transformations on the raw sample passed in and populates 
        self.inputs with these transformations. E.g. scale pyramids.
        """
        self.inputs = {}

        if self.sample["pyramid"]:
            sample_pyr = pyramid.pyramid_from(self.sample["input"], self.sample["pyr_height"])

            for index in range(len(sample_pyr)):
                self.inputs["sample-{}".format(index)] = (sample_pyr[index], "{}-{}".format(self.sample["loss_name"], index))
        else:
            self.inputs["sample"] = (self.sample["input"], self.sample["loss_name"])

    def loss_fn(self) -> torch.Tensor:
        """
        For each input, runs that input through the model, collects the
        appropriate losses, calls backwards on that loss, and returns it.
        """
        self.optimizer.zero_grad()
        
        self.prepair_inputs()
        
        for input_name in self.inputs:
            inp, loss_name = self.inputs[input_name]
            self.model(inp)
            self.loss_collector.collect_losses_for(loss_name)
        
        loss = self.loss_collector.get_loss(reset=True)
        loss.backward()
        return loss

