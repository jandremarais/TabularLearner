from fastai.callback import Callback
from fastai.basic_train import Learner
from fastai.torch_core import torch
import numpy as np
import pdb

class SwapNoiseCallback(Callback):
    "Callback that creates the input with swap noise."
    def __init__(self, learner:Learner, alpha:float=0.4):
        self.learner = learner
        self.alpha = alpha
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return (last_input, last_target) 
        lambd = [inp.new(np.random.binomial(1,self.alpha, inp.shape)) for inp in last_input]
        shuffle = []
        for inp in last_input:
            shuffle.append(torch.cat([inp[torch.randperm(inp.size(0)).to(inp.device), i].view(-1,1) for i in range(inp.size(1))], 1))

        new_input = [last_input[i]*(1-lambd[i]) + shuffle[i]*lambd[i] for i in range(len(last_input))]
        
        #  new_conts = (last_conts * lambd.view(lambd.size(0),1) + xconts1 * (1-lambd).view(lambd.size(0),1))
        return (new_input, last_target) 