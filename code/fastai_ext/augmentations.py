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


class TabularMixUpCallback(Callback):
    "Callback that creates the mixed-up input and target for tabular data."
    def __init__(self, learner:Learner, alpha:float=0.4, stack_y:bool=True):
        self.learner, self.alpha, self.stack_y = learner, alpha, stack_y
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        last_cats, last_conts = last_input
        last_cats = torch.cat([e(last_cats[:,i]) for i,e in enumerate(self.learner.model.embeds)],1)
        
        if not train: return ([last_cats,last_conts], last_target) 
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_conts.new(lambd)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_conts.device)
        xcats1, xconts1, y1 = last_cats[shuffle], last_conts[shuffle], last_target[shuffle]
        new_cats = (last_cats * lambd.view(lambd.size(0),1) + xcats1 * (1-lambd).view(lambd.size(0),1))
        new_conts = (last_conts * lambd.view(lambd.size(0),1) + xconts1 * (1-lambd).view(lambd.size(0),1))
        
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return ([new_cats,new_conts], new_target) 