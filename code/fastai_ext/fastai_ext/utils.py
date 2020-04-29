from matplotlib import pyplot as plt
from fastai.callback import Callback
from fastai.callbacks import hook_output

def request_lr(learn, **kwargs):
    learn.lr_find(**kwargs)
    learn.recorder.plot()#suggestion=False
    plt.show()
    return float(input('Select LR: '))

def auto_lr(learn, **kwargs):
    learn.lr_find(**kwargs)
    learn.recorder.plot()
    return learn.recorder.min_grad_lr


def transfer_from_dae(learn_cls, learn_dae):
    learn_cls.model.embeds.load_state_dict(learn_dae.model.embeds.state_dict())
    learn_cls.model.bn_cont.load_state_dict(learn_dae.model.bn_cont.state_dict())
    learn_cls.model.layers[:-1].load_state_dict(learn_dae.model.layers[:-1].state_dict())

def freeze_layer(m):
    for param in m.parameters(): param.requires_grad=False

def freeze_but_last(learn):
    freeze_layer(learn.model.embeds)
    freeze_layer(learn.model.bn_cont)
    freeze_layer(learn.model.layers[:-1])

def unfreeze_all(learn):
    for param in learn.model.parameters(): param.requires_grad=True

class StoreHook(Callback):
    def __init__(self, module):
        super().__init__()
        self.custom_hook = hook_output(module)
        self.outputs = []
        
    def on_batch_end(self, train, **kwargs): 
        if (not train): self.outputs.append(self.custom_hook.stored)
