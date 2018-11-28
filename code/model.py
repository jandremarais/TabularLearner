from fastai import *
from fastai.tabular import *
from fastai.callbacks.mixup import *


class TabularModel(nn.Module):
    "Basic model for tabular data."

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None,
                 emb_drop=0., y_range=None, use_bn=True, bn_final=False, 
                 act_func='relu', residual=False, mixup_aug=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        act_dict = {'relu': nn.ReLU(inplace=True), 'selu': nn.SELU(inplace=True)}
        act_func = act_dict[act_func]
        actns = [act_func] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)
        if get_layer_name(act_func)=='selu':
            for l in self.layers:
                if get_layer_name(l) == 'Linear':
                    selu_weights_init(l)
        self.residual = residual
        if residual: 
            res_lin = bn_drop_lin(self.n_emb+self.n_cont, sizes[-2], bn=False, p=dp, actn=act_func)
            self.res_lin = nn.Sequential(*res_lin)
        self.mixup_aug = mixup_aug

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont) -> Tensor:
        if self.n_emb != 0:
            if self.mixup_aug: 
                x = x_cat
            else:
                x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
                x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        if self.residual: x = self.layers[-1](self.res_lin(x) + self.layers[:-1](x))
        else: x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x
    
def selu_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.5 / math.sqrt(m.weight.numel()))

    elif classname.find('Linear') != -1:
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns

        m.weight.data.normal_(0.0, 1.0 / math.sqrt(fan_in))
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
        
def get_layer_name(m):
    return m.__class__.__name__

@dataclass
class TabularMixUpCallback(Callback):
    "Callback that creates the mixed-up input and target."
    learner:Learner
    alpha:float=0.4
    stack_x:bool=False
    stack_y:bool=True
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        last_cats, last_conts = last_input
        last_cats = torch.cat([e(last_cats[:,i]) for i,e in enumerate(self.learner.model.embeds)],1)
        if not train: return ([last_cats,last_conts], last_target) 
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_conts.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_conts.device)
        xcats1, xconts1, y1 = last_cats[shuffle], last_conts[shuffle], last_target[shuffle]
        if self.stack_x:
            # new_input = [last_input, last_input[shuffle], lambd]
            new_cats = [last_cats, last_cats[shuffle], lambd]
            new_conts = [last_conts, last_conts[shuffle], lambd]
        else: 
            # new_input = (last_input * lambd.view(lambd.size(0),1) + x1 * (1-lambd).view(lambd.size(0),1))
            new_cats = (last_cats * lambd.view(lambd.size(0),1) + xcats1 * (1-lambd).view(lambd.size(0),1))
            new_conts = (last_conts * lambd.view(lambd.size(0),1) + xconts1 * (1-lambd).view(lambd.size(0),1))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return ([new_cats,new_conts], new_target) 
    
def tabular_learner(data, layers, emb_szs=None, metrics=None,
                    ps=None, emb_drop=0., y_range=None, use_bn=True, 
                    act_func='relu', residual=False, mixup_aug=False, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range, use_bn=use_bn, act_func=act_func, residual=residual, mixup_aug=mixup_aug)
    l = Learner(data, model, metrics=metrics, **kwargs)
    if mixup_aug: 
        l.loss_func = MixUpLoss(l.loss_func)
        l.callback_fns.append(partial(TabularMixUpCallback, alpha=0.4))
    return l