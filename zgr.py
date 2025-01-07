"""
ZGR gradient estimator

There are several implementations, which should all be equivalent
The implementation in ZGR() below appears more simple, is numerically stable. 
The overloaded backward implementation in ZGR_Function class may save some memory
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
import math
import copy


def ZGR(logits, x=None):
    """Returns a categorical sample from Categorical softmax(logits) (over axis=-1) as a
    one-hot vector, with ZGR gradient.
    
    Input: 
    logits [*, C], where C is the number of categories
    x: (optional) categorical sample to use instead of drawing a new sample. [*], dtype=int64.
    
    Output: categorical samples with ZGR gradient [*,C] encoded as one_hot
    """
    # return ZGR_Function().apply(logits, x)
    # using surrogate loss
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()  # [*, C]
    dx_ST = p  # [*,C]
    index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()  # [*]
    num_classes = logits.shape[-1]
    y = F.one_hot(index, num_classes=num_classes).to(p)  # [*,C], same dtype as p
    logpx = logp.gather(-1, index.unsqueeze(-1)) # [*,1] -- log probability of drawn sample
    dx_RE = (y - p.detach()) * logpx
    dx = (dx_ST + dx_RE) / 2
    return y + (dx - dx.detach())


def ZGR_binary(logits:Tensor)->Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [*]
    Output: binary samples with ZGR gradient [*], dtype as logits
    """
    p = torch.sigmoid(logits)
    b = p.bernoulli()
    J = (b * (1-p) + (1-b)*p )/2
    return b + J.detach()*(logits - logits.detach()) # value of x with J on backprop to logits


""" implementation by specifying backward in torch.autograd.Function
"""
class ZGR_Function(torch.autograd.Function):
    @staticmethod
    # so this propagates forward 1-hot categorical
    def forward(ctx, logits, x=None):
        # x (*)  integer
        if x is None:
            x = torch.distributions.categorical.Categorical(logits=logits,validate_args=False).sample()
        num_classes = logits.shape[-1]
        D = F.one_hot(x, num_classes).float() # (*, num_classes)
        ctx.save_for_backward(logits, x)
        D.requires_grad = True
        return D

    @staticmethod
    # for back prop it lets the ongoing grad flow just pass without any change...
    def backward(ctx, J):
        # J [*,C] -- gradient in the 1-hot embedding
        logits, x = ctx.saved_tensors # [*,C], [*]
        p = F.softmax(logits, dim=-1) # [*,C]
        xu = x.unsqueeze(-1)
        # according to equation (50)
        J_logits = (J - J.gather(-1,xu)) * p / 2   # (*,C)
        s = J_logits.sum(dim=-1, keepdim=True) # (*,1)
        J_logits.scatter_add_(-1, xu, -s)
        return J_logits, None

        # according to equation (60)
        inner = (J*p).sum(dim=-1, keepdim=True)
        dx = (J*p - inner*p)/2
        vals = (J.gather(-1,xu) - inner)/2
        dx.scatter_add_(-1, xu, vals)
        dx -= vals * p   # correction to maintain zero drift (a constant grad in all scores)
        return dx, None


if __name__ == '__main__':
    torch.manual_seed(0)
    logits = torch.arange(12).reshape(2,-1).float()
    logits.requires_grad = True
    c = ZGR(logits)
    loss = torch.sum((c-1.0)**2)
    loss.backward()
    g = logits.grad
    print(g)
    print(g.sum(dim=-1))
