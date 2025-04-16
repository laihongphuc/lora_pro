import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import einops
from jaxtyping import Float

class AdamW:
    def __init__(
        self,
        model_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
        T_max=1000,
        L2_regularization=False,
    ):
        self.lr = lr
        self.original_lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.model_params = list(model_params)
        self.n_steps = 0
        self.L2_regularization = L2_regularization
        self.T_max = T_max
        self.avg_grads = [torch.zeros_like(p.data) for p in self.model_params]
        self.avg_sq_grads = [torch.zeros_like(p.data) for p in self.model_params]
    
    def set_warmup(self):
        if self.n_steps == self.T_max:
            self.T_max *= 2
            self.n_steps = 0
        schedule_mul = 0.5 + 0.5 * np.cos(self.n_steps / self.T_max * np.pi)
        self.lr = self.original_lr * schedule_mul

        


    def zero_grad(self, set_to_none: bool = False):
        for p in self.model_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.data.zero_()
    
    @torch.no_grad()
    def step(self):
        self.set_warmup()
        self.n_steps += 1
        for param, avg_grad, avg_sq_grad in zip(self.model_params, self.avg_grads, self.avg_sq_grads):
            grad = param.grad
            # apply L2 regularization
            if self.L2_regularization:
                grad += self.weight_decay * param.data
            # update moving averages
            avg_grad.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            avg_sq_grad.mul_(self.beta2).add_(grad**2, alpha=1 - self.beta2)
            # compute bias-corrected moments
            bias_correction1 = 1 - self.beta1 ** self.n_steps
            bias_correction2 = 1 - self.beta2 ** self.n_steps
            avg_grad_corr = avg_grad / bias_correction1
            avg_sq_grad_corr = avg_sq_grad / bias_correction2
            
            # apply AdamW update
            std = (avg_sq_grad_corr.sqrt() + self.eps).sqrt()
            param.sub_(self.lr * avg_grad_corr / std + self.weight_decay * param.data)
            