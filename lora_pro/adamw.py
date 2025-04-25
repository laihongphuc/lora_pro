import math
import random
from typing import Optional, Tuple, Union, Dict, List, Iterable

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
        self.model_params = [param for param in model_params if param.requires_grad]
        self.n_warmup_steps = 0
        self.global_steps = 0
        self.L2_regularization = L2_regularization
        self.T_max = T_max
        self.avg_grads = {}
        self.avg_sq_grads = {}
    
    def set_warmup(self):
        if self.n_warmup_steps == self.T_max:
            self.T_max *= 2
            self.n_warmup_steps = 0
        schedule_mul = 0.5 + 0.5 * np.cos(self.n_warmup_steps / self.T_max * np.pi)
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
        self.n_warmup_steps += 1
        self.global_steps += 1
        # for param, avg_grad, avg_sq_grad in zip(self.model_params, self.avg_grads, self.avg_sq_grads):
        for idx, param in enumerate(self.model_params):
            if self.global_steps == 1:
                self.avg_grads[idx] = torch.zeros_like(param.data)
                self.avg_sq_grads[idx] = torch.zeros_like(param.data)
            avg_grad = self.avg_grads[idx]
            avg_sq_grad = self.avg_sq_grads[idx]
            grad = param.grad
            # apply L2 regularization
            if self.L2_regularization:
                grad += self.weight_decay * param.data
            # update moving averages
            avg_grad.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            avg_sq_grad.mul_(self.beta2).add_(grad**2, alpha=1 - self.beta2)

            # compute bias-corrected moments
            bias_correction1 = 1 - self.beta1 ** self.global_steps
            bias_correction2 = 1 - self.beta2 ** self.global_steps
            avg_grad_corr = avg_grad / bias_correction1
            avg_sq_grad_corr = avg_sq_grad / bias_correction2
            
            # apply AdamW update
            std = (avg_sq_grad_corr.sqrt() + self.eps)
            param.sub_(self.lr * (avg_grad_corr / std + self.weight_decay * param.data))

def solve_sylvester(A, B, C, X=None):
    ''' From the answer here: 
        https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch
    '''
    if A.dtype is torch.bfloat16:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
    B = -B
    m = B.shape[-1];
    n = A.shape[-1];
    try:
        R, U = torch.linalg.eig(A)
    except:
        print(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
    return X.real if all(torch.isreal(x.flatten()[0]) 
                for x in [A, B, C]) else X
class AdamWLoraPro:
    def __init__(
        self,
        model_params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        T_max: int = 1000,
        L2_regularization: bool = False,
        scaling_factor: float = 2.0,
    ):
        self.lr = lr
        self.original_lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.model_params = [param for param in model_params if param.requires_grad]
        self.n_warmup_steps = 0
        self.global_steps = 0
        self.L2_regularization = L2_regularization
        self.T_max = T_max
        self.scaling_factor = scaling_factor
        self.avg_grads = {}
        self.avg_sq_grads = {}
    
    def set_warmup(self):
        if self.n_warmup_steps == self.T_max:
            self.T_max *= 2
            self.n_warmup_steps = 0
        schedule_mul = 0.5 + 0.5 * np.cos(self.n_warmup_steps / self.T_max * np.pi)
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
        self.n_warmup_steps += 1
        self.global_steps += 1
        for idx, (lora_A, lora_B) in enumerate(zip(self.model_params[::2], self.model_params[1::2])):
            A = lora_A.data
            B = lora_B.data
            grad_A_origin = lora_A.grad
            grad_B_origin = lora_B.grad
            AA_T = A @ A.T 
            B_TB = B.T @ B 
            if self.global_steps == 1:
                # for lora grad
                AA_T_inv = torch.linalg.pinv(AA_T + self.eps * torch.eye(A.shape[0], device=grad_A_origin.device))
                AA_T_inv = AA_T_inv.to(grad_A_origin.dtype)
                grad_A = grad_A_origin
                grad_B = (1 / self.scaling_factor ** 2) * grad_B_origin @ AA_T_inv
            else:
                AA_T_inv = torch.linalg.pinv(AA_T + self.eps * torch.eye(A.shape[0], device=grad_A_origin.device)) 
                B_TB_inv = torch.linalg.pinv(B_TB + self.eps * torch.eye(A.shape[0], device=grad_A_origin.device)) 
                AA_T_inv = AA_T_inv.to(A.dtype)
                B_TB_inv = B_TB_inv.to(A.dtype)
                grad_A = (1 / self.scaling_factor ** 2) * B_TB_inv @ grad_A_origin
                grad_B = (1 / self.scaling_factor ** 2) * ((torch.eye(B.shape[0], device=grad_A_origin.device, dtype=A.dtype) - B @ B_TB_inv @ B.T) @ grad_B_origin @ AA_T_inv)   
            # equiv_grad = self.scaling_factor * B @ grad_A + self.scaling_factor * grad_B @ A
            if self.global_steps == 1:
                self.avg_grads[idx*2] = torch.zeros_like(grad_A_origin)
                self.avg_grads[idx*2+1] = torch.zeros_like(grad_B_origin)
                self.avg_sq_grads[idx*2] = torch.zeros_like(grad_A_origin)
                self.avg_sq_grads[idx*2+1] = torch.zeros_like(grad_B_origin)
            avg_grad_A = self.avg_grads[idx*2]
            avg_sq_grad_A = self.avg_sq_grads[idx*2]
            avg_grad_B = self.avg_grads[idx*2+1]
            avg_sq_grad_B = self.avg_sq_grads[idx*2+1]
            
            # avg_grad.mul_(self.beta1).add_(equiv_grad, alpha=1 - self.beta1)
            # avg_sq_grad.mul_(self.beta2).add_(equiv_grad**2, alpha=1 - self.beta2)
            avg_grad_A.mul_(self.beta1).add_(grad_A, alpha=1 - self.beta1)
            avg_sq_grad_A.mul_(self.beta2).add_(grad_A**2, alpha=1 - self.beta2)
            avg_grad_B.mul_(self.beta1).add_(grad_B, alpha=1 - self.beta1)
            avg_sq_grad_B.mul_(self.beta2).add_(grad_B**2, alpha=1 - self.beta2)
            # compute bias-corrected moments
            bias_correction1 = 1 - self.beta1 ** self.global_steps
            bias_correction2 = 1 - self.beta2 ** self.global_steps
            avg_grad_corr_A = avg_grad_A / bias_correction1
            avg_sq_grad_corr_A = avg_sq_grad_A / bias_correction2
            avg_grad_corr_B = avg_grad_B / bias_correction1
            avg_sq_grad_corr_B = avg_sq_grad_B / bias_correction2
            
            # apply AdamW update
            std_A = (avg_sq_grad_corr_A.sqrt() + self.eps)
            std_B = (avg_sq_grad_corr_B.sqrt() + self.eps)
            g_A = (avg_grad_corr_A / std_A)
            g_B = (avg_grad_corr_B / std_B)
            g_A = g_A.to(grad_A.dtype)
            g_B = g_B.to(grad_B.dtype)
            equiv_grad = self.scaling_factor * B @ g_A + self.scaling_factor * g_B @ A

            grad_A_orin_ = self.scaling_factor * B.T @ equiv_grad 
            grad_B_orin_ = self.scaling_factor * equiv_grad @ A.T 

            grad_A_origin.data = grad_A_orin_
            grad_B_origin.data = grad_B_orin_

            if self.global_steps == 1:
                grad_A = grad_A_orin_ 
                grad_B = (1 / self.scaling_factor ** 2) * grad_B_orin_ @ AA_T_inv 
            else:
                X = solve_sylvester(B.T @ B, A @ A.T, -(1 / self.scaling_factor ** 2) * B_TB_inv @ grad_A_orin_ @ A.T)
                # X = torch.tensor(X, device=grad_A_origin.device, dtype=B.dtype)
                X = X.to(device=grad_A_origin.device, dtype=B.dtype)

                grad_A = (1 / self.scaling_factor ** 2) * B_TB_inv @ grad_A_orin_ + X @ A
                grad_B = (1 / self.scaling_factor ** 2) * ((torch.eye(B.shape[0], device=grad_A_origin.device, dtype=A.dtype) - B @ B_TB_inv @ B.T) @ grad_B_orin_ @ AA_T_inv) - B @ X  

            lora_A.sub_(self.lr * (grad_A + self.weight_decay * lora_A.data))
            lora_B.sub_(self.lr * (grad_B + self.weight_decay * lora_B.data))