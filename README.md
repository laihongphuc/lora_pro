# lora_pro
A simple implementation of optimizer proposed in [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/abs/2407.18242)
## Installation
```bash
# Clone the repository
git clone https://github.com/laihongphuc/lora_pro.git
cd lora_pro

# Install the package in development mode
pip install -e .
```

## Dependencies
- torch>=2.0.0
- peft>=0.4.0
- transformers>=4.30.0

## How to Use

### Basic Usage
```python
from lora_pro import AdamWLoRAPro
from peft import LoraConfig, get_peft_model
import torch.nn as nn

# Create your model
model = YourModel()

# Convert to LoRA model
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, lora_config)

# Create an optimizer for head layer
classifier_params = list(peft_model.classifier.parameters())
other_params = [p for n, p in peft_model.named_parameters() if "classifier" not in n]
optimizer_cls = torch.optim.AdamW(
    classifier_params,
    lr=1e-3,
    weight_decay=1e-2
)
# Create an optimizer for LoRA layer
optimizer_lora = AdamWLoRAPro(
    other_params,
    lr=1e-3,
    weight_decay=0.01,
    T_max=1000,  # warmup steps
    scaling_factor=2.0, 
)
optimizer = [optimizer_lora, optimizer_cls]

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer[0].step()
        optimizer[1].step()
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
```

## Example: Fine-tuning on CIFAR100

| Configuration | Test Accuracy (%) | Peak Memory (GB) | Trainable Parameters (%) |
|--------------|-------------------|-------------------|--------------------------|
| LoRA_r8_s16 | 91.30 | 13 | 0.8593 |
| LoRA_r16_s16 | 91.48 | 13 | 0.8593 |
| LoRA-Pro_r16_s16 | 92.00 | 24 | 0.8593 |

The table above shows the performance comparison between different fine-tuning approaches on CIFAR100 using `VIT-base-patch16-224` as the base model. LoRA-Pro demonstrates competitive performance.

## Features
- Implements the optimizer proposed in the LoRA-Pro paper
- Supports warmup scheduling
- Compatible with PEFT library for LoRA implementation
- Easy integration with existing PyTorch training pipelines
# Technical 
### Low-rank gradient update
- For *fully-finetuning*, we assume $dL$ and $dW$ are changes in the loss and changes in the weights
```math
dL = \left<\frac{\delta L}{\delta W}, dW\right>_F
```
in gradient-descent, people tend to choose $dW = -\frac{\delta L}{\delta W}=g_{full}$ so the change in loss is maximize.
- For *lora-tuning*, we update matrix $A$ and $B$, the new weight is $W = W_0 + sBA, B\in\mathbb{R}^{m\times r}, A\in \mathbb{R}^{r \times n}$. Using chain-rule
```math
\begin{align*}
dL &= \left<\frac{\delta L}{\delta W}, dW\right>_F \\
&= \left<\frac{\delta L}{\delta W}, \frac{\delta W}{\delta A}dA + \frac{\delta W}{\delta B}d_B\right>_F \\
&= \left<\frac{\delta L}{\delta A}, dA\right>_F + \left<\frac{\delta L}{\delta B}, dB\right>_F \\
\end{align*}
```
in gradient-descent, we choose $dA = -\frac{\delta L}{\delta A}=-\frac{\delta L}{\delta W}\frac{\delta W}{\delta A}=-sB^T g_{full}=g^A_{lora}$ and $dB = -\frac{\delta L}{\delta B}=g^B_{lora}=-sg_{full}A^T$ 
So the changes in $A$ and $B$ is equivalent to changes in $W$ as below
```math
dW = sB(dA) + s(dB)A = -s^2\left(BB^T g_{full} + g_{full} A^TA\right)
```
=> **equivalent to low-rank gradient update**
- **The optimization problem:** find $dA$ and $dB$ so that 
```math
\text{ min }_{g^A, g^B}\| sBg^A + sg^BA - g_{full}\|_F^2 
```
### Background on Linear Algebra
#### Least Square
- The solution $H$ to $\|HA - X\|_F^2$ is $H = XA^T(AA^T)^{-1}$
- The solution $H$ to $\|BH - X\|_F^2$ is $H=(B^TB)^{-1}B^TX$ 
#### Projection Matrix
- Given matrix $A$, the projection matrix to the column space of $A$ is $P$. Assume vector $b$ has projection in $A$ is $p$, we have $p=Ax$ and $A^T(b-p)=0$ => $A^Tb = A^TAx \rightarrow x = (A^TA)^{-1}A^Tb \rightarrow p=(A^TA)^{-1}A^Tb$ so the *projection matrix is* $(A^TA)^{-1}A^TB$
- The projection matrix to the null-space of $A$ is $I-P$ because $(I-P)b = b - p \in \text{null}(A)$ 
- The solution is matrix equation $(I - P)X = (I-P)X$, because $(I-P)(X-M) = 0$ so we can ensure that $X-M \in col(A)$ and $X = M + CA$ 
## The closed form solution
 - We denote $L=\|sBg^A + sg^BA - g\|_F^2$, to solve the optimization problem, we need to satisfy the following condition
```math
\begin{align}
\frac{\delta L}{\delta g^A} &= 2sB^T(sBg^A + sg^BA - g) = 0 \text{ (1)}\\
\frac{\delta L}{\delta g^B} &= 2(sBg^A + sg^BA - g)sA^T = 0 \text{ (2)}\\
\end{align}
```
- From equation (2) we can derive $g^B = \frac{1}{s}gA^T (AA^T)^{-1} - Bg^A A^T(AA^T)^{-1}$ (3)
- By compute $g^B$ from equation (2) and substitue this into equation (1) we obtain the following linear equation
```math
g^A(I - A^T(AA^T)^{-1}A) = \frac{1}{s}(B^TB)^{-1}B^Tg[I - A^T(AA^T)^{-1}A]
```
From the background section, we can have the solution of this equation is $g^A = \frac{1}{s} (B^TB)^{-1}B^Tg + XA$, substitue to (3) we have $g^B = \frac{1}{s}gA^T(AA^T)^{-1} - \frac{1}{s} (B^TB)^{-1}B^TgA^T(AA^T)^{-1} - BX$ 
- Because the solution is depend on $g$ (the full-gradient we don't have) but we could use $B^Tg = g^A_{lora}$ and $gA^T = g^B_{lora}$ so that is OKE.
*Note*: the matrix $X$ could be abitrary matrix => but we should choose $X$ so that the diverse between $g^A, g^B$ and $g^A_{lora}, g^B_{lora}$ is minimize. (*The author's assumption:*)
## The stronger condition
- We consider the optimization problem
```math
\text{min}_X L_2 =  \|g^A - g^A_{lora}\|_F^2 + \|g^B - g^B_{lora}\|_F^2 + 
```
where $g^A$ and $g^B$ are the optimla solutions as stated as follow. The optimal $X$ can be determined by derive $\frac{\delta L_2}{\delta X}=0$, as
```math
B^TB X + XAA^T = -\frac{1}{s^2} (B^B)^{-1} g^A_{lora}A^T
```
which is a Sylvester equation.
## Citation
If you use this code in your research, please cite the original LoRA-Pro paper:

```bibtex
@article{wang2024lora,
  title={LoRA-Pro: Are Low-Rank Adapters Properly Optimized?},
  author={Wang, Zhengbo and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  journal={arXiv preprint arXiv:2407.18242},
  year={2024}
}
```
