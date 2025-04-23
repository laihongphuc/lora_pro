# lora_pro
A simple implementation of optimizer proposed in LoRA-Pro paper for Low-rank finetuning

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

## Citation
If you use this code in your research, please cite the original LoRA-Pro paper:

```bibtex
@article{lora_pro,
  title={LoRA-Pro: Optimizer for Low-rank Adaptation},
  author={Original Authors},
  journal={Conference/Journal Name},
  year={2024}
}
```
