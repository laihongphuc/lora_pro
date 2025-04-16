import torch 
import torch.nn as nn
import lora_pro.adamw as adamw

def test_adamw():
    # create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
        
    # create a simple model
    model = SimpleModel()

    # create a simple optimizer
    optimizer = adamw.AdamW(model.parameters())

    # create a simple loss function
    criterion = nn.MSELoss()    

    # create a simple input
    x = torch.randn(10)

    # forward pass
    y = model(x)

    # compute loss
    loss = criterion(y, torch.randn(1))

    # backward pass
    loss.backward()

    # step optimizer
    optimizer.step()

    assert True
    
