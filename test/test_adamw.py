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

def test_warmup():
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
    optimizer = adamw.AdamW(model.parameters(), lr=1., T_max=1)
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
    for i in range(10):
        # step optimizer
        optimizer.step()
        if i == 0:
            assert optimizer.lr == 1.
            # n = 0, T = 1
        elif i == 1:
            assert optimizer.lr == 1.
            # n = 1, T = 1 => n = 0, T = 2
            # print(optimizer.T_max)
        elif i == 7:
            assert optimizer.lr == 1.
            # n = 2, T = 2 => n = 0, T = 4
            # print(optimizer.T_max)
    assert True

def test_lora_pro():
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
    # create a lora model
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["fc1"]
    )
    model = get_peft_model(model, lora_config)
    # create a simple optimizer
    optimizer = adamw.AdamWLoraPro(model.parameters(), lr=1, T_max=1, weight_decay=0.01)
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
    for i in range(10):
        # step optimizer
        optimizer.step()
        if i == 0:
            assert optimizer.lr == 1.
            # n = 0, T = 1
        elif i == 1:
            assert optimizer.lr == 1.
            # n = 1, T = 1 => n = 0, T = 2
            # print(optimizer.T_max)
        elif i == 7:
            assert optimizer.lr == 1.
            # n = 2, T = 2 => n = 0, T = 4
            # print(optimizer.T_max)
    assert True


if __name__ == "__main__":
    test_extract_lora_params()