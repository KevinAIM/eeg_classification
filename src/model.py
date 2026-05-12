import torch
import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # define your layers here

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(64, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4096, 1)
        
    def forward(self, x):
        # define how data flows through layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
    
if __name__ == "__main__":
    model = EEGClassifier()
    dummy = torch.zeros(1, 1, 64, 513)
    out = model(dummy)
    print(out.shape)  # should print torch.Size([1, 1])
    print(out)        # should print a value between 0 and 1
