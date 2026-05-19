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
    
class EEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16):
        super().__init__()
        self.elu = nn.ELU()
        self.linear = nn.Linear(256, 1)

        #regular conv2d (temporal)
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)

        #depthwise convolution (spatial)
        self.depthwise = nn.Conv2d(F1, F1*D, kernel_size=(64, 1), groups=F1, bias=False)
        nn.Conv2d(F1, F1*D, kernel_size=(64, 1), groups=F1, bias=False)

        #separable convolution
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.5)

        self.separable = nn.Conv2d(F1*D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.linear(x)
        return torch.sigmoid(x)
    
if __name__ == "__main__":
    # model = EEGClassifier()
    # dummy = torch.zeros(1, 1, 64, 513)
    # out = model(dummy)
    # print(out.shape)  # should print torch.Size([1, 1])
    # print(out)        # should print a value between 0 and 1

    model = EEGNet()
    dummy = torch.zeros(1, 1, 64, 513)
    out = model(dummy)
    print(out.shape)
    print(out)
