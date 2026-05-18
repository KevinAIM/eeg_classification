import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import EEGClassifier
from pathlib import Path

# Load preprocessed data
data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'
X = np.load(data_dir / 'X.npy')
y = np.load(data_dir / 'y.npy')

subjects = np.load(data_dir / 'subjects.npy')

# Subject-based train/test split
train_mask = subjects <= 87
test_mask = subjects > 87

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Normalize data with training only, not test data to prevent data leakage
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add channel dimension and convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")


# Create DataLoaders for training and testing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Batch size of 32 because you don't feed the whole dataset at once.
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = EEGClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        predicted = (output > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
torch.save(model.state_dict(), results_dir / 'model.pth')
print("Model saved.")

print(X.shape, y.shape)