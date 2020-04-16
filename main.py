import torch
import torch.optim as optim

from model import STN
from train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 20 + 1):
    train(epoch)
    test()