import torch
import torch.optim as optim

from model import NeuralNet
from train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(1, 15 + 1):
        train(epoch, model, optimizer, device)
        test(model, device)