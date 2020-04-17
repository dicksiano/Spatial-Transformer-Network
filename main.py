import torch
import torch.optim as optim

from model import NeuralNet
from train import train, test, test_dataset
from visualizer import show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizer, device)
        test(model, device)

    show(model, test_dataset, device)