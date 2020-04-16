import torch
from torchvision import datasets, transforms

import torch.nn.functional as F

# Training dataset
train_dataset = torch.utils.data.DataLoader(
                                            datasets.MNIST(
                                                            root='.', 
                                                            train=True, 
                                                            download=True,
                                                            transform=transforms.Compose( [ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ] )
                                                        ),
                                            batch_size=64, 
                                            shuffle=True, 
                                            num_workers=4
                                        )
# Test dataset
test_dataset = torch.utils.data.DataLoader(
                                            datasets.MNIST(
                                                            root='.', 
                                                            train=False, 
                                                            transform=transforms.Compose( [ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ] )
                                                        ), 
                                            batch_size=64, 
                                            shuffle=True, 
                                            num_workers=4
                                        )

def train(epoch, model, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataset):
        data   = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                        epoch, 
                                                                        batch_idx * len(data), 
                                                                        len(train_dataset.dataset),
                                                                        100. * batch_idx / len(train_dataset), 
                                                                        loss.item()
                                                                    ))

def test(model, device):
    with torch.no_grad():
        model.eval()
        loss = 0
        correct = 0

        for data, target in test_dataset:
            data   = data.to(device)
            target = target.to(device)
            output = model(data)

            loss += F.nll_loss(output, target, size_average=False).item() # negative log likelihood loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest: Loss avg: {:.4f}, Acc: ({:.0f}%)\n'.format(
                                                                    loss / len(test_dataset.dataset), 
                                                                    100. * correct / len(test_dataset.dataset)
                                                                ))