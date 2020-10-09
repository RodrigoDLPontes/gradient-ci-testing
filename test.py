import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

def main():
    # Model
    model = models.alexnet(pretrained=True).cuda()
    in_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_ftrs, 10).cuda()
    # Optimization
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = F.cross_entropy
    # Dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda i: torch.stack([i] * 3, dim=1)[0]),
        transforms.Normalize(mean, std)
    ])
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True)
    # Execution
    train(model, optimizer, criterion, train_loader, 1)
    loss = test(model, optimizer, criterion, test_loader, len(test_data))
    print(f'Final test loss: {loss}')
    with open('/artifacts/results.txt', 'w') as f:
        f.write(f'Final test loss: {loss}')

def train(model, optimizer, criterion, train_loader, epochs):
    model.train()
    print('Starting training...')
    for i in range(epochs):
        print(f'Running epoch {i + 1}')
        for batch_idx, batch in enumerate(train_loader):
            images, targets = map(lambda b: b.cuda(), batch)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print(f'\tMini-batch training loss: {loss.item()}')

def test(model, optimizer, criterion, test_loader, n_samples):
    loss = 0
    model.eval()
    print('Testing...')
    for batch_idx, batch in enumerate(test_loader):
        images, targets = map(lambda b: b.cuda(), batch)
        output = model(images)
        loss += criterion(output, targets, size_average=False).data
    loss /= n_samples
    return loss

main()