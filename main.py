import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, vgg16, densenet201, efficientnet_b0, ResNet50_Weights, VGG16_Weights, DenseNet201_Weights, EfficientNet_B0_Weights
from dataset import load_cifar10, load_cifar100

# Define the available architectures and their associated weights
architectures = {
    'resnet50': (resnet50, ResNet50_Weights),
    'vgg16': (vgg16, VGG16_Weights),
    'efficientnet_b0': (efficientnet_b0, EfficientNet_B0_Weights),
    'densenet201': (densenet201, DenseNet201_Weights)
}

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train a neural network on CIFAR-10 or CIFAR-100')
parser.add_argument('--dataset', type=str, required=True,
                    choices=['cifar10', 'cifar100'], help='Which dataset to use')
parser.add_argument('--arch', type=str, required=True,
                    choices=list(architectures.keys()), help='Which architecture to use')
args = parser.parse_args()

# Set up the device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
if args.dataset == 'cifar10':
    train_loader, test_loader = load_cifar10(batch_size=128)
else:
    train_loader, test_loader = load_cifar100(batch_size=128)

# Define the model and move it to the device
model_fn, weights = architectures[args.arch]
model = model_fn(weights=weights.DEFAULT).to(device)

if args.dataset == 'cifar100':
    if args.arch == 'resnet50':
        model.fc = nn.Linear(2048, 100)
    elif args.arch == 'vgg16':
        model.classifier[6] = nn.Linear(4096, 100)
    elif args.arch == 'efficientnet_b0':
        model._fc = nn.Linear(1280, 100)
    elif args.arch == 'densenet201':
        model.classifier = nn.Linear(1920, 100)
elif args.dataset == 'cifar10':
    if args.arch == 'resnet50':
        model.fc = nn.Linear(2048, 10)
    elif args.arch == 'vgg16':
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.arch == 'efficientnet_b0':
        model._fc = nn.Linear(1280, 10)
    elif args.arch == 'densenet201':
        model.classifier = nn.Linear(1920, 10)


# Define the loss function, optimizer and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Define early stopping criteria
best_acc = 0.0
patience = 10
counter = 0

# Train the model
for epoch in range(200):
    # Train for one epoch
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluate on the test set and update the learning rate scheduler
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print('Epoch %d: Test accuracy = %.2f%%' % (epoch, acc))
    lr_scheduler.step(acc)

    # Check if the current model is the best so far
    if acc > best_acc:
        print('Saving checkpoint...')
        torch.save(model.state_dict(), 'weights/best_%s.pth' % args.arch)
        best_acc = acc
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping...')
            break

print('Training finished.')
