import argparse
from dataset import OmniMNIST, OmniFashionMNIST
from sphere_cnn import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F


class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        self.conv1 = SphereConv2D(1, 32, stride=1, mode='bilinear')
        self.pool1 = SphereMaxPool2D(stride=2, mode='bilinear')
        self.conv2 = SphereConv2D(32, 64, stride=1, mode='bilinear')
        self.pool2 = SphereMaxPool2D(stride=2, mode='bilinear')
        
        self.fc = nn.Linear(14400, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 14400)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64*13*13, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64*13*13)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

       
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if data.dim() == 3:
            data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if data.dim() == 3:
                data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SphereNet Example')
    parser.add_argument('--data', type=str, default='FashionMNIST',
                        help='dataset for training, options={"FashionMNIST", "MNIST"} (default: FashionMNIST)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before saving model weights')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.data == 'FashionMNIST':
        train_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=True)
        test_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=False)
    elif args.data == 'MNIST':
        train_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=True)
        test_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    
    # Train
    sphere_model = SphereNet().to(device)
    sphere_optimizer = torch.optim.SGD(sphere_model.parameters(), lr=args.lr, momentum=args.momentum)
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
    	# conventional CNN
        print('{} Conventional CNN {}'.format('='*10, '='*10))
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        if epoch % args.save_interval == 0:
        	torch.save(model.state_dict(), 'model.pkl')
        # SphereCNN
        # sphere_model.load_state_dict(torch.load('sphere_model.pkl'))
        print('{} Sphere CNN {}'.format('='*10, '='*10))
        train(args, sphere_model, device, train_loader, sphere_optimizer, epoch)
        test(args, sphere_model, device, test_loader)
        if epoch % args.save_interval == 0:
        	torch.save(sphere_model.state_dict(), 'sphere_model.pkl')



if __name__ == '__main__':
    main()