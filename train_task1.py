import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import HuaweiDataset
import tqdm
import numpy as np
from torch.nn import functional as F


class BaselinePredictor(nn.Module):
    '''
    We propose a simple ANN with linear layers, BN and PReLU activations.
    Description of the inputs to be found in HuaweiDataset class in dataset.py
    '''
    def __init__(self):
        super(BaselinePredictor, self).__init__()
        self.predictor = nn.Sequential(nn.Linear(22, 64),
                                       nn.BatchNorm1d(64),
                                       nn.PReLU(),
                                       nn.Linear(64, 64),
                                       nn.BatchNorm1d(64),
                                       nn.PReLU(),
                                       nn.Linear(64, 64),
                                       nn.BatchNorm1d(64),
                                       nn.PReLU(),
                                       nn.Linear(64, 32),
                                       nn.BatchNorm1d(32),
                                       nn.PReLU(),
                                       nn.Linear(32, 1))

    def forward(self, x):
        return self.predictor(x)


def export_result(net, test_loader):
    # Collecting the predictions in one array and saving it to the hard drive.
    if isinstance(net, nn.Module):
        net.eval()
    with torch.no_grad():
        distances_pred = []
        for i, data in enumerate(test_loader):
            signal_powers, _ = data
            signal_powers = signal_powers.cuda()
            signal_powers = signal_powers.view(-1, 22)
            distances_pred.append(net(signal_powers).squeeze())
        distances_pred = torch.cat(distances_pred, dim=0).cpu().numpy()
        test_loader.dataset.export_results(distances_pred)


def test(net, test_loader):
    # Testing function for calculating valdidation loss
    if isinstance(net, nn.Module):
        net.eval()
    with torch.no_grad():
        distances_pred = []
        distances_gt = []
        for i, data in enumerate(test_loader):
            signal_powers, distance = data
            signal_powers = signal_powers.cuda()
            distance = distance.cuda()
            signal_powers = signal_powers.view(-1, 22)
            distances_pred.append(net(signal_powers).squeeze())
            distances_gt.append(distance)
        distances_gt = torch.cat(distances_gt, dim=0).cpu().numpy()
        distances_pred = torch.cat(distances_pred, dim=0).cpu().numpy()
        print(np.mean(np.abs(distances_gt - distances_pred)))


if __name__ == '__main__':
    # Initializing dataset splits
    dataset = HuaweiDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=8)
    val_dataset = HuaweiDataset(split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8)
    test_dataset = HuaweiDataset(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=8)

    # NN, optimizer, loss function and exponential learning rate scheduler
    net = BaselinePredictor().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    loss_fn = lambda x, y: F.l1_loss(x, y) + 0.5 * F.mse_loss(x, y) # L1 loss focuses on small distances, MSE puts more weight into big distances
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    # Training loop
    for epoch in range(10):
        net.train()
        running_loss = []
        tbar = tqdm.tqdm(dataloader, total=len(dataloader))
        for i, data in enumerate(tbar):
            optimizer.zero_grad()
            signal_powers, distance = data
            signal_powers = signal_powers.cuda()
            distance = distance.cuda().squeeze()
            signal_powers = signal_powers.view(-1, 22)
            distance_pred = net(signal_powers)
            loss = loss_fn(distance_pred.squeeze(), distance)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            tbar.set_postfix_str(str(sum(running_loss) / len(running_loss))) # Logging loss values per epoch
        scheduler.step()
        test(net, test_loader=val_dataloader)
    torch.save(net.state_dict(), 'model.pth')
    export_result(net, test_loader=test_dataloader)

