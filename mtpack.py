import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from torch.optim import AdamW, Muon
import matplotlib.pyplot as plt

def download_kaggle_mnist(target_dir: Path):
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('oddrationale/mnist-in-csv', path=str(target_dir), unzip=True)

    train_csv = target_dir / 'mnist_train.csv'
    test_csv = target_dir / 'mnist_test.csv'
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError('Не найдены mnist_train.csv / mnist_test.csv после загрузки из Kaggle')
    return train_csv, test_csv

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, keep_digits=None, only_digits=None):
        df = pd.read_csv(csv_path)
        labels = df.iloc[:, 0].to_numpy()
        images = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
        mask = np.ones(len(labels), dtype=bool)
        if keep_digits is not None:
            mask &= np.isin(labels, keep_digits)
        if only_digits is not None:
            mask &= np.isin(labels, only_digits)
        self.labels = labels[mask]
        self.images = images[mask]
        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tf(Image.fromarray(self.images[idx], mode='L'))
        y = int(self.labels[idx])
        return x, y


class Derf(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.4)
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.s = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return self.alpha * torch.erf(self.beta * x + self.s)


class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=stride, padding=2, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.derf = Derf(out_ch)

    def forward(self, x):
        return self.derf(self.pw(self.dw(x)))


class SimpleDerfNet(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleDerfNet, self).__init__()
        self.layer1 = DSBlock(1, 4, stride=1)
        self.layer2 = DSBlock(4, 8, stride=2)
        self.layer3 = DSBlock(8, 16, stride=2)
        self.layer4 = DSBlock(16, 1, stride=1)
        self.fc = nn.Linear(49, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x).view(x.size(0), -1)
        return self.fc(x)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    correct = 0

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_func(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        correct += ((logits.argmax(1) == y) + 0.).mean().item()

    return total_loss / len(loader), correct / len(loader)


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_func(logits, y)

        total_loss += loss.item()
        correct += ((logits.argmax(1) == y) + 0.).mean().item()

    return total_loss / len(loader), correct / len(loader)


class HybridOptimizer:
    def __init__(self, model, lr=0.001):
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.ndim != 2]

        self.muon = torch.optim.Muon(muon_params, lr=lr, momentum=0.95)
        self.adamw = torch.optim.AdamW(adamw_params, lr=lr / 10, weight_decay=0.01)

    def zero_grad(self):
        self.muon.zero_grad()
        self.adamw.zero_grad()

    def step(self):
        self.muon.step()
        self.adamw.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str, default='/Volumes/HP/proj/mnist_data/data')
    parser.add_argument('--out-dir', type=str, default='/Volumes/HP/proj/mnist_data/results')
    args = parser.parse_args()

    device = pick_device()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv, test_csv = download_kaggle_mnist(data_dir)

    train_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_ds = CSVImageDataset(train_csv, keep_digits=train_digits)
    val_cls_ds = CSVImageDataset(test_csv, keep_digits=train_digits)

    num_workers = 0 if device.type == 'mps' else 2
    pin_memory = device.type == 'cuda'

    train_loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_cls_loader = DataLoader(val_cls_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = SimpleDerfNet(num_classes=10).to(device)
    print(f'Model has: {sum(p.numel() for p in model.parameters())} parameters')
    optimizer = HybridOptimizer(model)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = eval_loader(model, val_cls_loader, device)
        print(f'Epoch: {epoch}, loss_tr: {tr_loss:0.3f}, tr_acc: {tr_acc:0.3f}, va_loss {va_loss:0.3f}, va_acc {va_acc:0.3f}')

if __name__ == '__main__':
    main()