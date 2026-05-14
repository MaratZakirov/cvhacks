import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import contextlib


def download_kaggle_mnist(target_dir: Path):
    if not target_dir.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi
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
    def __init__(self, csv_path, device, keep_digits=None, only_digits=None):
        df = pd.read_csv(csv_path)
        labels = df.iloc[:, 0].to_numpy()
        images = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
        mask = np.ones(len(labels), dtype=bool)
        if keep_digits is not None:
            mask &= np.isin(labels, keep_digits)
        if only_digits is not None:
            mask &= np.isin(labels, only_digits)

        # Внутри CSVImageDataset.__init__
        self.images = torch.from_numpy(images[mask]).float().to(device) / 255.0  # Сразу в тензор [N, 28, 28]
        self.images = self.images.unsqueeze(1)  # Добавляем канал -> [N, 1, 28, 28]
        self.labels = torch.tensor(labels[mask], dtype=torch.long).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class Derf(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Изменяем размерность параметров под формат [B, N, C]
        # Последняя размерность — каналы (dim)
        self.alpha = nn.Parameter(torch.ones(1, 1, dim) * 0.4)
        self.beta = nn.Parameter(torch.ones(1, 1, dim))
        self.s = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return self.alpha * torch.erf(self.beta * x + self.s)

class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise свертка остается без изменений
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=stride, padding=2, groups=in_ch)

        # Вместо Conv2d 1x1 используем Linear. Веса будут иметь честную 2D-форму.
        self.pw = nn.Linear(in_ch, out_ch)
        self.derf = Derf(out_ch)

    def forward(self, x):
        # 1. Прогоняем через пространственную depthwise свертку
        x = self.dw(x)  # [B, in_ch, H, W]

        # 2. Перекладываем оси для линейного слоя: [B, in_ch, H, W] -> [B, H, W, in_ch]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, in_ch]
        x = x.view(B, H * W, C)  # Схлопываем пространство: [B, H*W, in_ch]

        # 3. Применяем Pointwise (Linear) и нелинейность Derf
        x = self.pw(x)  # [B, H*W, out_ch]
        x = self.derf(x)

        # 4. Возвращаем исходную структуру [B, out_ch, H, W] для следующего слоя
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

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


def train_one_epoch(model, loader, optim, device, use_bf16=False):
    model.train()
    total_loss = 0.0
    correct = 0

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            loss = loss_func(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        total_loss += loss.item()
        correct += ((logits.argmax(1) == y) + 0.).mean().item()

    return total_loss / len(loader), correct / len(loader)


@torch.no_grad()
def eval_loader(model, loader, device, use_bf16=False):
    model.eval()
    total_loss = 0.0
    correct = 0

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
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

class FastLoader:
    def __init__(self, images, labels, batch_size, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = images.shape[0]

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n, device=self.images.device)
            self.images = self.images[indices]
            self.labels = self.labels[indices]

        for i in range(0, self.n, self.batch_size):
            yield self.images[i:i + self.batch_size], self.labels[i:i + self.batch_size]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def collect_energy(model, loader, temperature, device, use_bf16):
    model.eval()
    energy_scores = []
    energy_logits = []

    # Определяем контекст автокаста
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float16
    ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_bf16 else contextlib.nullcontext()

    with torch.no_grad():
        for x, _ in loader:
            with ctx:
                logits = model(x)  # [B, 10]
                if use_bf16:
                    logits = logits.float()

            # Формула свободной энергии Гельмгольца (Weitang Liu et al.)
            # E(x) = -T * logsumexp(logits / T)
            scaled_logits = logits / temperature
            energy = -temperature * torch.logsumexp(scaled_logits, dim=1)
            energy_scores.append(energy.cpu())
            energy_logits.append(logits.cpu())

    energy_scores = torch.cat(energy_scores)
    energy_logits = torch.cat(energy_logits, dim=0)

    return energy_scores.numpy(), energy_logits.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str, default='/Volumes/HP/proj/mnist_data/data')
    parser.add_argument('--out-dir', type=str, default='/Volumes/HP/proj/mnist_data/results')
    parser.add_argument('--bf16', action='store_true')
    args = parser.parse_args()

    device = pick_device()

    use_bf16 = args.bf16 and (device.type == 'cuda' or device.type == 'mps')

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv, test_csv = download_kaggle_mnist(data_dir)

    train_digits = [0, 2, 3, 4, 5, 6, 7, 8, 9]

    train_ds   = CSVImageDataset(train_csv, device, keep_digits=train_digits)
    val_cls_ds = CSVImageDataset(test_csv, device, keep_digits=train_digits)

    train_loader   = FastLoader(train_ds.images, train_ds.labels, batch_size=64, shuffle=True)
    val_cls_loader = FastLoader(val_cls_ds.images, val_cls_ds.labels, batch_size=64, shuffle=False)

    model = SimpleDerfNet(num_classes=10).to(device)
    print(f'Model has: {sum(p.numel() for p in model.parameters())} parameters')
    optimizer = HybridOptimizer(model)

    for epoch in range(args.epochs):
        start_epoch = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, use_bf16)
        va_loss, va_acc = eval_loader(model, val_cls_loader, device, use_bf16)
        end_epoch = time.time()
        print(f'Epoch: {epoch}, loss_tr: {tr_loss:0.3f}, tr_acc: {tr_acc:0.3f}, va_loss {va_loss:0.3f}, va_acc {va_acc:0.3f} took {end_epoch - start_epoch:.3f} seconds')

    # === КОД ДЛЯ OOD ДЕТЕКЦИИ ПОСЛЕ ЦИКЛА ОБУЧЕНИЯ ===
    print("\nStarting Energy-based OOD detection analysis...")

    # 1. Загружаем датасеты, содержащие ТОЛЬКО 7 и ТОЛЬКО 1
    # Предполагается, что CSVImageDataset принимает оригинальные метки Kaggle (0-9)
    ds_id_7 = CSVImageDataset(test_csv, device, keep_digits=[7])
    ds_ood_1 = CSVImageDataset(test_csv, device, keep_digits=[1])

    loader_id_7 = FastLoader(ds_id_7.images, ds_id_7.labels, batch_size=args.batch_size, shuffle=False)
    loader_ood_1 = FastLoader(ds_ood_1.images, ds_ood_1.labels, batch_size=args.batch_size, shuffle=False)

    # 2. Собираем значения энергии для ID (7) и OOD (1)
    T = args.temperature
    energy_id, logits_id = collect_energy(model, loader_id_7, T,  device, use_bf16)
    energy_ood, logits_ood = collect_energy(model, loader_ood_1, T, device, use_bf16)

    # Сохраняем логиты для анализа
    logit_path = out_dir / 'logits.csv'
    df = pd.DataFrame(np.concatenate([logits_id, logits_ood], axis=0))  # Создает 10 колонок для логитов (0-9)
    df.insert(0, "energy", np.concatenate([energy_id, energy_ood]))
    df.insert(0, "is_ood", np.concatenate([np.zeros(len(energy_id)), np.ones(len(energy_ood))]))
    df.columns = ['OOD', 'F'] + [f'{i}' for i in range(logits_id.shape[1])]
    df.to_csv(logit_path, index=False)

    # 3. Построение и сохранение графиков
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Строим гистограммы плотности распределения (density=True)
    plt.hist(energy_id, bins=50, alpha=0.6, label='ID (Digit 7)', color='blue', edgecolor='k', density=True)
    plt.hist(energy_ood, bins=50, alpha=0.6, label='OOD (Digit 1)', color='red', edgecolor='k', density=True)

    plt.title(f'Helmholtz Energy Distribution for ID vs OOD (T={T})')
    plt.xlabel('Energy Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    # Сохраняем результат в out_dir
    plot_path = out_dir / 'energy_ood_histogram.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"OOD Analysis finished. Plot saved to: {plot_path}")
    print(f"Mean Energy ID (7): {energy_id.mean():.4f}")
    print(f"Mean Energy OOD (1): {energy_ood.mean():.4f}")

if __name__ == '__main__':
    main()