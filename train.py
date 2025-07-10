import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_utils import load_and_aggregate, ElectricityDataset
# from models import LSTMModel, TransformerModel
from models import LSTMModel, TransformerModel, AttentionLSTMModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x).cpu()
            preds.append(out)
            targets.append(y)
    if not preds:
        return float("nan"), float("nan")
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return mse, mae


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df = load_and_aggregate(args.train)
    test_df = load_and_aggregate(args.test)

    train_dataset = ElectricityDataset(train_df, args.input_days, args.pred_days)
    test_dataset = ElectricityDataset(test_df, args.input_days, args.pred_days)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    input_size = len(train_dataset.feature_cols)
    if args.model == 'lstm':
        model = LSTMModel(input_size, output_len=args.pred_days)
    elif args.model == 'transformer':
        model = TransformerModel(input_size, output_len=args.pred_days)
    else:
        model = AttentionLSTMModel(input_size, output_len=args.pred_days)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        mse, mae = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f} mse={mse:.4f} mae={mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train electricity prediction model")
    parser.add_argument('--train', type=Path, default='train.csv', help='Training CSV file')
    parser.add_argument('--test', type=Path, default='test.csv', help='Test CSV file')
    parser.add_argument('--model', choices=['lstm', 'transformer', 'lstm_attn'], default='lstm')
    parser.add_argument('--input-days', type=int, default=90)
    parser.add_argument('--pred-days', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)