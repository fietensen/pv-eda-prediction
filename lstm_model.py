import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, df, target_col, lookback_window, feature_cols=None):
        self.lookback_window = lookback_window
        self.target_col = target_col
        if feature_cols is None:
            self.feature_cols = [c for c in df.columns if c != target_col]
        else:
            self.feature_cols = feature_cols
        self.features = df[self.feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.n_samples = len(df) - lookback_window

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.lookback_window]
        y = self.targets[idx+self.lookback_window]
        return torch.tensor(x), torch.tensor([y])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMForecaster:
    def __init__(self, lookback_window=24, hidden_size=64, num_layers=2, dropout=0.2,
                 lr=0.001, batch_size=32, epochs=10, device=None):
        self.lookback_window = lookback_window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_cols = None

    def fit(self, df, target_col='y', feature_cols=None):
        df = df.copy()
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
        self.feature_cols = feature_cols

        dataset = SequenceDataset(df, target_col, self.lookback_window, self.feature_cols)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMModel(input_size=len(self.feature_cols),
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f}")

    def predict(self, df, target_col='y'):
        df = df.copy()
        dataset = SequenceDataset(df, target_col, self.lookback_window, self.feature_cols)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        self.model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                preds.append(self.model(X).cpu().item())
                actuals.append(y.item())
        return pd.DataFrame({"actual": actuals, "predicted": preds})
