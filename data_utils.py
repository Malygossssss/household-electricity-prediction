import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Load minute level csv and aggregate to daily statistics."""
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    df["Date"] = df["DateTime"].dt.date

    agg_dict = {
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
    }
    # Weather related columns if present
    for col in ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    daily = df.groupby("Date").agg(agg_dict).reset_index()
    return daily


class ElectricityDataset(Dataset):
    """Sliding window dataset for sequence to sequence forecasting."""

    def __init__(self, data: pd.DataFrame, input_days: int = 90, pred_days: int = 90):
        self.feature_cols = [c for c in data.columns if c != "Date"]
        values = data[self.feature_cols].values.astype(np.float32)
        self.inputs = []
        self.targets = []
        for start in range(len(data) - input_days - pred_days + 1):
            end = start + input_days
            self.inputs.append(values[start:end])
            target_slice = values[end : end + pred_days, 0]  # assume first column is target
            self.targets.append(target_slice)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.inputs)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.tensor(self.inputs[idx])
        y = torch.tensor(self.targets[idx])
        return x, y