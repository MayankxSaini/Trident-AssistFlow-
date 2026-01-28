"""Ticket ingestion and data loading"""

import pandas as pd
import re
from typing import Optional

import sys
sys.path.append(".")
from config import (
    DATA_PATH, COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION,
    COL_PRIORITY, COL_TICKET_TYPE, VALID_PRIORITIES
)


def load_raw_tickets(file_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path)


def create_full_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subject = df[COL_SUBJECT].fillna("")
    description = df[COL_DESCRIPTION].fillna("")
    df["full_text"] = subject + " | " + description
    return df


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def prepare_training_data(df: pd.DataFrame, target_column: str) -> tuple:
    df = df.copy()
    df = df.dropna(subset=[target_column])
    
    if target_column == COL_PRIORITY:
        df = df[df[target_column].isin(VALID_PRIORITIES)]
    
    df = df.dropna(subset=["full_text"])
    df = df[df["full_text"].str.strip() != ""]
    
    X = df["full_text"].apply(clean_text).values
    y = df[target_column].values
    return X, y


def load_and_prepare_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    df = load_raw_tickets(file_path)
    df = create_full_text(df)
    df["full_text_clean"] = df["full_text"].apply(clean_text)
    return df


def parse_resolution_time(time_str: str) -> Optional[float]:
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    
    try:
        return float(time_str)
    except (ValueError, TypeError):
        pass
    
    time_str = str(time_str).lower()
    total_hours = 0.0
    
    if "day" in time_str:
        match = re.search(r"(\d+)\s*day", time_str)
        if match:
            total_hours += int(match.group(1)) * 24
    
    if "hour" in time_str:
        match = re.search(r"(\d+)\s*hour", time_str)
        if match:
            total_hours += int(match.group(1))
    
    return total_hours if total_hours > 0 else None


if __name__ == "__main__":
    print("Loading tickets...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} tickets")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df['full_text'].iloc[0][:200]}...")
    print(f"\nPriority dist:\n{df[COL_PRIORITY].value_counts()}")
