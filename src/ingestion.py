"""
AssistFlow AI - Ticket Ingestion Module

WORKFLOW STEP 1: TICKET INGESTION
- Load ticket data from CSV
- Combine subject and description into full_text
- Store raw tickets without modification

This module handles all data loading and preprocessing operations.
"""

import pandas as pd
from typing import Optional

import sys
sys.path.append(".")
from config import (
    DATA_PATH,
    COL_TICKET_ID,
    COL_SUBJECT,
    COL_DESCRIPTION,
    COL_PRIORITY,
    COL_TICKET_TYPE,
    COL_TIME_TO_RESOLUTION,
    VALID_PRIORITIES
)


def load_raw_tickets(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw ticket data from CSV file without any modifications.
    
    Args:
        file_path: Path to the CSV file containing ticket data
        
    Returns:
        DataFrame with raw ticket data
    """
    df = pd.read_csv(file_path)
    return df


def create_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Ticket Subject and Ticket Description into a single text field.
    
    WHY: ML models work better with more context. Combining subject and 
    description provides richer text for TF-IDF vectorization.
    
    Args:
        df: DataFrame containing ticket data
        
    Returns:
        DataFrame with new 'full_text' column added
    """
    df = df.copy()
    
    # Handle missing values by converting to empty string
    subject = df[COL_SUBJECT].fillna("")
    description = df[COL_DESCRIPTION].fillna("")
    
    # Combine with a separator for clarity
    df["full_text"] = subject + " | " + description
    
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning for ML processing.
    
    WHY: Consistent text format improves model accuracy.
    We keep it simple - just lowercase and strip whitespace.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    return text


def prepare_training_data(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Prepare data for model training by filtering valid records.
    
    WHY: Models need clean, labeled data. We filter out records with
    missing targets or invalid priority values.
    
    Args:
        df: DataFrame with full_text column
        target_column: Name of the target column (e.g., COL_PRIORITY)
        
    Returns:
        Tuple of (X: texts, y: labels) ready for training
    """
    df = df.copy()
    
    # Remove rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # For priority prediction, filter to valid priority values only
    if target_column == COL_PRIORITY:
        df = df[df[target_column].isin(VALID_PRIORITIES)]
    
    # Remove rows with missing text
    df = df.dropna(subset=["full_text"])
    df = df[df["full_text"].str.strip() != ""]
    
    X = df["full_text"].apply(clean_text).values
    y = df[target_column].values
    
    return X, y


def load_and_prepare_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Complete data loading pipeline: load, create full_text, clean.
    
    This is the main entry point for ticket ingestion.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame ready for processing
    """
    # Step 1: Load raw data
    df = load_raw_tickets(file_path)
    
    # Step 2: Create combined text field
    df = create_full_text(df)
    
    # Step 3: Clean the full_text column
    df["full_text_clean"] = df["full_text"].apply(clean_text)
    
    return df


def parse_resolution_time(time_str: str) -> Optional[float]:
    """
    Parse Time to Resolution string and convert to hours.
    
    WHY: The dataset may have various time formats. We need consistent
    numeric hours for SLA calculations.
    
    Args:
        time_str: String representation of resolution time
        
    Returns:
        Resolution time in hours, or None if unparseable
    """
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    
    try:
        # Try direct conversion (assuming it might be in hours already)
        return float(time_str)
    except (ValueError, TypeError):
        pass
    
    # Handle common time formats like "2 days 3 hours"
    time_str = str(time_str).lower()
    total_hours = 0.0
    
    # Extract days
    if "day" in time_str:
        import re
        days_match = re.search(r"(\d+)\s*day", time_str)
        if days_match:
            total_hours += int(days_match.group(1)) * 24
    
    # Extract hours
    if "hour" in time_str:
        import re
        hours_match = re.search(r"(\d+)\s*hour", time_str)
        if hours_match:
            total_hours += int(hours_match.group(1))
    
    return total_hours if total_hours > 0 else None


if __name__ == "__main__":
    # Quick test of the module
    print("Loading ticket data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} tickets")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample full_text:\n{df['full_text'].iloc[0][:200]}...")
    print(f"\nPriority distribution:\n{df[COL_PRIORITY].value_counts()}")
