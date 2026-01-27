import os
import sys
sys.path.append(".")

from config import (
    DATA_PATH,
    COL_PRIORITY,
    COL_TICKET_TYPE,
    VALID_PRIORITIES
)
from src.ingestion import load_and_prepare_data, prepare_training_data
from src.models import PriorityModel, IssueTypeModel


def train_priority_model(df):
    """
    Train the priority prediction model.
    
    This is the MANDATORY model for AssistFlow AI.
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: PRIORITY PREDICTION")
    print("=" * 60)
    
    # Prepare training data
    print("\nPreparing training data...")
    X, y = prepare_training_data(df, COL_PRIORITY)
    print(f"Training samples: {len(X)}")
    print(f"Priority distribution:")
    for priority in VALID_PRIORITIES:
        count = sum(1 for label in y if label == priority)
        print(f"  {priority}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train model
    print("\nTraining model (TF-IDF + Logistic Regression)...")
    model = PriorityModel()
    metrics = model.train(X, y, evaluate=True)
    
    # Print results
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   Train Samples: {metrics['train_samples']}")
    print(f"   Test Samples: {metrics['test_samples']}")
    
    if 'classification_report' in metrics:
        print("\n   Per-class Performance:")
        for cls in VALID_PRIORITIES:
            if cls in metrics['classification_report']:
                cls_metrics = metrics['classification_report'][cls]
                print(f"   {cls}:")
                print(f"      Precision: {cls_metrics['precision']:.2%}")
                print(f"      Recall: {cls_metrics['recall']:.2%}")
                print(f"      F1-Score: {cls_metrics['f1-score']:.2%}")
    
    # Save model
    print("\nSaving model...")
    model.save()
    
    return model


def train_issue_type_model(df):
    """
    Train the issue type prediction model.
    
    This is an OPTIONAL model that supports routing and explanation.
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: ISSUE TYPE PREDICTION")
    print("=" * 60)
    
    # Check if issue type column exists
    if COL_TICKET_TYPE not in df.columns:
        print(f"WARNING: Column '{COL_TICKET_TYPE}' not found. Skipping issue type model.")
        return None
    
    # Prepare training data
    print("\nPreparing training data...")
    X, y = prepare_training_data(df, COL_TICKET_TYPE)
    print(f"Training samples: {len(X)}")
    
    # Get unique issue types
    unique_types = list(set(y))
    print(f"Issue types found: {len(unique_types)}")
    for issue_type in unique_types[:10]:  # Show first 10
        count = sum(1 for label in y if label == issue_type)
        print(f"  {issue_type}: {count} ({count/len(y)*100:.1f}%)")
    if len(unique_types) > 10:
        print(f"  ... and {len(unique_types) - 10} more")
    
    # Train model
    print("\nTraining model (TF-IDF + Logistic Regression)...")
    model = IssueTypeModel()
    metrics = model.train(X, y, evaluate=True)
    
    # Print results
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   Train Samples: {metrics['train_samples']}")
    print(f"   Test Samples: {metrics['test_samples']}")
    
    # Save model
    print("\nSaving model...")
    model.save()
    
    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ASSISTFLOW AI - MODEL TRAINING")
    print("=" * 60)
    
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n📂 Loading ticket data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Loaded {len(df)} tickets")
    
    # Step 2: Train priority model (mandatory)
    priority_model = train_priority_model(df)
    
    # Step 3: Train issue type model (optional)
    issue_type_model = train_issue_type_model(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\n✅ Priority Model: Trained and saved")
    if issue_type_model:
        print("✅ Issue Type Model: Trained and saved")
    else:
        print("⚠️ Issue Type Model: Not trained (column not found)")
    
    print("\n📁 Model files saved to ./models/")
    print("\nNext steps:")
    print("1. Run demo.py to see the pipeline in action")
    print("2. Use pipeline.py to analyze new tickets")


if __name__ == "__main__":
    main()
