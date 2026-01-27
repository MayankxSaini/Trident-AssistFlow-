import os
import sys
sys.path.append(".")

from config import DATA_PATH, COL_PRIORITY, COL_TICKET_TYPE, VALID_PRIORITIES
from src.ingestion import load_and_prepare_data, prepare_training_data
from src.models import PriorityModel, IssueTypeModel


def train_priority_model(df):
    print("\n" + "=" * 60)
    print("TRAINING: PRIORITY PREDICTION MODEL")
    print("=" * 60)
    
    print("\nPreparing data...")
    X, y = prepare_training_data(df, COL_PRIORITY)
    print(f"Samples: {len(X)}")
    
    for priority in VALID_PRIORITIES:
        count = sum(1 for label in y if label == priority)
        print(f"  {priority}: {count} ({count/len(y)*100:.1f}%)")
    
    print("\nTraining...")
    model = PriorityModel()
    metrics = model.train(X, y, evaluate=True)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
    
    if 'classification_report' in metrics:
        print("\nPer-class metrics:")
        for cls in VALID_PRIORITIES:
            if cls in metrics['classification_report']:
                m = metrics['classification_report'][cls]
                print(f"  {cls}: P={m['precision']:.2%} R={m['recall']:.2%} F1={m['f1-score']:.2%}")
    
    model.save()
    return model


def train_issue_type_model(df):
    print("\n" + "=" * 60)
    print("TRAINING: ISSUE TYPE MODEL")
    print("=" * 60)
    
    if COL_TICKET_TYPE not in df.columns:
        print(f"Column '{COL_TICKET_TYPE}' not found. Skipping.")
        return None
    
    print("\nPreparing data...")
    X, y = prepare_training_data(df, COL_TICKET_TYPE)
    print(f"Samples: {len(X)}")
    
    unique_types = list(set(y))
    print(f"Issue types: {len(unique_types)}")
    for t in unique_types[:10]:
        count = sum(1 for label in y if label == t)
        print(f"  {t}: {count} ({count/len(y)*100:.1f}%)")
    
    print("\nTraining...")
    model = IssueTypeModel()
    metrics = model.train(X, y, evaluate=True)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
    
    model.save()
    return model


def main():
    print("=" * 60)
    print("ASSISTFLOW AI - MODEL TRAINING")
    print("=" * 60)
    
    os.makedirs("./models", exist_ok=True)
    
    print("\nLoading data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Loaded {len(df)} tickets")
    
    priority_model = train_priority_model(df)
    issue_type_model = train_issue_type_model(df)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nPriority Model: OK")
    print(f"Issue Type Model: {'OK' if issue_type_model else 'Skipped'}")
    print("\nModels saved to ./models/")
    print("\nRun demo.py to test the pipeline")


if __name__ == "__main__":
    main()
