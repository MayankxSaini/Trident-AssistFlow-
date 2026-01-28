"""Demo script for AssistFlow AI"""

import sys
sys.path.append(".")

from src.pipeline import AssistFlowPipeline, print_analysis_report, analyze_single_ticket
from src.ingestion import load_and_prepare_data
from config import DATA_PATH, COL_TICKET_ID


def demo_single_ticket():
    print("\n" + "=" * 70)
    print("DEMO: SINGLE TICKET ANALYSIS")
    print("=" * 70)
    
    sample_ticket = """
    Product setup issue | I purchased a new laptop last week and having 
    trouble with initial setup. Screen keeps flickering during Windows 
    installation. Tried restarting multiple times but issue persists. 
    Need this for work tomorrow. Please help immediately.
    """
    
    print("\nSample Ticket:")
    print("-" * 50)
    print(sample_ticket.strip())
    print("-" * 50)
    
    result = analyze_single_ticket(
        text=sample_ticket,
        ticket_id="DEMO-001",
        resolution_hours=20.0
    )
    print_analysis_report(result)


def demo_batch_analysis():
    print("\n" + "=" * 70)
    print("DEMO: BATCH ANALYSIS")
    print("=" * 70)
    
    print("\nLoading dataset...")
    df = load_and_prepare_data(DATA_PATH)
    sample_df = df.head(5)
    print(f"Analyzing {len(sample_df)} tickets...")
    
    pipeline = AssistFlowPipeline()
    if not pipeline.load_models():
        print("ERROR: Models not loaded. Run train_models.py first.")
        return
    
    results = pipeline.analyze_batch(sample_df)
    
    print("\nBatch Results:")
    print("-" * 50)
    
    for result in results:
        print(f"\nTicket {result.ticket_id}:")
        print(f"  Priority: {result.predicted_priority} -> {result.final_priority}")
        print(f"  Issue: {result.issue_type or 'N/A'}")
        print(f"  SLA: {result.sla_hours}h ({result.sla_status})")
        print(f"  Handler: {result.handler_type}")
        if result.was_escalated:
            print(f"  ESCALATED")


def demo_workflow():
    print("\n" + "=" * 70)
    print("ASSISTFLOW AI WORKFLOW")
    print("=" * 70)
    print("""
    Pipeline:
    1. Ticket Ingestion -> combine subject + description
    2. Priority Prediction (ML) -> TF-IDF + LogReg
    3. Issue Type Prediction (ML) -> TF-IDF + LogReg
    4. SLA Assignment -> Low=72h, Medium=48h, High=24h, Critical=6h
    5. SLA Status -> compare to limit
    6. Escalation -> at_risk: Medium->High, High->Critical
    7. Handler Decision -> High/Critical/Billing/Security -> Human
    8. LLM Assistance -> summary and response
    """)


def main():
    print("=" * 70)
    print("ASSISTFLOW AI DEMO")
    print("=" * 70)
    
    demo_workflow()
    demo_single_ticket()
    demo_batch_analysis()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
