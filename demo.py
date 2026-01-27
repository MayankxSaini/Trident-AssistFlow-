"""
AssistFlow AI - Demo Script

This script demonstrates the complete AssistFlow AI pipeline.
Run train_models.py first before running this demo.
"""

import sys
sys.path.append(".")

from src.pipeline import (
    AssistFlowPipeline,
    print_analysis_report,
    analyze_single_ticket
)
from src.ingestion import load_and_prepare_data
from config import DATA_PATH, COL_TICKET_ID, COL_TIME_TO_RESOLUTION


def demo_single_ticket():
    """Demonstrate analysis of a single ticket."""
    print("\n" + "=" * 70)
    print("DEMO 1: SINGLE TICKET ANALYSIS")
    print("=" * 70)
    
    # Sample ticket
    sample_ticket = """
    Product setup issue | I purchased a new laptop last week and I'm having 
    trouble with the initial setup. The screen keeps flickering during the 
    Windows installation process. I've tried restarting multiple times but 
    the issue persists. This is very urgent as I need this laptop for work 
    tomorrow. Please help immediately.
    """
    
    print("\n📝 Sample Ticket:")
    print("-" * 50)
    print(sample_ticket.strip())
    print("-" * 50)
    
    # Analyze ticket
    result = analyze_single_ticket(
        text=sample_ticket,
        ticket_id="DEMO-001",
        resolution_hours=20.0  # Simulated: took 20 hours
    )
    
    # Print full report
    print_analysis_report(result)


def demo_batch_analysis():
    """Demonstrate batch analysis of tickets from the dataset."""
    print("\n" + "=" * 70)
    print("DEMO 2: BATCH TICKET ANALYSIS")
    print("=" * 70)
    
    # Load sample tickets from dataset
    print("\n📂 Loading tickets from dataset...")
    df = load_and_prepare_data(DATA_PATH)
    
    # Take first 5 tickets for demo
    sample_df = df.head(5)
    print(f"Analyzing {len(sample_df)} sample tickets...")
    
    # Initialize pipeline
    pipeline = AssistFlowPipeline()
    if not pipeline.load_models():
        print("ERROR: Models not loaded. Run train_models.py first.")
        return
    
    # Analyze batch
    results = pipeline.analyze_batch(sample_df)
    
    # Print summary
    print("\n📊 BATCH ANALYSIS SUMMARY:")
    print("-" * 50)
    
    for result in results:
        print(f"\nTicket {result.ticket_id}:")
        print(f"  Priority: {result.predicted_priority} → {result.final_priority}")
        print(f"  Issue Type: {result.issue_type or 'N/A'}")
        print(f"  SLA: {result.sla_hours}h ({result.sla_status})")
        print(f"  Handler: {result.handler_type}")
        if result.was_escalated:
            print(f"  ⬆️ ESCALATED")
    
    # Convert to DataFrame
    results_df = pipeline.results_to_dataframe(results)
    print("\n\n📋 Results DataFrame Columns:")
    print(list(results_df.columns))


def demo_workflow_explanation():
    """Demonstrate the workflow with explanations."""
    print("\n" + "=" * 70)
    print("ASSISTFLOW AI - WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    print("""
    AssistFlow AI processes tickets in this exact order:
    
    1️⃣  TICKET INGESTION
        Load ticket, combine subject + description into full_text
        
    2️⃣  PRIORITY PREDICTION (ML)
        TF-IDF + Logistic Regression predicts priority
        
    3️⃣  ISSUE TYPE PREDICTION (ML - Optional)
        TF-IDF + Logistic Regression predicts issue type
        
    4️⃣  ─────── ML STOPS HERE ───────
    
    5️⃣  SLA ASSIGNMENT (RULES)
        Assign hours: Low=72, Medium=48, High=24, Critical=6
        
    6️⃣  SLA STATUS CALCULATION (RULES)
        Compare resolution time to SLA → met/at_risk/breached
        
    7️⃣  ESCALATION (RULES)
        If at_risk: Medium→High, High→Critical
        
    8️⃣  HANDLER DECISION (RULES)
        High/Critical priority OR Billing/Security → Human
        Otherwise → AI allowed
        
    9️⃣  LLM ASSISTANCE (AFTER ALL DECISIONS)
        Generate summary, explanation, suggested response
        LLM can NOT change any decisions
        
    🔟  OUTPUT COMPLETE ANALYSIS
        All fields populated for review
    """)


def main():
    """Run all demos."""
    print("=" * 70)
    print("ASSISTFLOW AI - DEMONSTRATION")
    print("=" * 70)
    
    # Show workflow
    demo_workflow_explanation()
    
    # Demo single ticket analysis
    demo_single_ticket()
    
    # Demo batch analysis
    demo_batch_analysis()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAssistFlow AI is ready for use!")
    print("Use src/pipeline.py for integration into your applications.")


if __name__ == "__main__":
    main()
