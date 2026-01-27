"""
AssistFlow AI - Main Pipeline

WORKFLOW STEP 10: OUTPUT STRUCTURE

This module orchestrates all components in the correct order:
1. Ticket Ingestion
2. Priority Prediction (ML)
3. Issue Type Prediction (ML - Optional)
4. Business Rules (SLA, Escalation)
5. Handler Decision (Rule-based)
6. LLM Assistance (After all decisions)

Every analyzed ticket produces:
- predicted_priority (from Model 1)
- issue_type (from Model 2, if available)
- sla_hours
- sla_status
- final_priority (after escalation)
- handler_type (AI or Human)
- explanation_text (from LLM)
- suggested_response (from LLM)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import pandas as pd

import sys
sys.path.append(".")
from config import (
    COL_TICKET_ID,
    COL_TIME_TO_RESOLUTION,
    COL_PRIORITY,
    COL_TICKET_TYPE
)
from src.ingestion import (
    load_and_prepare_data,
    parse_resolution_time,
    clean_text
)
from src.models import PriorityModel, IssueTypeModel
from src.business_rules import process_business_rules
from src.handler_decision import determine_handler
from src.llm_assistance import generate_llm_assistance


@dataclass
class TicketAnalysisResult:
    """
    Complete output structure for an analyzed ticket.
    
    This contains ALL required outputs as specified in the workflow.
    """
    # Ticket identification
    ticket_id: Any
    full_text: str
    
    # ML Predictions (Step 2 & 3)
    predicted_priority: str
    priority_confidence: float
    issue_type: Optional[str]
    issue_type_confidence: Optional[float]
    
    # Business Rules Results (Steps 5, 6, 7)
    sla_hours: int
    sla_status: str
    was_escalated: bool
    escalation_reason: Optional[str]
    final_priority: str
    
    # Handler Decision (Step 8)
    handler_type: str
    handler_reason: str
    
    # LLM Assistance (Step 9)
    ticket_summary: str
    explanation_text: str
    suggested_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)


class AssistFlowPipeline:
    """
    Main pipeline orchestrating all AssistFlow AI components.
    
    This class ensures the correct workflow order is followed:
    1. ML predictions first
    2. Business rules applied to ML outputs
    3. LLM assistance generated AFTER all decisions are final
    """
    
    def __init__(self):
        """Initialize the pipeline with both models."""
        self.priority_model = PriorityModel()
        self.issue_type_model = IssueTypeModel()
        self._models_loaded = False
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if priority model loaded successfully
        """
        priority_loaded = self.priority_model.load()
        issue_loaded = self.issue_type_model.load()
        
        if not priority_loaded:
            print("ERROR: Priority model not found. Please train models first.")
            return False
        
        if not issue_loaded:
            print("WARNING: Issue type model not found. Continuing without it.")
        
        self._models_loaded = priority_loaded
        return priority_loaded
    
    def analyze_ticket(
        self,
        full_text: str,
        ticket_id: Any = None,
        time_to_resolution_hours: Optional[float] = None
    ) -> TicketAnalysisResult:
        """
        Analyze a single ticket through the complete pipeline.
        
        This is the main entry point for ticket analysis.
        
        Args:
            full_text: Combined subject and description text
            ticket_id: Unique identifier for the ticket
            time_to_resolution_hours: Historical resolution time (for SLA simulation)
            
        Returns:
            TicketAnalysisResult containing all analysis outputs
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Clean text for prediction
        clean_full_text = clean_text(full_text)
        
        # =========================================================
        # STEP 2: PRIORITY PREDICTION (ML)
        # =========================================================
        predicted_priority, priority_confidence = self.priority_model.get_prediction_confidence(
            clean_full_text
        )
        
        # =========================================================
        # STEP 3: ISSUE TYPE PREDICTION (ML - Optional)
        # =========================================================
        issue_type = None
        issue_type_confidence = None
        
        if self.issue_type_model.model is not None:
            try:
                issue_type, issue_type_confidence = self.issue_type_model.get_prediction_confidence(
                    clean_full_text
                )
            except Exception as e:
                print(f"Warning: Issue type prediction failed: {e}")
        
        # =========================================================
        # STEPS 5, 6, 7: BUSINESS RULES (SLA + ESCALATION)
        # NO MORE ML FROM HERE - PURE RULES
        # =========================================================
        sla_result, escalation_result = process_business_rules(
            predicted_priority=predicted_priority,
            time_to_resolution_hours=time_to_resolution_hours
        )
        
        # =========================================================
        # STEP 8: HANDLER DECISION (RULES, NOT ML)
        # =========================================================
        handler_decision = determine_handler(
            final_priority=escalation_result.final_priority,
            issue_type=issue_type
        )
        
        # =========================================================
        # STEP 9: LLM ASSISTANCE (AFTER ALL DECISIONS)
        # =========================================================
        llm_result = generate_llm_assistance(
            full_text=full_text,
            predicted_priority=predicted_priority,
            final_priority=escalation_result.final_priority,
            issue_type=issue_type,
            sla_hours=sla_result.sla_hours,
            sla_status=sla_result.sla_status,
            was_escalated=escalation_result.was_escalated,
            handler_type=handler_decision.handler_type
        )
        
        # =========================================================
        # STEP 10: COMPILE OUTPUT STRUCTURE
        # =========================================================
        return TicketAnalysisResult(
            ticket_id=ticket_id,
            full_text=full_text,
            predicted_priority=predicted_priority,
            priority_confidence=priority_confidence,
            issue_type=issue_type,
            issue_type_confidence=issue_type_confidence,
            sla_hours=sla_result.sla_hours,
            sla_status=sla_result.sla_status,
            was_escalated=escalation_result.was_escalated,
            escalation_reason=escalation_result.escalation_reason,
            final_priority=escalation_result.final_priority,
            handler_type=handler_decision.handler_type,
            handler_reason=handler_decision.reason,
            ticket_summary=llm_result.ticket_summary,
            explanation_text=llm_result.explanation_text,
            suggested_response=llm_result.suggested_response
        )
    
    def analyze_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "full_text",
        id_column: str = COL_TICKET_ID,
        resolution_column: str = COL_TIME_TO_RESOLUTION
    ) -> List[TicketAnalysisResult]:
        """
        Analyze a batch of tickets from a DataFrame.
        
        Args:
            df: DataFrame containing ticket data
            text_column: Name of the text column
            id_column: Name of the ID column
            resolution_column: Name of the resolution time column
            
        Returns:
            List of TicketAnalysisResult objects
        """
        results = []
        
        for idx, row in df.iterrows():
            full_text = row.get(text_column, "")
            ticket_id = row.get(id_column, idx)
            
            # Parse resolution time for SLA simulation
            resolution_time = None
            if resolution_column in row:
                resolution_time = parse_resolution_time(row[resolution_column])
            
            try:
                result = self.analyze_ticket(
                    full_text=full_text,
                    ticket_id=ticket_id,
                    time_to_resolution_hours=resolution_time
                )
                results.append(result)
            except Exception as e:
                print(f"Error analyzing ticket {ticket_id}: {e}")
        
        return results
    
    def results_to_dataframe(
        self,
        results: List[TicketAnalysisResult]
    ) -> pd.DataFrame:
        """
        Convert analysis results to a DataFrame.
        
        Args:
            results: List of TicketAnalysisResult objects
            
        Returns:
            DataFrame with all result fields
        """
        records = [result.to_dict() for result in results]
        return pd.DataFrame(records)


def analyze_single_ticket(
    text: str,
    ticket_id: Any = None,
    resolution_hours: Optional[float] = None
) -> TicketAnalysisResult:
    """
    Convenience function to analyze a single ticket.
    
    Initializes pipeline, loads models, and analyzes the ticket.
    Use this for one-off analysis or testing.
    
    Args:
        text: Ticket text (subject + description)
        ticket_id: Unique identifier
        resolution_hours: Historical resolution time
        
    Returns:
        TicketAnalysisResult
    """
    pipeline = AssistFlowPipeline()
    if not pipeline.load_models():
        raise RuntimeError("Failed to load models")
    
    return pipeline.analyze_ticket(
        full_text=text,
        ticket_id=ticket_id,
        time_to_resolution_hours=resolution_hours
    )


def print_analysis_report(result: TicketAnalysisResult) -> None:
    """
    Print a formatted analysis report for a ticket.
    
    Args:
        result: TicketAnalysisResult to display
    """
    print("=" * 70)
    print(f"TICKET ANALYSIS REPORT - ID: {result.ticket_id}")
    print("=" * 70)
    
    print(f"\n📝 TICKET SUMMARY:")
    print(f"   {result.ticket_summary}")
    
    print(f"\n🎯 ML PREDICTIONS:")
    print(f"   Priority: {result.predicted_priority} (confidence: {result.priority_confidence:.2%})")
    if result.issue_type:
        print(f"   Issue Type: {result.issue_type} (confidence: {result.issue_type_confidence:.2%})")
    
    print(f"\n📊 BUSINESS RULES RESULTS:")
    print(f"   SLA Hours: {result.sla_hours}")
    print(f"   SLA Status: {result.sla_status.upper()}")
    print(f"   Final Priority: {result.final_priority}")
    if result.was_escalated:
        print(f"   ⬆️ ESCALATED: {result.escalation_reason}")
    
    print(f"\n👤 HANDLER DECISION:")
    print(f"   Handler: {result.handler_type}")
    print(f"   Reason: {result.handler_reason}")
    
    print(f"\n📖 EXPLANATION:")
    print(result.explanation_text)
    
    print(f"\n✉️ SUGGESTED RESPONSE:")
    print("-" * 50)
    print(result.suggested_response)
    print("-" * 50)


if __name__ == "__main__":
    print("AssistFlow AI - Main Pipeline")
    print("Run train_models.py first to train the models.")
    print("Then use this module to analyze tickets.")
