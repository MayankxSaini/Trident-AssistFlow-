"""Main pipeline for ticket analysis"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import pandas as pd

import sys
sys.path.append(".")
from config import COL_TICKET_ID, COL_TIME_TO_RESOLUTION
from src.ingestion import load_and_prepare_data, parse_resolution_time, clean_text
from src.models import PriorityModel, IssueTypeModel
from src.business_rules import process_business_rules
from src.handler_decision import determine_handler
from src.llm_assistance import generate_llm_assistance


@dataclass
class TicketAnalysisResult:
    ticket_id: Any
    full_text: str
    predicted_priority: str
    priority_confidence: float
    issue_type: Optional[str]
    issue_type_confidence: Optional[float]
    sla_hours: int
    sla_status: str
    was_escalated: bool
    escalation_reason: Optional[str]
    final_priority: str
    handler_type: str
    handler_reason: str
    ticket_summary: str
    explanation_text: str
    suggested_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssistFlowPipeline:
    def __init__(self):
        self.priority_model = PriorityModel()
        self.issue_type_model = IssueTypeModel()
        self._models_loaded = False
    
    def load_models(self) -> bool:
        priority_loaded = self.priority_model.load()
        issue_loaded = self.issue_type_model.load()
        
        if not priority_loaded:
            print("ERROR: Priority model not found. Train models first.")
            return False
        
        if not issue_loaded:
            print("WARNING: Issue type model not found.")
        
        self._models_loaded = priority_loaded
        return priority_loaded
    
    def analyze_ticket(self, full_text: str, ticket_id: Any = None,
                       time_to_resolution_hours: Optional[float] = None) -> TicketAnalysisResult:
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        clean_full_text = clean_text(full_text)
        
        # ML predictions
        predicted_priority, priority_confidence = self.priority_model.get_prediction_confidence(clean_full_text)
        
        issue_type = None
        issue_type_confidence = None
        if self.issue_type_model.model is not None:
            try:
                issue_type, issue_type_confidence = self.issue_type_model.get_prediction_confidence(clean_full_text)
            except Exception as e:
                print(f"Issue type prediction failed: {e}")
        
        # Business rules
        sla_result, escalation_result = process_business_rules(
            predicted_priority=predicted_priority,
            time_to_resolution_hours=time_to_resolution_hours
        )
        
        # Handler decision
        handler_decision = determine_handler(
            final_priority=escalation_result.final_priority,
            issue_type=issue_type
        )
        
        # LLM assistance
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
    
    def analyze_batch(self, df: pd.DataFrame, text_column: str = "full_text",
                      id_column: str = COL_TICKET_ID,
                      resolution_column: str = COL_TIME_TO_RESOLUTION) -> List[TicketAnalysisResult]:
        results = []
        
        for idx, row in df.iterrows():
            full_text = row.get(text_column, "")
            ticket_id = row.get(id_column, idx)
            
            resolution_time = None
            if resolution_column in row:
                resolution_time = parse_resolution_time(row[resolution_column])
            
            try:
                result = self.analyze_ticket(full_text=full_text, ticket_id=ticket_id,
                                             time_to_resolution_hours=resolution_time)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing ticket {ticket_id}: {e}")
        
        return results
    
    def results_to_dataframe(self, results: List[TicketAnalysisResult]) -> pd.DataFrame:
        records = [result.to_dict() for result in results]
        return pd.DataFrame(records)


def analyze_single_ticket(text: str, ticket_id: Any = None,
                          resolution_hours: Optional[float] = None) -> TicketAnalysisResult:
    pipeline = AssistFlowPipeline()
    if not pipeline.load_models():
        raise RuntimeError("Failed to load models")
    return pipeline.analyze_ticket(full_text=text, ticket_id=ticket_id, time_to_resolution_hours=resolution_hours)


def print_analysis_report(result: TicketAnalysisResult) -> None:
    print("=" * 70)
    print(f"TICKET ANALYSIS: {result.ticket_id}")
    print("=" * 70)
    
    print(f"\n📝 SUMMARY: {result.ticket_summary}")
    
    print(f"\n🎯 ML PREDICTIONS:")
    print(f"   Priority: {result.predicted_priority} ({result.priority_confidence:.1%})")
    if result.issue_type:
        print(f"   Issue: {result.issue_type} ({result.issue_type_confidence:.1%})")
    
    print(f"\n📊 BUSINESS RULES:")
    print(f"   SLA: {result.sla_hours}h | Status: {result.sla_status.upper()}")
    print(f"   Final Priority: {result.final_priority}")
    if result.was_escalated:
        print(f"   ⬆️ ESCALATED: {result.escalation_reason}")
    
    print(f"\n👤 HANDLER: {result.handler_type}")
    print(f"   {result.handler_reason}")
    
    print(f"\n📖 EXPLANATION:\n{result.explanation_text}")
    
    print(f"\n✉️ SUGGESTED RESPONSE:\n{'-' * 50}\n{result.suggested_response}\n{'-' * 50}")


if __name__ == "__main__":
    print("AssistFlow AI Pipeline")
    print("Run train_models.py first, then use this module to analyze tickets.")
