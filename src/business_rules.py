"""Business rules for SLA and escalation"""

from typing import Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
sys.path.append(".")
from config import SLA_HOURS, SLA_AT_RISK_THRESHOLD, ESCALATION_MAP, VALID_PRIORITIES


@dataclass
class SLAResult:
    sla_hours: int
    sla_status: str
    time_to_resolution_hours: Optional[float]
    sla_deadline: Optional[datetime]


@dataclass
class EscalationResult:
    original_priority: str
    final_priority: str
    was_escalated: bool
    escalation_reason: Optional[str]


def assign_sla_hours(priority: str) -> int:
    if priority not in SLA_HOURS:
        print(f"Warning: Unknown priority '{priority}', using Critical SLA")
        return SLA_HOURS["Critical"]
    return SLA_HOURS[priority]


def calculate_sla_deadline(ticket_creation_time: datetime, sla_hours: int) -> datetime:
    return ticket_creation_time + timedelta(hours=sla_hours)


def calculate_sla_status(time_to_resolution_hours: Optional[float], sla_hours: int) -> str:
    if time_to_resolution_hours is None:
        return "at_risk"
    
    at_risk_threshold = sla_hours * SLA_AT_RISK_THRESHOLD
    
    if time_to_resolution_hours > sla_hours:
        return "breached"
    elif time_to_resolution_hours > at_risk_threshold:
        return "at_risk"
    return "met"


def process_sla(priority: str, time_to_resolution_hours: Optional[float],
                ticket_creation_time: Optional[datetime] = None) -> SLAResult:
    sla_hours = assign_sla_hours(priority)
    sla_deadline = None
    if ticket_creation_time:
        sla_deadline = calculate_sla_deadline(ticket_creation_time, sla_hours)
    sla_status = calculate_sla_status(time_to_resolution_hours, sla_hours)
    
    return SLAResult(
        sla_hours=sla_hours,
        sla_status=sla_status,
        time_to_resolution_hours=time_to_resolution_hours,
        sla_deadline=sla_deadline
    )


def apply_escalation(predicted_priority: str, sla_status: str) -> EscalationResult:
    original_priority = predicted_priority
    final_priority = predicted_priority
    was_escalated = False
    escalation_reason = None
    
    if sla_status == "at_risk":
        if predicted_priority in ESCALATION_MAP:
            final_priority = ESCALATION_MAP[predicted_priority]
            was_escalated = True
            escalation_reason = f"Escalated from {original_priority} to {final_priority} due to SLA at-risk"
    elif sla_status == "breached":
        escalation_reason = f"SLA breached. Priority remains {original_priority}. Immediate action required."
    
    return EscalationResult(
        original_priority=original_priority,
        final_priority=final_priority,
        was_escalated=was_escalated,
        escalation_reason=escalation_reason
    )


def process_business_rules(predicted_priority: str, time_to_resolution_hours: Optional[float],
                           ticket_creation_time: Optional[datetime] = None) -> Tuple[SLAResult, EscalationResult]:
    sla_result = process_sla(priority=predicted_priority, time_to_resolution_hours=time_to_resolution_hours,
                             ticket_creation_time=ticket_creation_time)
    escalation_result = apply_escalation(predicted_priority=predicted_priority, sla_status=sla_result.sla_status)
    return sla_result, escalation_result


if __name__ == "__main__":
    print("Testing business rules\n")
    
    print("Test 1: Medium priority, at risk (40h, limit 48h)")
    sla, esc = process_business_rules("Medium", 40.0)
    print(f"  SLA: {sla.sla_hours}h, Status: {sla.sla_status}")
    print(f"  Final: {esc.final_priority}, Escalated: {esc.was_escalated}")
    
    print("\nTest 2: High priority, met (10h, limit 24h)")
    sla, esc = process_business_rules("High", 10.0)
    print(f"  SLA: {sla.sla_hours}h, Status: {sla.sla_status}")
    print(f"  Final: {esc.final_priority}, Escalated: {esc.was_escalated}")
