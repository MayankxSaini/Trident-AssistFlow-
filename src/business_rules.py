"""
AssistFlow AI - Business Rules Module

WORKFLOW STEPS 4-7: BUSINESS RULES LAYER

After ML predictions, we STOP using ML and apply deterministic rules only.

This module handles:
- SLA Assignment (Step 5): Static hours based on priority
- SLA Status Calculation (Step 6): Simulated using historical data
- SLA-Aware Escalation (Step 7): Rule-based priority upgrades

IMPORTANT: NO AI/ML IN THIS MODULE
All logic here is pure Python business rules.
"""

from typing import Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
sys.path.append(".")
from config import (
    SLA_HOURS,
    SLA_AT_RISK_THRESHOLD,
    ESCALATION_MAP,
    VALID_PRIORITIES
)


@dataclass
class SLAResult:
    """
    Container for SLA calculation results.
    
    Provides a clean structure for passing SLA information through the pipeline.
    """
    sla_hours: int
    sla_status: str  # "met", "at_risk", "breached"
    time_to_resolution_hours: Optional[float]
    sla_deadline: Optional[datetime]


@dataclass
class EscalationResult:
    """
    Container for escalation decision results.
    
    Tracks whether escalation occurred and why.
    """
    original_priority: str
    final_priority: str
    was_escalated: bool
    escalation_reason: Optional[str]


# =============================================================================
# STEP 5: SLA ASSIGNMENT (STATIC RULES)
# =============================================================================

def assign_sla_hours(priority: str) -> int:
    """
    Assign SLA hours based on ticket priority.
    
    WHY: SLA is a BUSINESS RULE, not a prediction. Each priority level
    has a fixed maximum resolution time defined by company policy.
    
    Rules:
        - Low → 72 hours
        - Medium → 48 hours
        - High → 24 hours
        - Critical → 6 hours
    
    Args:
        priority: Predicted or current priority level
        
    Returns:
        SLA hours for this priority level
    """
    # Validate priority
    if priority not in SLA_HOURS:
        # Default to most restrictive if unknown priority
        print(f"Warning: Unknown priority '{priority}', defaulting to Critical SLA")
        return SLA_HOURS["Critical"]
    
    return SLA_HOURS[priority]


def calculate_sla_deadline(
    ticket_creation_time: datetime, 
    sla_hours: int
) -> datetime:
    """
    Calculate the SLA deadline from ticket creation time.
    
    Args:
        ticket_creation_time: When the ticket was created
        sla_hours: Maximum hours allowed for resolution
        
    Returns:
        DateTime of SLA deadline
    """
    return ticket_creation_time + timedelta(hours=sla_hours)


# =============================================================================
# STEP 6: SLA STATUS CALCULATION (SIMULATED USING DATASET)
# =============================================================================

def calculate_sla_status(
    time_to_resolution_hours: Optional[float], 
    sla_hours: int
) -> str:
    """
    Determine SLA status by comparing resolution time to SLA limit.
    
    WHY: This is a SIMULATION using historical data. In production,
    you would compare current time against SLA deadline.
    
    Status Logic:
        - If resolution time > SLA hours → "breached"
        - If resolution time > 80% of SLA hours → "at_risk"  
        - Otherwise → "met"
    
    Args:
        time_to_resolution_hours: Actual time taken to resolve (from dataset)
        sla_hours: Maximum allowed hours
        
    Returns:
        SLA status string: "met", "at_risk", or "breached"
    """
    # If no resolution time available, treat as at_risk
    # WHY: Missing data should be handled conservatively
    if time_to_resolution_hours is None:
        return "at_risk"
    
    # Calculate thresholds
    at_risk_threshold = sla_hours * SLA_AT_RISK_THRESHOLD
    
    if time_to_resolution_hours > sla_hours:
        return "breached"
    elif time_to_resolution_hours > at_risk_threshold:
        return "at_risk"
    else:
        return "met"


def process_sla(
    priority: str,
    time_to_resolution_hours: Optional[float],
    ticket_creation_time: Optional[datetime] = None
) -> SLAResult:
    """
    Complete SLA processing: assign hours, calculate deadline, determine status.
    
    Args:
        priority: Current ticket priority
        time_to_resolution_hours: Historical resolution time (for simulation)
        ticket_creation_time: When ticket was created (optional)
        
    Returns:
        SLAResult containing all SLA information
    """
    # Step 5a: Assign SLA hours based on priority
    sla_hours = assign_sla_hours(priority)
    
    # Step 5b: Calculate deadline if creation time provided
    sla_deadline = None
    if ticket_creation_time:
        sla_deadline = calculate_sla_deadline(ticket_creation_time, sla_hours)
    
    # Step 6: Calculate SLA status (simulation)
    sla_status = calculate_sla_status(time_to_resolution_hours, sla_hours)
    
    return SLAResult(
        sla_hours=sla_hours,
        sla_status=sla_status,
        time_to_resolution_hours=time_to_resolution_hours,
        sla_deadline=sla_deadline
    )


# =============================================================================
# STEP 7: SLA-AWARE ESCALATION (KEY FEATURE)
# =============================================================================

def apply_escalation(
    predicted_priority: str, 
    sla_status: str
) -> EscalationResult:
    """
    Apply SLA-aware escalation rules to upgrade priority if needed.
    
    WHY: Priority is NOT static. When SLA is at risk, we proactively
    escalate to ensure the ticket gets more attention before breaching.
    
    Escalation Rules:
        - Only escalate if SLA status is "at_risk"
        - Medium → High
        - High → Critical
        - Low stays Low (minor issues don't need escalation)
        - Critical stays Critical (already highest)
    
    Args:
        predicted_priority: ML-predicted priority
        sla_status: Current SLA status
        
    Returns:
        EscalationResult with final priority and escalation details
    """
    original_priority = predicted_priority
    final_priority = predicted_priority
    was_escalated = False
    escalation_reason = None
    
    # Only escalate if SLA is at risk
    if sla_status == "at_risk":
        # Check if this priority can be escalated
        if predicted_priority in ESCALATION_MAP:
            final_priority = ESCALATION_MAP[predicted_priority]
            was_escalated = True
            escalation_reason = (
                f"Priority escalated from {original_priority} to {final_priority} "
                f"due to SLA at-risk status. Resolution time is approaching limit."
            )
    
    # Log breached SLAs but don't escalate (too late)
    elif sla_status == "breached":
        escalation_reason = (
            f"SLA already breached. Priority remains {original_priority}. "
            f"Immediate attention required."
        )
    
    return EscalationResult(
        original_priority=original_priority,
        final_priority=final_priority,
        was_escalated=was_escalated,
        escalation_reason=escalation_reason
    )


def process_business_rules(
    predicted_priority: str,
    time_to_resolution_hours: Optional[float],
    ticket_creation_time: Optional[datetime] = None
) -> Tuple[SLAResult, EscalationResult]:
    """
    Complete business rules processing pipeline.
    
    This is the main entry point for the business rules layer.
    It processes SLA and escalation in the correct order.
    
    Args:
        predicted_priority: Priority from ML model
        time_to_resolution_hours: Historical resolution time
        ticket_creation_time: When ticket was created
        
    Returns:
        Tuple of (SLAResult, EscalationResult)
    """
    # Step 1: Process SLA (assign hours, calculate status)
    sla_result = process_sla(
        priority=predicted_priority,
        time_to_resolution_hours=time_to_resolution_hours,
        ticket_creation_time=ticket_creation_time
    )
    
    # Step 2: Apply escalation rules based on SLA status
    escalation_result = apply_escalation(
        predicted_priority=predicted_priority,
        sla_status=sla_result.sla_status
    )
    
    return sla_result, escalation_result


if __name__ == "__main__":
    # Quick test of business rules
    print("Testing Business Rules Module\n")
    
    # Test Case 1: Medium priority, at risk
    print("Test 1: Medium priority, at risk (40 hours, limit 48)")
    sla, escalation = process_business_rules("Medium", 40.0)
    print(f"  SLA Hours: {sla.sla_hours}")
    print(f"  SLA Status: {sla.sla_status}")
    print(f"  Final Priority: {escalation.final_priority}")
    print(f"  Was Escalated: {escalation.was_escalated}")
    print(f"  Reason: {escalation.escalation_reason}\n")
    
    # Test Case 2: High priority, met
    print("Test 2: High priority, met (10 hours, limit 24)")
    sla, escalation = process_business_rules("High", 10.0)
    print(f"  SLA Hours: {sla.sla_hours}")
    print(f"  SLA Status: {sla.sla_status}")
    print(f"  Final Priority: {escalation.final_priority}")
    print(f"  Was Escalated: {escalation.was_escalated}")
