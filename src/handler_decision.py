"""
AssistFlow AI - Handler Decision Module

WORKFLOW STEP 8: AI VS HUMAN DECISION (RISK CONTROL)

This module determines whether a ticket can be handled by AI assistance
or requires human intervention.

CRITICAL: This uses RULES, NOT ML.
The LLM must NEVER make this decision.

Rules:
- If final_priority is High or Critical → Human
- If issue_type is Billing or Security → Human
- Otherwise → AI allowed
"""

from typing import Optional
from dataclasses import dataclass

import sys
sys.path.append(".")
from config import (
    HUMAN_REQUIRED_PRIORITIES,
    HUMAN_REQUIRED_ISSUE_TYPES
)


@dataclass
class HandlerDecision:
    """
    Container for handler assignment decision.
    
    Provides clear documentation of why a handler type was chosen.
    """
    handler_type: str  # "Human" or "AI"
    reason: str
    priority_triggered: bool
    issue_type_triggered: bool


def determine_handler(
    final_priority: str,
    issue_type: Optional[str] = None
) -> HandlerDecision:
    """
    Determine whether AI assistance is allowed or human is required.
    
    WHY RULE-BASED: This is a RISK CONTROL decision. We cannot let
    ML models make decisions about escalation to humans because:
    - High-priority tickets need guaranteed human attention
    - Sensitive issue types (Billing, Security) require human judgment
    - Regulatory compliance may require human oversight
    
    Decision Logic:
    1. Check priority first (High/Critical → Human)
    2. Check issue type second (Billing/Security → Human)
    3. If neither triggered → AI allowed
    
    Args:
        final_priority: Priority after any escalations applied
        issue_type: Predicted issue type (optional)
        
    Returns:
        HandlerDecision with handler type and reasoning
    """
    priority_triggered = False
    issue_type_triggered = False
    reasons = []
    
    # Rule 1: Check priority
    if final_priority in HUMAN_REQUIRED_PRIORITIES:
        priority_triggered = True
        reasons.append(
            f"Priority '{final_priority}' requires human handling"
        )
    
    # Rule 2: Check issue type (if provided)
    if issue_type and issue_type in HUMAN_REQUIRED_ISSUE_TYPES:
        issue_type_triggered = True
        reasons.append(
            f"Issue type '{issue_type}' requires human handling"
        )
    
    # Determine final handler type
    if priority_triggered or issue_type_triggered:
        handler_type = "Human"
        reason = ". ".join(reasons) + "."
    else:
        handler_type = "AI"
        reason = (
            f"AI assistance allowed. Priority '{final_priority}' and "
            f"issue type '{issue_type or 'Unknown'}' do not require human intervention."
        )
    
    return HandlerDecision(
        handler_type=handler_type,
        reason=reason,
        priority_triggered=priority_triggered,
        issue_type_triggered=issue_type_triggered
    )


def is_human_required(
    final_priority: str,
    issue_type: Optional[str] = None
) -> bool:
    """
    Simple boolean check if human is required.
    
    Use this for quick checks without needing full decision details.
    
    Args:
        final_priority: Priority after any escalations
        issue_type: Predicted issue type (optional)
        
    Returns:
        True if human required, False if AI allowed
    """
    decision = determine_handler(final_priority, issue_type)
    return decision.handler_type == "Human"


def get_handler_recommendation(
    final_priority: str,
    issue_type: Optional[str] = None,
    sla_status: Optional[str] = None
) -> str:
    """
    Get a human-readable recommendation for ticket handling.
    
    WHY: This provides context for support managers and agents
    about why a ticket was routed a certain way.
    
    Args:
        final_priority: Priority after escalations
        issue_type: Predicted issue type
        sla_status: Current SLA status
        
    Returns:
        Recommendation string for display
    """
    decision = determine_handler(final_priority, issue_type)
    
    recommendation = f"RECOMMENDED HANDLER: {decision.handler_type}\n"
    recommendation += f"Reason: {decision.reason}\n"
    
    # Add SLA context if provided
    if sla_status:
        if sla_status == "breached":
            recommendation += "\n⚠️ WARNING: SLA has been breached. Immediate action required."
        elif sla_status == "at_risk":
            recommendation += "\n⚡ ALERT: SLA is at risk. Prioritize this ticket."
    
    # Add specific guidance based on handler type
    if decision.handler_type == "Human":
        recommendation += "\n\nGuidance for Human Agent:"
        if decision.priority_triggered:
            recommendation += "\n- This is a high-priority ticket requiring immediate attention"
        if decision.issue_type_triggered:
            recommendation += f"\n- {issue_type} issues require careful human review"
    else:
        recommendation += "\n\nGuidance for AI Assistance:"
        recommendation += "\n- AI can draft initial response"
        recommendation += "\n- AI can summarize ticket content"
        recommendation += "\n- Human review still recommended before sending"
    
    return recommendation


if __name__ == "__main__":
    # Test handler decision logic
    print("Testing Handler Decision Module\n")
    
    # Test Case 1: High priority
    print("Test 1: High priority, Technical issue")
    decision = determine_handler("High", "Technical issue")
    print(f"  Handler: {decision.handler_type}")
    print(f"  Reason: {decision.reason}\n")
    
    # Test Case 2: Medium priority, Billing
    print("Test 2: Medium priority, Billing issue")
    decision = determine_handler("Medium", "Billing")
    print(f"  Handler: {decision.handler_type}")
    print(f"  Reason: {decision.reason}\n")
    
    # Test Case 3: Low priority, General
    print("Test 3: Low priority, General issue")
    decision = determine_handler("Low", "General")
    print(f"  Handler: {decision.handler_type}")
    print(f"  Reason: {decision.reason}\n")
    
    # Test full recommendation
    print("=" * 50)
    print("Full Recommendation Example:\n")
    print(get_handler_recommendation("Critical", "Security", "at_risk"))
