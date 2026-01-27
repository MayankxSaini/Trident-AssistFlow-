"""Handler decision logic"""

from typing import Optional
from dataclasses import dataclass

import sys
sys.path.append(".")
from config import HUMAN_REQUIRED_PRIORITIES, HUMAN_REQUIRED_ISSUE_TYPES


@dataclass
class HandlerDecision:
    handler_type: str
    reason: str
    priority_triggered: bool
    issue_type_triggered: bool


def determine_handler(final_priority: str, issue_type: Optional[str] = None) -> HandlerDecision:
    priority_triggered = False
    issue_type_triggered = False
    reasons = []
    
    if final_priority in HUMAN_REQUIRED_PRIORITIES:
        priority_triggered = True
        reasons.append(f"Priority '{final_priority}' requires human handling")
    
    if issue_type and issue_type in HUMAN_REQUIRED_ISSUE_TYPES:
        issue_type_triggered = True
        reasons.append(f"Issue type '{issue_type}' requires human handling")
    
    if priority_triggered or issue_type_triggered:
        handler_type = "Human"
        reason = ". ".join(reasons) + "."
    else:
        handler_type = "AI"
        reason = f"AI allowed. Priority '{final_priority}' and issue type '{issue_type or 'Unknown'}' don't require human."
    
    return HandlerDecision(
        handler_type=handler_type,
        reason=reason,
        priority_triggered=priority_triggered,
        issue_type_triggered=issue_type_triggered
    )


def is_human_required(final_priority: str, issue_type: Optional[str] = None) -> bool:
    decision = determine_handler(final_priority, issue_type)
    return decision.handler_type == "Human"


def get_handler_recommendation(final_priority: str, issue_type: Optional[str] = None,
                                sla_status: Optional[str] = None) -> str:
    decision = determine_handler(final_priority, issue_type)
    
    recommendation = f"HANDLER: {decision.handler_type}\nReason: {decision.reason}\n"
    
    if sla_status:
        if sla_status == "breached":
            recommendation += "\n⚠️ SLA breached. Immediate action required."
        elif sla_status == "at_risk":
            recommendation += "\n⚡ SLA at risk. Prioritize this ticket."
    
    if decision.handler_type == "Human":
        recommendation += "\n\nHuman guidance:"
        if decision.priority_triggered:
            recommendation += "\n- High-priority, needs immediate attention"
        if decision.issue_type_triggered:
            recommendation += f"\n- {issue_type} requires careful review"
    else:
        recommendation += "\n\nAI guidance:"
        recommendation += "\n- Can draft initial response"
        recommendation += "\n- Human review recommended before sending"
    
    return recommendation


if __name__ == "__main__":
    print("Testing handler decision\n")
    
    print("Test 1: High priority, Technical")
    d = determine_handler("High", "Technical issue")
    print(f"  Handler: {d.handler_type}")
    print(f"  Reason: {d.reason}")
    
    print("\nTest 2: Medium priority, Billing")
    d = determine_handler("Medium", "Billing")
    print(f"  Handler: {d.handler_type}")
    print(f"  Reason: {d.reason}")
    
    print("\nTest 3: Low priority, General")
    d = determine_handler("Low", "General")
    print(f"  Handler: {d.handler_type}")
    print(f"  Reason: {d.reason}")
