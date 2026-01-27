"""
AssistFlow AI - Ticket State Management

Manages ticket states and transitions for the support system.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


class TicketState(Enum):
    """Ticket lifecycle states."""
    NEW = "New"
    IN_PROGRESS = "In Progress"
    AI_HANDLING = "AI Handling"
    WAITING_FOR_HUMAN = "Waiting for Human"
    RESOLVED = "Resolved"


@dataclass
class TicketStateTransition:
    """Records a state transition."""
    from_state: TicketState
    to_state: TicketState
    timestamp: datetime
    reason: str
    actor: str  # "System", "AI", or agent name


# Valid state transitions
VALID_TRANSITIONS = {
    TicketState.NEW: [
        TicketState.IN_PROGRESS,
        TicketState.AI_HANDLING,
        TicketState.WAITING_FOR_HUMAN
    ],
    TicketState.IN_PROGRESS: [
        TicketState.AI_HANDLING,
        TicketState.WAITING_FOR_HUMAN,
        TicketState.RESOLVED
    ],
    TicketState.AI_HANDLING: [
        TicketState.WAITING_FOR_HUMAN,  # Escalation
        TicketState.RESOLVED
    ],
    TicketState.WAITING_FOR_HUMAN: [
        TicketState.IN_PROGRESS,
        TicketState.RESOLVED
    ],
    TicketState.RESOLVED: []  # Terminal state
}


def get_initial_state(handler_type: str) -> TicketState:
    """
    Determine initial state based on handler type.
    
    Args:
        handler_type: "AI" or "Human"
        
    Returns:
        Initial ticket state
    """
    if handler_type == "AI":
        return TicketState.AI_HANDLING
    else:
        return TicketState.WAITING_FOR_HUMAN


def can_transition(current_state: TicketState, target_state: TicketState) -> bool:
    """Check if a state transition is valid."""
    return target_state in VALID_TRANSITIONS.get(current_state, [])


def get_state_color(state: TicketState) -> str:
    """Get display color for a state."""
    colors = {
        TicketState.NEW: "#3498db",           # Blue
        TicketState.IN_PROGRESS: "#f39c12",   # Orange
        TicketState.AI_HANDLING: "#2ecc71",   # Green
        TicketState.WAITING_FOR_HUMAN: "#e74c3c",  # Red
        TicketState.RESOLVED: "#95a5a6"       # Gray
    }
    return colors.get(state, "#000000")


def get_state_icon(state: TicketState) -> str:
    """Get emoji icon for a state."""
    icons = {
        TicketState.NEW: "🆕",
        TicketState.IN_PROGRESS: "🔄",
        TicketState.AI_HANDLING: "🤖",
        TicketState.WAITING_FOR_HUMAN: "👤",
        TicketState.RESOLVED: "✅"
    }
    return icons.get(state, "❓")
