"""Ticket state management"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass
from datetime import datetime


class TicketState(Enum):
    NEW = "New"
    IN_PROGRESS = "In Progress"
    AI_HANDLING = "AI Handling"
    WAITING_FOR_HUMAN = "Waiting for Human"
    RESOLVED = "Resolved"


@dataclass
class TicketStateTransition:
    from_state: TicketState
    to_state: TicketState
    timestamp: datetime
    reason: str
    actor: str


VALID_TRANSITIONS = {
    TicketState.NEW: [TicketState.IN_PROGRESS, TicketState.AI_HANDLING, TicketState.WAITING_FOR_HUMAN],
    TicketState.IN_PROGRESS: [TicketState.AI_HANDLING, TicketState.WAITING_FOR_HUMAN, TicketState.RESOLVED],
    TicketState.AI_HANDLING: [TicketState.WAITING_FOR_HUMAN, TicketState.RESOLVED],
    TicketState.WAITING_FOR_HUMAN: [TicketState.IN_PROGRESS, TicketState.RESOLVED],
    TicketState.RESOLVED: []
}


def get_initial_state(handler_type: str) -> TicketState:
    return TicketState.AI_HANDLING if handler_type == "AI" else TicketState.WAITING_FOR_HUMAN


def can_transition(current_state: TicketState, target_state: TicketState) -> bool:
    return target_state in VALID_TRANSITIONS.get(current_state, [])


def get_state_color(state: TicketState) -> str:
    colors = {
        TicketState.NEW: "#3498db",
        TicketState.IN_PROGRESS: "#f39c12",
        TicketState.AI_HANDLING: "#2ecc71",
        TicketState.WAITING_FOR_HUMAN: "#e74c3c",
        TicketState.RESOLVED: "#95a5a6"
    }
    return colors.get(state, "#000000")


def get_state_icon(state: TicketState) -> str:
    icons = {
        TicketState.NEW: "🆕",
        TicketState.IN_PROGRESS: "🔄",
        TicketState.AI_HANDLING: "🤖",
        TicketState.WAITING_FOR_HUMAN: "👤",
        TicketState.RESOLVED: "✅"
    }
    return icons.get(state, "❓")
