"""
AssistFlow AI - Configuration Settings

This file contains all configurable constants used throughout the system.
Centralizing configuration makes it easy to adjust settings without
modifying business logic.
"""

# =============================================================================
# FILE PATHS
# =============================================================================
DATA_PATH = "./data/customer_support_tickets.csv"
PRIORITY_MODEL_PATH = "./models/priority_model.pkl"
PRIORITY_VECTORIZER_PATH = "./models/priority_vectorizer.pkl"
ISSUE_TYPE_MODEL_PATH = "./models/issue_type_model.pkl"
ISSUE_TYPE_VECTORIZER_PATH = "./models/issue_type_vectorizer.pkl"

# =============================================================================
# DATASET COLUMN NAMES
# =============================================================================
COL_TICKET_ID = "Ticket ID"
COL_SUBJECT = "Ticket Subject"
COL_DESCRIPTION = "Ticket Description"
COL_PRIORITY = "Ticket Priority"
COL_TICKET_TYPE = "Ticket Type"
COL_TIME_TO_RESOLUTION = "Time to Resolution"
COL_FIRST_RESPONSE_TIME = "First Response Time"
COL_SATISFACTION = "Customer Satisfaction Rating"

# =============================================================================
# SLA CONFIGURATION (STATIC RULES)
# Maps priority levels to maximum allowed hours for resolution
# =============================================================================
SLA_HOURS = {
    "Low": 72,
    "Medium": 48,
    "High": 24,
    "Critical": 6
}

# =============================================================================
# SLA STATUS THRESHOLDS
# Percentage of SLA time consumed to determine status
# =============================================================================
# If resolution time > SLA hours → breached
# If resolution time > 80% of SLA hours → at_risk
# Otherwise → met
SLA_AT_RISK_THRESHOLD = 0.80

# =============================================================================
# ESCALATION RULES (PRIORITY UPGRADES)
# When SLA is at risk, priorities can be escalated
# =============================================================================
ESCALATION_MAP = {
    "Medium": "High",
    "High": "Critical"
}
# Note: "Low" does not escalate, "Critical" cannot escalate further

# =============================================================================
# HANDLER ASSIGNMENT RULES
# Determines whether AI can assist or human is required
# =============================================================================
HUMAN_REQUIRED_PRIORITIES = ["High", "Critical"]
HUMAN_REQUIRED_ISSUE_TYPES = ["Billing", "Security"]

# =============================================================================
# MODEL TRAINING PARAMETERS
# =============================================================================
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
LOGISTIC_REGRESSION_MAX_ITER = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# VALID PRIORITY LABELS
# Used for validation and filtering
# =============================================================================
VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]
