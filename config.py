"""Configuration settings for AssistFlow AI"""

DATA_PATH = "./data/customer_support_tickets.csv"
PRIORITY_MODEL_PATH = "./models/priority_model.pkl"
PRIORITY_VECTORIZER_PATH = "./models/priority_vectorizer.pkl"
ISSUE_TYPE_MODEL_PATH = "./models/issue_type_model.pkl"
ISSUE_TYPE_VECTORIZER_PATH = "./models/issue_type_vectorizer.pkl"

COL_TICKET_ID = "Ticket ID"
COL_SUBJECT = "Ticket Subject"
COL_DESCRIPTION = "Ticket Description"
COL_PRIORITY = "Ticket Priority"
COL_TICKET_TYPE = "Ticket Type"
COL_TIME_TO_RESOLUTION = "Time to Resolution"
COL_FIRST_RESPONSE_TIME = "First Response Time"
COL_SATISFACTION = "Customer Satisfaction Rating"

SLA_HOURS = {
    "Low": 72,
    "Medium": 48,
    "High": 24,
    "Critical": 6
}

SLA_AT_RISK_THRESHOLD = 0.80

ESCALATION_MAP = {
    "Medium": "High",
    "High": "Critical"
}

HUMAN_REQUIRED_PRIORITIES = ["High", "Critical"]
HUMAN_REQUIRED_ISSUE_TYPES = ["Billing", "Security"]

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_REGRESSION_MAX_ITER = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42

VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]
