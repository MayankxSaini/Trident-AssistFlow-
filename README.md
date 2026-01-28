# AssistFlow AI

## SLA-Aware Intelligent Customer Support System

---

## 1. Problem Statement

Customer support platforms typically process incoming tickets in a **first-come, first-served** manner or rely on static, manually assigned priorities. This leads to multiple operational challenges:

* Critical issues (financial loss, access failures, security risks) get delayed
* Low-impact or repetitive queries consume human agent time
* SLA breaches occur silently until it is too late
* Support teams operate reactively instead of proactively

Keyword-based or rule-only systems lack semantic understanding and fail to capture **intent, urgency, and business risk** from unstructured customer messages.

---

## 2. Solution Overview – AssistFlow AI

AssistFlow AI introduces an **intelligent ML-driven decision layer** before tickets reach support agents. The system augments existing support workflows by:

* Automatically understanding ticket content
* Assigning and continuously updating priority based on risk and SLA
* Deciding AI vs Human handling safely
* Assisting agents with context, explanations, and response drafts

The goal is **optimization, not automation-forcing** — humans remain in control of all final decisions.

---

## 3. High-Level Architecture

```
Customer Ticket
      │
      ▼
Text Ingestion & Cleaning
      │
      ▼
ML Classification Layer
(Issue Type + Priority)
      │
      ▼
Business Rules Engine
(Priority Mapping + SLA Assignment)
      │
      ▼
SLA Monitoring & Dynamic Escalation
      │
      ▼
Handler Decision Engine
(AI vs Human)
      │
      ▼
LLM Assistance Layer
(Summary, Explanation, Draft Response)
```

---

## 4. Machine Learning Pipeline (Detailed)

This section describes the **end-to-end ML pipeline** used in AssistFlow AI, from raw ticket text to structured predictions. The pipeline is intentionally designed to be **simple, explainable, and production-aligned**, rather than over-engineered.

### 4.1 Data Ingestion

* Source: `customer_support_tickets.csv`
* Raw fields used:

  * Ticket Subject
  * Ticket Description

Both fields are concatenated to form a single text input, ensuring that short subjects are enriched with detailed descriptions.

Basic preprocessing:

* Lowercasing
* Removal of special characters
* Handling null or empty text safely

No aggressive text cleaning is applied to preserve intent-bearing words.

---

### 4.2 Feature Engineering

* **TF-IDF Vectorization** is used to convert raw text into numerical features.

Configuration highlights:

* N-grams to capture short phrases (e.g., "unable to login")
* Max feature limits to control dimensionality

Why TF-IDF?

* Proven performance on support-ticket style text
* Lightweight and fast during inference
* Transparent and debuggable compared to embeddings

---

### 4.3 Model Training Strategy

Two independent supervised models are trained:

1. **Issue Type Classifier**
   Predicts the category of the issue (Technical, Billing, Account, etc.)

2. **Priority Prediction Model**
   Predicts the initial urgency level (Low, Medium, High, Critical)

Model choices:

* Logistic Regression / Linear SVM

Rationale:

* Strong baselines for text classification
* Low latency and memory footprint
* Easier calibration and confidence estimation

---

### 4.4 Evaluation & Validation

Models are evaluated using:

* Accuracy
* Class-wise precision and recall

Train vs test performance is explicitly logged to detect overfitting.

The system treats ML outputs as **signals**, not final truth.

---

### 4.5 Model Persistence & Loading

* Trained models and vectorizers are serialized
* Stored under `/models/`
* Loaded once at application startup for efficiency

This design supports:

* Easy retraining
* Model versioning
* Safe deployment without pipeline changes

---

### 4.2 Feature Engineering

* **TF-IDF Vectorization**

  * Captures term importance across tickets
  * Handles sparse, high-dimensional text efficiently

Reasoning:

* Lightweight and fast
* Works well for short-to-medium support messages
* Interpretable feature contributions

---

### 4.3 Model Training

Two supervised classification models are trained:

1. **Issue Type Classifier**
2. **Priority Predictor**

Model choices:

* Logistic Regression / Linear SVM (configurable)

Why linear models?

* Fast training and inference
* Stable performance on text classification
* Easier explainability compared to deep models

Evaluation Metrics:

* Accuracy
* Class-wise precision and recall

Models are serialized and stored under `/models/`.

---

## 5. Business Logic Layer (Non-ML)

ML predictions alone are **not trusted blindly**.

### 5.1 Priority Mapping Rules

Predicted priority is adjusted using deterministic rules:

* Issue type severity
* Keywords indicating financial loss or access blockage
* Customer impact assumptions

This ensures:

* Safety
* Predictable behavior
* Business-aligned decisions

---

### 5.2 SLA Assignment

Each priority maps to an SLA window:

| Priority | SLA (Hours) |
| -------- | ----------- |
| Low      | 72          |
| Medium   | 48          |
| High     | 24          |
| Critical | 6           |

SLA exists for **monitoring and accountability**, not pressure.

---

## 6. SLA-Aware Dynamic Escalation (Core Innovation)

Unlike static systems, AssistFlow AI **re-evaluates tickets over time**.

### SLA Status Computation

Based on:

* Time since ticket creation
* Assigned SLA window

Status values:

* `met`
* `at_risk`
* `breached`

### Escalation Logic

If a ticket is `at_risk` or `breached`:

* Priority is escalated automatically
* Ticket is surfaced higher in queues

This prevents silent SLA failures and shifts the system from **reactive to proactive**.

---

## 7. Handler Decision Engine (AI vs Human)

Routing logic combines:

* Final priority
* Risk level
* Confidence of ML predictions

Rules:

* Low-risk + repetitive → AI-assisted
* High-risk / sensitive → Human-only

Output:

```
handler_type = AI_ASSISTED | HUMAN
```

AI never resolves high-risk tickets autonomously.

---

## 8. LLM Assistance Layer

Large Language Models are used **only for augmentation**, not decision-making.

Capabilities:

* Ticket summarization
* Explanation of priority & escalation
* Draft response suggestions

All outputs are:

* Read-only suggestions
* Reviewed by human agents

---

## 9. Backend & Orchestration

### Backend

* **FastAPI** for APIs and pipeline orchestration
* Modular services for ingestion, ML inference, business rules

### Frontend

* **Streamlit** dashboard

  * Ticket list
  * Priority & SLA indicators
  * AI explanations

### Storage

* SQLite / PostgreSQL
* Stores:

  * Ticket metadata
  * SLA states
  * Resolution outcomes

---

## 10. End-to-End Ticket Flow

1. Ticket submitted by customer
2. Text analyzed by ML models
3. Priority & issue type predicted
4. Business rules applied
5. SLA assigned
6. SLA continuously monitored
7. Priority escalated if required
8. Routed to AI or Human
9. Agent receives AI assistance

---

## 11. Project Structure

```
JAI GANESH/
├── config.py
├── train_models.py
├── demo.py
├── requirements.txt
├── data/
│   └── customer_support_tickets.csv
├── models/
└── src/
    ├── ingestion.py
    ├── models.py
    ├── business_rules.py
    ├── handler_decision.py
    ├── llm_assistance.py
    └── pipeline.py
```

---

## 12. Setup & Execution

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python train_models.py
python demo.py
```

---

## 13. Output Schema

Each ticket produces:

* `issue_type`
* `predicted_priority`
* `sla_hours`
* `sla_status`
* `final_priority`
* `handler_type`
* `explanation_text`
* `suggested_response`

---

## 14. Team

* Mayank Saini
* Saurabh (ML Engineering & Pipeline Design)
* Ritik Tanwar
* **Hsns (ML Pipeline Development & System Integration)**

---

## 15. Summary

AssistFlow AI demonstrates how **classical ML + rule-based systems + LLMs** can be combined to build a production-realistic, SLA-aware support intelligence layer. The system is explainable, safe, scalable, and aligned with real-world customer support operations.
