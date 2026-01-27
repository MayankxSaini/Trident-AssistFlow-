# 🚀 AssistFlow AI
### SLA-Aware Intelligent Customer Support Co-Pilot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Google Gemini](https://img.shields.io/badge/Google-Gemini_LLM-4285F4?logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow)

**🎯 Transforming Customer Support Operations with AI-Driven Intelligence**

[Live Demo](https://trident-assistflow-whha.onrender.com/) • [API Docs](https://trident-assistflow.onrender.com/docs) • [Problem Statement](#-the-problem-flowbridge-technologies)

</div>

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [The Problem: FlowBridge Technologies](#-the-problem-flowbridge-technologies)
- [Our Solution: AssistFlow AI](#-our-solution-assistflow-ai)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technical Implementation](#-technical-implementation)
- [AI/ML Pipeline](#-aiml-pipeline)
- [Business Rules Engine](#-business-rules-engine)
- [LLM Integration](#-llm-integration)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [ROI & Business Impact](#-roi--business-impact)
- [Demo & Screenshots](#-demo--screenshots)
- [Team](#-team)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 Executive Summary

**AssistFlow AI** is an intelligent customer support co-pilot designed to solve the escalating ticket management crisis faced by high-growth SaaS companies. Unlike traditional keyword-based triage systems, AssistFlow AI leverages **Machine Learning** for intelligent classification and **Large Language Models (LLMs)** for human-like assistance—while maintaining **strict SLA compliance** through deterministic business rules.

### Key Metrics Improvement Targets

| Metric | Before | After AssistFlow AI | Improvement |
|--------|--------|---------------------|-------------|
| **First Response Time** | 4 hours | 30 minutes | **87.5% ↓** |
| **Resolution Time** | 48 hours | 24 hours | **50% ↓** |
| **First Contact Resolution** | 60% | 80% | **33% ↑** |
| **CSAT Score** | 3.8 | 4.2+ | **10.5% ↑** |
| **Agent Productivity** | Baseline | +15% | **15% ↑** |
| **Annual Cost Savings** | — | $500,000+ | **Significant** |

---

## 🔥 The Problem: FlowBridge Technologies

### Company Background

**FlowBridge Technologies** is a global SaaS company providing collaboration and workflow tools to large enterprises. With **300+ support agents worldwide** and **7,000+ daily tickets** from email, chat, and web forms, their customer support operations had reached a critical breaking point.

### The Crisis (Wednesday Morning Dashboard)

```
📊 SUPPORT DASHBOARD - CRITICAL ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Ticket Backlog:     +30% (3 months)
📬 Open Tickets:       2,000+ daily
⏱️ First Response:     4 hours (Target: 30 min)
🔄 Resolution Time:    48 hours (Target: 24 hours)
📉 CSAT Score:         3.8 → Down from 4.2
🎯 FCR Rate:           60% (Target: 80%)
💰 Annual Cost Impact: $800,000+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Root Cause Analysis

| Problem | Impact |
|---------|--------|
| **Manual Triage Bottleneck** | Agents spend excessive time understanding each ticket before action |
| **Keyword-Based Routing Failure** | Cannot understand context, urgency, or semantic meaning |
| **Misrouted Tickets** | 40% of tickets reach wrong specialists first |
| **Generic Initial Responses** | Customers frustrated by templated replies |
| **No Proactive SLA Management** | Breaches discovered only after they occur |
| **Priority Misjudgment** | Static systems miss true urgency signals |

### Stakeholder Concerns

> *"Agents spend too much time figuring out the real issue in each ticket. First response time has increased to 4 hours. Many customers get generic answers before reaching the right expert."*  
> — **Emily Tan**, VP of Customer Experience

> *"Our keyword-based triage tool cannot understand the meaning or urgency of customer messages. This leads to many tickets being misrouted or incorrectly prioritized."*  
> — **Arjun Mehta**, Head of Analytics

---

## 💡 Our Solution: AssistFlow AI

### Vision: AI as a Co-Pilot, Not a Replacement

AssistFlow AI is designed as an **intelligent assistant** that augments human agents rather than replacing them. It handles the cognitive load of understanding, prioritizing, and routing tickets—freeing agents to focus on resolution.

### Core Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASSISTFLOW AI PRINCIPLES                     │
├─────────────────────────────────────────────────────────────────┤
│  ✅ AI ASSISTS decisions      │  ❌ AI MAKES final decisions   │
│  ✅ Deterministic SLA rules   │  ❌ ML-based SLA calculations  │
│  ✅ Explainable predictions   │  ❌ Black-box neural networks  │
│  ✅ Human oversight always    │  ❌ Fully autonomous handling  │
│  ✅ Transparent reasoning     │  ❌ Hidden decision logic      │
└─────────────────────────────────────────────────────────────────┘
```

### What AssistFlow AI Does

1. **🔍 Intelligent Understanding** — Reads and comprehends ticket content using NLP
2. **🎯 Smart Prioritization** — ML-based priority prediction (Critical/High/Medium/Low)
3. **📂 Issue Classification** — Automatic categorization (Billing/Technical/Refund/etc.)
4. **⏰ SLA Compliance Engine** — Real-time monitoring with proactive alerts
5. **🤖↔️👤 Smart Routing** — Decides AI-handleable vs. human-required tickets
6. **💬 Response Generation** — LLM-powered draft responses for agent approval
7. **📊 Explainable AI** — Clear reasoning for every decision made

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ASSISTFLOW AI ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌─────────────────────────────────────────────┐     │
│   │   TICKETS   │───▶│              INGESTION LAYER               │     │
│   │  (CSV/API)  │    │   • Data Loading & Validation               │     │
│   └─────────────┘    │   • Text Preprocessing & Cleaning           │     │
│                      └─────────────────┬───────────────────────────┘     │
│                                        │                                 │
│                                        ▼                                 │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │                    ML PREDICTION LAYER                          │    │
│   │  ┌─────────────────┐    ┌─────────────────┐                    │     │
│   │  │  Model 1:       │    │  Model 2:       │                    │     │
│   │  │  Priority       │    │  Issue Type     │                    │     │
│   │  │  Classifier     │    │  Classifier     │                    │     │
│   │  │  (TF-IDF + LR)  │    │  (TF-IDF + LR)  │                    │     │
│   │  └────────┬────────┘    └────────┬────────┘                    │     │
│   │           │                      │                              │    │
│   │           └──────────┬───────────┘                              │    │
│   └──────────────────────┼──────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │               BUSINESS RULES ENGINE (NO ML)                    │     │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │     │
│   │  │ SLA Hours    │  │ SLA Status   │  │ Escalation   │          │     │
│   │  │ Assignment   │  │ Calculator   │  │ Rules        │          │     │
│   │  │              │  │              │  │              │          │     │
│   │  │ Critical: 6h │  │ MET          │  │ AT_RISK →    │          │     │
│   │  │ High: 24h    │  │ AT_RISK      │  │ Escalate     │          │     │
│   │  │ Medium: 48h  │  │ BREACHED     │  │ Priority     │          │     │
│   │  │ Low: 72h     │  │              │  │              │          │     │
│   │  └──────────────┘  └──────────────┘  └──────────────┘          │     │
│   └──────────────────────┬──────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │              HANDLER DECISION LAYER (RULE-BASED)               │     │
│   │                                                                │     │
│   │   Priority = Critical/High  ──────────▶  👤 HUMAN QUEUE       │     │
│   │   Issue = Billing/Security  ──────────▶  👤 HUMAN QUEUE       │     │
│   │   Otherwise                 ──────────▶  🤖 AI HANDLING       │     │
│   │                                                                │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │                 LLM ASSISTANCE LAYER                           │     │
│   │              (AFTER ALL DECISIONS FINAL)                       │     │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │     │
│   │  │ Ticket       │  │ Decision     │  │ Response     │          │     │
│   │  │ Summary      │  │ Explanation  │  │ Draft        │          │     │
│   │  └──────────────┘  └──────────────┘  └──────────────┘          │     │
│   │                                                                │     │
│   │  🔒 LLM CANNOT modify priority, SLA, or handler decisions     │      │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │                    OUTPUT LAYER                                │     │
│   │  • Complete ticket analysis result                             │     │
│   │  • Real-time dashboard updates                                 │     │
│   │  • API response for integrations                               │     │
│   └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | Real-time Operations Dashboard |
| **Backend API** | FastAPI | REST API for Integrations |
| **ML Models** | scikit-learn | TF-IDF + Logistic Regression |
| **LLM Provider** | Google Gemini | Response Generation & Summarization |
| **Data Processing** | Pandas, NumPy | Data Manipulation |
| **Deployment** | Render | Cloud Hosting |

---

## ✨ Key Features

### 📥 Unified Ticket Inbox
- Real-time view of all unresolved tickets
- Color-coded priority indicators
- SLA status badges (MET ✅ / AT_RISK ⚠️ / BREACHED 🔴)
- Quick filters by priority, SLA status, and issue type

### 🤖 AI Handling Queue
- Tickets suitable for AI-assisted resolution
- AI-generated response drafts ready for review
- One-click approval or escalation
- Confidence scores for predictions

### 👤 Human Agent Queue
- Critical and sensitive tickets requiring human judgment
- Clear escalation reasons displayed
- Sorted by urgency and SLA deadline
- Context-rich ticket summaries

### 📊 Operations Dashboard
- Real-time processing metrics
- AI vs Human distribution charts
- SLA compliance rates
- Escalation trend analysis
- Priority distribution visualization

### 🔍 Deep Ticket Analysis
- Detailed view of individual tickets
- Complete decision explanation
- Historical context
- Suggested response with edit capability

---

## 🤖 AI/ML Pipeline

### Model Architecture

We deliberately chose **simple, explainable models** over complex deep learning:

```python
# Why TF-IDF + Logistic Regression?
# ✅ Interpretable - Can explain via feature weights
# ✅ Fast inference - Milliseconds per prediction  
# ✅ Low resource - No GPU required
# ✅ Reliable - Well-understood behavior
# ❌ No black box neural networks
```

### Model 1: Priority Classifier

| Specification | Value |
|--------------|-------|
| **Input** | Ticket Subject + Description (combined text) |
| **Output** | Priority ∈ {Low, Medium, High, Critical} |
| **Algorithm** | TF-IDF Vectorizer + Logistic Regression |
| **Features** | 5,000 max features, (1,2)-gram range |
| **Confidence** | Probability scores for explainability |

### Model 2: Issue Type Classifier

| Specification | Value |
|--------------|-------|
| **Input** | Ticket Subject + Description (combined text) |
| **Output** | Issue Type ∈ {Billing, Technical, Refund, Product, Access, General} |
| **Algorithm** | TF-IDF Vectorizer + Logistic Regression |
| **Features** | 5,000 max features, (1,2)-gram range |
| **Confidence** | Probability scores for routing decisions |

### Prediction Pipeline

```
                    ┌─────────────────┐
                    │   Raw Ticket    │
                    │  Subject + Desc │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Cleaning   │
                    │ • Lowercase     │
                    │ • Remove noise  │
                    │ • Normalize     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ TF-IDF Vector   │           │ TF-IDF Vector   │
    │ (Priority)      │           │ (Issue Type)    │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ LogReg Model    │           │ LogReg Model    │
    │ → Priority      │           │ → Issue Type    │
    │ → Confidence    │           │ → Confidence    │
    └─────────────────┘           └─────────────────┘
```

---

## ⚖️ Business Rules Engine

### Why Rules Over ML for Critical Decisions?

```
┌───────────────────────────────────────────────────────────────────┐
│  CRITICAL INSIGHT: SLA and Escalation are BUSINESS decisions,     │
│  not predictions. They must be deterministic and auditable.       │
│                                                                   │
│  ❌ ML models can be wrong → unacceptable for SLA compliance     │
│  ✅ Rules are predictable → guaranteed policy enforcement        │
└───────────────────────────────────────────────────────────────────┘
```

### SLA Assignment Rules

```python
SLA_HOURS = {
    "Critical": 6,    # Must resolve within 6 hours
    "High":     24,   # Must resolve within 24 hours
    "Medium":   48,   # Must resolve within 48 hours
    "Low":      72    # Must resolve within 72 hours
}
```

### SLA Status Calculation

```python
def calculate_sla_status(time_elapsed, sla_hours):
    if time_elapsed > sla_hours:
        return "BREACHED"     # 🔴 SLA violated
    elif time_elapsed > sla_hours * 0.80:
        return "AT_RISK"      # ⚠️ 80% of SLA consumed
    else:
        return "MET"          # ✅ Within SLA
```

### Escalation Rules

| Condition | Action |
|-----------|--------|
| SLA Status = BREACHED | Escalate to next priority level |
| SLA Status = AT_RISK + High Volume | Escalate to next priority level |
| Issue Type = Security | Always route to Human |
| Issue Type = Billing | Always route to Human |

### Handler Decision Rules

```python
# RULE-BASED (NOT ML) - This is a risk control decision
def determine_handler(priority, issue_type):
    if priority in ["Critical", "High"]:
        return "Human"  # High-risk tickets need human oversight
    
    if issue_type in ["Billing", "Security"]:
        return "Human"  # Sensitive issues need human judgment
    
    return "AI"  # Low-risk tickets can be AI-assisted
```

---

## 🧠 LLM Integration

### Role of LLM in AssistFlow AI

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM USAGE BOUNDARIES                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ✅ LLM CAN:                    │  ❌ LLM CANNOT:                 │
│  • Summarize ticket content     │  • Change priority               │
│  • Explain decisions            │  • Modify SLA hours              │
│  • Draft response messages      │  • Override handler decision     │
│  • Provide context              │  • Bypass escalation rules       │
│                                                                    │
│  LLM is invoked ONLY AFTER all decisions are finalized             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Supported LLM Providers

| Provider | Model | Use Case |
|----------|-------|----------|
| **Google Gemini** | gemini-1.5-flash | Primary (Fast, Cost-effective) |
| **OpenAI** | gpt-3.5-turbo | Alternative |
| **Ollama** | llama2 | Local/Offline |
| **Template** | N/A | Fallback (No API) |

### LLM Outputs

1. **Ticket Summary** — 2-3 sentence overview for quick agent context
2. **Decision Explanation** — Why this priority/SLA/handler was assigned
3. **Suggested Response** — Draft reply ready for agent approval

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- pip package manager
- Google Gemini API Key (optional, for LLM features)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/MayankxSaini/Trident-AssistFlow-.git
cd Trident-AssistFlow-

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 6. Run the application
streamlit run app.py
```

### Environment Variables

```env
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
USE_LLM=true
```

### Running the API (Optional)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 📁 Project Structure

```
AssistFlow-AI/
├── 📄 app.py                    # Streamlit Dashboard (Main UI)
├── 📄 api.py                    # FastAPI REST API
├── 📄 config.py                 # Centralized Configuration
├── 📄 train_models.py           # Model Training Script
├── 📄 demo.py                   # Demo/Testing Script
├── 📄 requirements.txt          # Python Dependencies
├── 📄 render.yaml               # Render Deployment Config
├── 📄 .env.example              # Environment Template
│
├── 📂 src/                      # Core Source Code
│   ├── 📄 __init__.py
│   ├── 📄 pipeline.py           # Main Orchestration Pipeline
│   ├── 📄 ingestion.py          # Data Loading & Preprocessing
│   ├── 📄 models.py             # ML Model Classes
│   ├── 📄 business_rules.py     # SLA & Escalation Rules
│   ├── 📄 handler_decision.py   # AI vs Human Routing
│   ├── 📄 llm_assistance.py     # LLM Integration Layer
│   └── 📄 ticket_state.py       # Ticket State Management
│
├── 📂 models/                   # Trained ML Models
│   ├── 📄 priority_model.pkl
│   ├── 📄 priority_vectorizer.pkl
│   ├── 📄 issue_type_model.pkl
│   └── 📄 issue_type_vectorizer.pkl
│
└── 📂 data/                     # Dataset
    └── 📄 customer_support_tickets.csv
```

---

## 💰 ROI & Business Impact

### CFO Question: "Will this help us increase CSAT by 0.2 points and reduce support costs by $500,000?"

### Answer: Yes. Here's the projected impact:

| Improvement Area | Current State | With AssistFlow AI | Impact |
|-----------------|---------------|-------------------|--------|
| **Ticket Triage Time** | 15 min/ticket | 30 sec/ticket | **97% reduction** |
| **First Response Time** | 4 hours | 30 minutes | **87.5% faster** |
| **Resolution Time** | 48 hours | 24 hours | **50% faster** |
| **FCR Rate** | 60% | 80% | **33% improvement** |
| **CSAT Score** | 3.8 | 4.0+ | **0.2+ point increase** |
| **Misrouted Tickets** | 40% | <10% | **75% reduction** |
| **Agent Productivity** | Baseline | +15% | **15% more tickets/agent** |

### Cost Savings Breakdown

```
┌────────────────────────────────────────────────────────────┐
│                 ANNUAL COST SAVINGS                        │
├────────────────────────────────────────────────────────────┤
│ Reduced handling time (15min → 30sec)     │  $250,000      │
│ Fewer escalations & rework                │  $150,000      │
│ Lower customer churn (better CSAT)        │  $200,000      │
│ Agent capacity increase (15%)             │  $100,000      │
├────────────────────────────────────────────────────────────┤
│ TOTAL ESTIMATED SAVINGS                   │  $700,000+     │
└────────────────────────────────────────────────────────────┘
```

---

## 🖼️ Demo & Screenshots

### 📥 Ticket Inbox View
Real-time view of all unresolved tickets with priority badges and SLA status indicators.

### 🤖 AI Handling Queue
Tickets that can be handled by AI with generated response drafts.

### 👤 Human Agent Queue  
Critical tickets requiring human intervention with clear escalation reasons.

### 📊 Analytics Dashboard
Comprehensive metrics showing ticket distribution, SLA compliance, and team performance.

---

## 👨‍💻 Team

<div align="center">

| Name | Role |
|------|------|
| **Mayank Saini** | Backend Developer & LLM integrator |
| **Saurabh** | ML Engineer |
| **Ritik Tanwar** | Frontend Developer & UI Designer |

**Team Name: Trident**

</div>

---

## 🔮 Future Roadmap

### Phase 2: Enhanced Intelligence
- [ ] Real-time ticket ingestion via webhooks
- [ ] Advanced sentiment analysis
- [ ] Customer emotion detection
- [ ] Predictive SLA breach alerts

### Phase 3: Enterprise Features  
- [ ] Multi-tenant support
- [ ] Custom SLA rule builder
- [ ] Agent performance analytics
- [ ] A/B testing for LLM models

### Phase 4: Integrations
- [ ] Zendesk integration
- [ ] Freshdesk integration
- [ ] Slack notifications
- [ ] Email automation

### Phase 5: Advanced AI
- [ ] RAG (Retrieval-Augmented Generation) for knowledge base
- [ ] Data drift detection
- [ ] Model retraining automation
- [ ] Custom fine-tuned models

---

## 📌 Conclusion

**AssistFlow AI** demonstrates how intelligent AI systems can transform customer support operations without replacing human judgment. By combining:

- ✅ **ML-powered understanding** for ticket classification
- ✅ **Deterministic business rules** for SLA compliance
- ✅ **LLM assistance** for agent productivity
- ✅ **Real-time dashboards** for operational visibility

We deliver a solution that is **accurate, explainable, and production-ready**.

---

<div align="center">

### 🏆 Built for FlowBridge Technologies Challenge

**AssistFlow AI** — *Intelligent Support, Human Trust*

Made with ❤️ by **Team Trident**

</div>

