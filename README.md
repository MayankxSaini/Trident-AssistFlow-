# 🧠 AssistFlow AI  
### SLA-Aware Intelligent Customer Support System

AssistFlow AI is an **AI-powered customer support operations dashboard** designed to help support teams **prioritize, route, and resolve tickets efficiently** while maintaining **strict SLA compliance**.

Instead of treating all tickets equally, AssistFlow AI intelligently decides:
- Which tickets need **immediate human attention**
- Which tickets can be **handled by AI**
- Which tickets are **at risk of SLA breach**

---

## 🚨 Business Problem

Customer support teams face recurring challenges:

1. **High Ticket Volume**  
   Large numbers of repetitive and low-impact tickets overwhelm agents.

2. **Poor Prioritization**  
   Static priority systems fail to capture real urgency.

3. **SLA Breaches**  
   Delayed responses lead to customer dissatisfaction and penalties.

Traditional systems are reactive.  
**AssistFlow AI is proactive and SLA-aware.**

---

## 💡 Solution Overview

AssistFlow AI acts as an **intelligent decision layer** between incoming tickets and support agents.

### Core Capabilities
- AI-based ticket analysis
- Priority and issue-type prediction
- SLA risk monitoring
- Smart AI vs Human routing
- Automatic escalation

All insights are presented in a **real-time operations dashboard**.

---

## 🔄 End-to-End Workflow

1. **Ticket Ingestion**  
   Tickets are loaded from a dataset simulating real-world support systems.

2. **AI Analysis Pipeline**  
   - ML models predict ticket priority and issue category  
   - Business rules compute SLA hours  
   - SLA status is classified as `MET`, `AT_RISK`, or `BREACHED`

3. **Handler Assignment**  
   - Low-risk tickets → 🤖 AI Handling Queue  
   - High-risk / critical tickets → 👤 Human Queue

4. **Operational Views**
   - 📥 Ticket Inbox: All unresolved tickets
   - 🤖 AI Handling Queue: AI-managed tickets
   - 👤 Human Queue: Human-required tickets

5. **Resolution & Escalation**
   - AI responses can be approved
   - Tickets can be escalated
   - SLA state updates dynamically
6. **Retrieval-Augmented Generation (RAG)**
---

## ✨ Key Features

### 📥 Ticket Inbox
- Unified view of unresolved tickets
- Filters for priority and SLA status
- Urgent alerts for critical tickets

### 🤖 AI Handling Queue
- Automated AI resolution
- Confidence-based escalation
- AI response preview

### 👤 Human Queue
- Critical and escalated tickets
- Sorted by urgency and SLA risk

### 📊 Operations Dashboard
- Ticket processing metrics
- AI vs Human distribution
- Priority and SLA analytics
- Escalation rate monitoring

---

## 🧠 Intelligence Layer

### Machine Learning
- Priority classification: `Low | Medium | High | Critical`
- Issue classification: `Billing | Technical | Refund | Product | Access`

### Business Rules
- SLA calculation
- Escalation logic
- Handler decision rules

### LLM (Optional Layer)
- Human-like response generation
- Explainable AI decisions

This hybrid architecture ensures **accuracy, transparency, and scalability**.

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** – REST API
- **Python** – Core logic

### Frontend
- **Streamlit** – Interactive dashboard

### AI & ML
- **scikit-learn**
- Rule-based SLA engine
- Optional LLM integration

### Data
- CSV-based dataset (database-ready design)

---

## 👥 Target Users

- Customer Support Teams
- SaaS Companies
- Operations Managers
- SLA-driven enterprises

---

## 🚀 Why AssistFlow AI?

✔ SLA-aware by design  
✔ AI + Human collaboration  
✔ Clear operational visibility  
✔ Explainable decisions  
✔ Production-oriented architecture  

AssistFlow AI focuses on **decision intelligence**, not just chat automation.

---


---

## 👨‍💻 Team

- **Mayank Saini**
- **Saurabh**
- **Ritik Tanwar**

---

## 🏁 Future Scope

- Real-time ticket ingestion
- Database-backed persistence
- Production LLM APIs
- Agent performance analytics
- Enterprise deployment

---

## 📌 Conclusion

AssistFlow AI demonstrates how **AI-driven prioritization and SLA awareness** can transform customer support operations.

It is not just a prototype — it is a **scalable, deployable support intelligence system**.


