# AssistFlow AI  
### SLA-Aware Intelligent Customer Support System

---

## Problem Overview

Organizations handling large volumes of customer support tickets face a common challenge: **all tickets are treated equally at the point of entry**. This leads to urgent issues being delayed by low-impact requests, inefficient use of human agents, missed SLAs, and declining customer satisfaction.

Traditional keyword-based systems fail to understand the true intent, urgency, and business risk associated with customer messages. As ticket volumes grow, these inefficiencies translate directly into longer resolution times, reduced First Contact Resolution (FCR), and increased operational costs.

---

## Our Service: AssistFlow AI

AssistFlow AI is an **AI-assisted decision-support system** designed to optimize how customer support tickets are handled. Instead of replacing human agents, the system introduces an **intelligent workflow layer** that evaluates every ticket before it reaches a support team.

The service focuses on three core outcomes:
- Ensuring urgent issues are identified and handled first
- Preventing silent SLA breaches through proactive monitoring
- Reducing agent workload by automating triage and assistance

---

## System Workflow

When a customer submits a support request, the ticket enters AssistFlow AI’s intelligent processing pipeline.

The system first analyzes the content of the ticket to understand the nature of the issue. By examining the subject and description together, it determines the issue category (such as billing, technical, or account-related), extracts key context, and produces a concise summary that can be easily reviewed by a human agent.

Using this understanding, the system assigns an initial priority level—Low, Medium, High, or Critical—based on the potential impact and risk associated with the issue. This prioritization ensures that problems involving financial loss, access issues, or security concerns are immediately distinguished from general inquiries.

Once priority is assigned, an SLA is attached to the ticket. Each priority level has a defined resolution window, not to increase pressure on low-priority cases, but to ensure **accountability** and prevent any request from being overlooked.

Unlike traditional systems where priority is fixed, AssistFlow AI continuously monitors the ticket’s SLA status. If a ticket approaches its SLA limit without progress, the system automatically escalates its priority and surfaces it higher in the queue. This SLA-aware escalation acts as a safety mechanism, ensuring issues are addressed before contractual or customer experience damage occurs.

Based on the final priority and risk level, the system determines whether the ticket can be safely handled with AI assistance or requires a human agent. Low-risk, repetitive issues may receive AI-assisted responses, while high-risk or sensitive cases are always routed to human support staff.

For human-handled tickets, AssistFlow AI provides contextual assistance by summarizing the issue, explaining why the ticket was prioritized or escalated, and suggesting possible resolution steps. Human agents remain fully in control, reviewing and approving all actions.

This workflow ensures that **the right issues reach the right people at the right time**, while continuously adapting to time-based risk through SLA monitoring.

---

## SLA-Aware Dynamic Escalation (Key Differentiator)

Most support systems assign ticket priority only once, at the moment of creation. AssistFlow AI treats priority as a dynamic attribute.

By continuously evaluating SLA risk, the system escalates tickets before breaches occur. This proactive approach shifts support operations from reactive firefighting to preventive intervention, significantly reducing customer dissatisfaction and operational penalties.

---

## Technology Stack

AssistFlow AI is built using a simple, scalable, and practical technology stack:

- **Frontend:** Streamlit, used to display tickets, priorities, SLA status, and AI suggestions to agents and managers.
- **Backend:** Python with FastAPI, responsible for orchestrating ticket flow, business logic, and API communication.
- **Intelligence Layer:**  
  - Machine Learning for text classification and signal extraction  
  - Large Language Models (LLMs) for summarization, explanation, and response drafting  
  - Rule-based logic for priority assignment, SLA escalation, and safety controls
- **Data Storage:** SQLite / PostgreSQL for storing ticket history, SLA status, and outcomes.

---

## Business Impact

By introducing intelligent prioritization and SLA-aware escalation, AssistFlow AI enables faster resolution times, higher First Contact Resolution, improved customer satisfaction, and better utilization of human agents. These improvements directly translate into reduced operational costs and more reliable support performance.

---

## Team Members

- **Mayank Saini**  
- **Saurabh**  
- **Ritik Tanwar**

---

## Conclusion

AssistFlow AI enhances customer support operations by introducing intelligence where it matters most—prioritization, accountability, and proactive risk management. By combining Machine Learning, explainable AI assistance, and SLA-aware workflows, the system delivers measurable business value while keeping humans firmly in control.
