"""
AssistFlow AI - LLM Assistance Module

WORKFLOW STEP 9: LLM USAGE (ASSISTANCE ONLY)

LLMs are used ONLY AFTER all decisions are final.

LLM Responsibilities:
- Summarize ticket content
- Explain why priority/SLA decisions were made
- Draft a response message

LLM must NEVER:
- Change priority
- Change SLA
- Change handler type

IMPORTANT: This module provides an interface for LLM usage.
Supports: OpenAI API, Ollama (local), or template fallback.
"""

import os
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.append(".")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"


@dataclass
class LLMAssistanceResult:
    """
    Container for all LLM-generated assistance outputs.
    
    These are SUGGESTIONS only - they do not change any decisions.
    """
    ticket_summary: str
    explanation_text: str
    suggested_response: str


def generate_ticket_summary(full_text: str) -> str:
    """
    Generate a concise summary of the ticket content.
    
    WHY: Support agents need quick context. A summary helps them
    understand the issue without reading the full description.
    
    Uses LLM if available, otherwise falls back to template-based approach.
    
    Args:
        full_text: Combined subject and description
        
    Returns:
        Summarized ticket text
    """
    provider = get_provider()
    
    # Try LLM-based summary if not using template provider
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            prompt = f"""Summarize this customer support ticket in 2-3 sentences. 
Focus on: the product/service involved, the issue type, and the customer's main concern.

Ticket:
{full_text}

Summary:"""
            return provider.generate(prompt, max_tokens=150)
        except Exception as e:
            print(f"LLM summary failed, using template: {e}")
    
    # Template-based summary (fallback)
    text_lower = full_text.lower()
    
    # Detect product mentions
    product_keywords = [
        "laptop", "phone", "tv", "camera", "software", "app", 
        "printer", "router", "tablet", "computer", "device"
    ]
    mentioned_products = [p for p in product_keywords if p in text_lower]
    
    # Detect issue types from text
    issue_indicators = {
        "not working": "functionality issue",
        "broken": "hardware/software failure",
        "error": "error encountered",
        "help": "assistance requested",
        "refund": "refund request",
        "cancel": "cancellation request",
        "billing": "billing concern",
        "password": "account access issue",
        "slow": "performance issue",
        "setup": "setup/installation help"
    }
    
    detected_issues = [v for k, v in issue_indicators.items() if k in text_lower]
    
    # Build summary
    summary_parts = []
    
    if mentioned_products:
        summary_parts.append(f"Product involved: {', '.join(mentioned_products)}")
    
    if detected_issues:
        summary_parts.append(f"Issue type: {', '.join(detected_issues[:2])}")
    
    # Add text snippet
    snippet_length = min(150, len(full_text))
    summary_parts.append(f"Description: {full_text[:snippet_length]}...")
    
    return " | ".join(summary_parts) if summary_parts else "Ticket requires review."


def generate_explanation(
    predicted_priority: str,
    final_priority: str,
    issue_type: Optional[str],
    sla_hours: int,
    sla_status: str,
    was_escalated: bool,
    handler_type: str
) -> str:
    """
    Generate explanation of why decisions were made.
    
    WHY: Transparency is crucial. Agents and customers should understand
    why a ticket was prioritized and routed a certain way.
    
    Uses LLM if available for more natural language.
    
    Args:
        predicted_priority: Original ML prediction
        final_priority: Priority after escalation
        issue_type: Predicted issue type
        sla_hours: Assigned SLA hours
        sla_status: Current SLA status
        was_escalated: Whether priority was escalated
        handler_type: AI or Human
        
    Returns:
        Explanation text
    """
    provider = get_provider()
    
    # Try LLM-based explanation
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            prompt = f"""Generate a professional explanation for a customer support ticket's processing decisions.

Details:
- Initial Priority (ML predicted): {predicted_priority}
- Final Priority: {final_priority}
- Issue Type: {issue_type or 'Not classified'}
- SLA Window: {sla_hours} hours
- SLA Status: {sla_status}
- Was Escalated: {'Yes' if was_escalated else 'No'}
- Handler Assignment: {handler_type}

Write a clear, professional explanation (4-5 sentences) covering:
1. Why this priority was assigned
2. SLA implications
3. {'Why the ticket was escalated' if was_escalated else ''}
4. Why this handler type was chosen

Explanation:"""
            return provider.generate(prompt, max_tokens=300)
        except Exception as e:
            print(f"LLM explanation failed, using template: {e}")
    
    # Template-based explanation (fallback)
    explanation_parts = []
    
    # Priority explanation
    explanation_parts.append(
        f"📊 PRIORITY ASSESSMENT:\n"
        f"   Initial analysis classified this ticket as '{predicted_priority}' priority "
        f"based on the content and keywords in the description."
    )
    
    # Escalation explanation (if applicable)
    if was_escalated:
        explanation_parts.append(
            f"\n⬆️ ESCALATION APPLIED:\n"
            f"   Priority was escalated from '{predicted_priority}' to '{final_priority}' "
            f"because the SLA status is '{sla_status}'. This ensures the ticket "
            f"receives appropriate attention before the deadline."
        )
    
    # SLA explanation
    explanation_parts.append(
        f"\n⏰ SLA INFORMATION:\n"
        f"   Based on the '{final_priority}' priority, this ticket has a "
        f"{sla_hours}-hour SLA window.\n"
        f"   Current SLA status: {sla_status.upper()}"
    )
    
    # Issue type context (if available)
    if issue_type:
        explanation_parts.append(
            f"\n📁 ISSUE CATEGORY:\n"
            f"   This appears to be a '{issue_type}' related issue."
        )
    
    # Handler explanation
    handler_emoji = "🤖" if handler_type == "AI" else "👤"
    explanation_parts.append(
        f"\n{handler_emoji} HANDLER ASSIGNMENT:\n"
        f"   This ticket is assigned to: {handler_type}\n"
        f"   Reason: "
    )
    
    if handler_type == "Human":
        reasons = []
        if final_priority in ["High", "Critical"]:
            reasons.append(f"'{final_priority}' priority requires human oversight")
        if issue_type in ["Billing", "Security"]:
            reasons.append(f"'{issue_type}' issues require human judgment")
        explanation_parts[-1] += ", ".join(reasons) if reasons else "Policy requires human review"
    else:
        explanation_parts[-1] += "Standard priority and issue type allow AI assistance"
    
    return "\n".join(explanation_parts)


def generate_suggested_response(
    full_text: str,
    issue_type: Optional[str],
    final_priority: str,
    handler_type: str
) -> str:
    """
    Generate a draft response message for the customer.
    
    WHY: Pre-drafted responses speed up agent workflow. Agents can
    review, edit, and personalize before sending.
    
    Uses LLM if available for more personalized responses.
    
    Args:
        full_text: Original ticket content
        issue_type: Predicted issue type
        final_priority: Final priority level
        handler_type: AI or Human
        
    Returns:
        Draft response text
    """
    provider = get_provider()
    
    # Try LLM-based response
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            prompt = f"""Write a professional customer support response email for this ticket.

Customer's Issue:
{full_text[:500]}

Context:
- Issue Type: {issue_type or 'General inquiry'}
- Priority Level: {final_priority}
- Handler: {handler_type}

Requirements:
1. Be professional and empathetic
2. Acknowledge the specific issue
3. Provide a timeline based on priority ({final_priority} = {'6 hours' if final_priority == 'Critical' else '24 hours' if final_priority == 'High' else '48 hours' if final_priority == 'Medium' else '72 hours'})
4. If Human handler, mention a dedicated agent will follow up
5. Keep it concise (under 150 words)

Response:"""
            return provider.generate(prompt, max_tokens=400)
        except Exception as e:
            print(f"LLM response failed, using template: {e}")
    
    # Template-based response (fallback)
    response_parts = [
        "Dear Valued Customer,\n",
        "Thank you for contacting our support team. We have received your ticket "
        "and are working to resolve your issue.\n"
    ]
    
    # Acknowledgment based on issue type
    if issue_type:
        issue_acknowledgments = {
            "Technical issue": (
                "We understand you're experiencing technical difficulties. "
                "Our technical team is reviewing your case."
            ),
            "Billing": (
                "We see this is regarding a billing matter. "
                "We take billing concerns seriously and will review your account."
            ),
            "Refund request": (
                "We've noted your refund request. "
                "Our team will review your case according to our refund policy."
            ),
            "Product inquiry": (
                "Thank you for your product inquiry. "
                "We'll provide you with the information you need."
            ),
            "Cancellation request": (
                "We've received your cancellation request. "
                "We'll process this and confirm the details with you."
            )
        }
        acknowledgment = issue_acknowledgments.get(
            issue_type,
            "We're reviewing your request and will get back to you soon."
        )
        response_parts.append(acknowledgment + "\n")
    
    # Priority-based timeline
    priority_timelines = {
        "Critical": "Given the urgency of your issue, a senior support specialist will contact you within 6 hours.",
        "High": "We're treating this as a high-priority issue and will respond within 24 hours.",
        "Medium": "We expect to provide a full response within 48 hours.",
        "Low": "We'll address your inquiry within 72 hours."
    }
    response_parts.append(priority_timelines.get(final_priority, "") + "\n")
    
    # Next steps
    if handler_type == "Human":
        response_parts.append(
            "A dedicated support agent has been assigned to your case and will "
            "reach out with a personalized solution.\n"
        )
    else:
        response_parts.append(
            "In the meantime, you may find helpful information in our FAQ section "
            "or knowledge base at [support link].\n"
        )
    
    # Closing
    response_parts.append(
        "\nIf you have any additional information to add, please reply to this ticket.\n"
        "\nBest regards,\n"
        "Customer Support Team"
    )
    
    return "\n".join(response_parts)


def generate_llm_assistance(
    full_text: str,
    predicted_priority: str,
    final_priority: str,
    issue_type: Optional[str],
    sla_hours: int,
    sla_status: str,
    was_escalated: bool,
    handler_type: str
) -> LLMAssistanceResult:
    """
    Complete LLM assistance pipeline.
    
    This is the main entry point for LLM assistance.
    Call this ONLY AFTER all decisions (priority, SLA, handler) are final.
    
    Args:
        full_text: Original ticket text
        predicted_priority: ML-predicted priority
        final_priority: Priority after escalation
        issue_type: Predicted issue type
        sla_hours: Assigned SLA hours
        sla_status: Current SLA status
        was_escalated: Whether priority was escalated
        handler_type: AI or Human
        
    Returns:
        LLMAssistanceResult containing summary, explanation, and response
    """
    # Generate all assistance components
    # NOTE: In production, these could be parallelized or combined into
    # a single LLM call for efficiency
    
    ticket_summary = generate_ticket_summary(full_text)
    
    explanation_text = generate_explanation(
        predicted_priority=predicted_priority,
        final_priority=final_priority,
        issue_type=issue_type,
        sla_hours=sla_hours,
        sla_status=sla_status,
        was_escalated=was_escalated,
        handler_type=handler_type
    )
    
    suggested_response = generate_suggested_response(
        full_text=full_text,
        issue_type=issue_type,
        final_priority=final_priority,
        handler_type=handler_type
    )
    
    return LLMAssistanceResult(
        ticket_summary=ticket_summary,
        explanation_text=explanation_text,
        suggested_response=suggested_response
    )


# =============================================================================
# LLM INTEGRATION INTERFACE (FOR FUTURE USE)
# =============================================================================

class LLMProvider:
    """
    Abstract interface for LLM providers.
    
    Implement this to integrate with OpenAI, Anthropic, local models, etc.
    """
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from prompt."""
        raise NotImplementedError
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text to max_length."""
        raise NotImplementedError


class TemplateLLMProvider(LLMProvider):
    """
    Default template-based provider (no actual LLM required).
    
    Use this as fallback or for testing without LLM API access.
    """
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return f"[Template response for: {prompt[:50]}...]"
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider for LLM assistance.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("WARNING: openai package not installed. Using template fallback.")
            except Exception as e:
                print(f"WARNING: Failed to initialize OpenAI client: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using OpenAI API."""
        if not self.client:
            return TemplateLLMProvider().generate(prompt, max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant. Be concise and professional."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text using OpenAI API."""
        prompt = f"Summarize the following customer support ticket in {max_length} characters or less:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider for LLM assistance.
    
    Requires GEMINI_API_KEY environment variable.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or GEMINI_MODEL
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ Gemini LLM initialized with model: {self.model_name}")
            except ImportError:
                print("WARNING: google-generativeai package not installed. Using template fallback.")
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini client: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Google Gemini API."""
        if not self.model:
            return TemplateLLMProvider().generate(prompt, max_tokens)
        
        try:
            # Add system context to the prompt
            full_prompt = f"""You are a helpful customer support assistant. Be concise and professional.

{prompt}"""
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text using Gemini API."""
        prompt = f"Summarize the following customer support ticket in {max_length} characters or less:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.
    
    Requires Ollama running locally.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or OLLAMA_BASE_URL or "http://localhost:11434"
        self.model = model or OLLAMA_MODEL
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama API."""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text using Ollama."""
        prompt = f"Summarize this customer support ticket briefly:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


def get_llm_provider() -> LLMProvider:
    """
    Get the appropriate LLM provider based on configuration.
    
    Priority:
    1. Gemini (if API key is set)
    2. OpenAI (if API key is set)
    3. Ollama (if base URL is set)
    4. Template fallback
    """
    if GEMINI_API_KEY and USE_LLM:
        provider = GeminiProvider()
        if provider.model:
            return provider
    
    if OPENAI_API_KEY and USE_LLM:
        provider = OpenAIProvider()
        if provider.client:
            return provider
    
    if OLLAMA_BASE_URL and USE_LLM:
        return OllamaProvider()
    
    return TemplateLLMProvider()


# Global LLM provider instance
_llm_provider: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    """Get or create the global LLM provider."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


if __name__ == "__main__":
    # Test LLM assistance module
    print("Testing LLM Assistance Module\n")
    print("=" * 60)
    
    result = generate_llm_assistance(
        full_text="My laptop screen is flickering and sometimes goes black. I've tried restarting but the problem persists. This is affecting my work.",
        predicted_priority="Medium",
        final_priority="High",
        issue_type="Technical issue",
        sla_hours=24,
        sla_status="at_risk",
        was_escalated=True,
        handler_type="Human"
    )
    
    print("TICKET SUMMARY:")
    print(result.ticket_summary)
    print("\n" + "=" * 60)
    print("\nEXPLANATION:")
    print(result.explanation_text)
    print("\n" + "=" * 60)
    print("\nSUGGESTED RESPONSE:")
    print(result.suggested_response)
