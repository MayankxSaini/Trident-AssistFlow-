"""LLM assistance module"""

import os
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.append(".")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"


@dataclass
class LLMAssistanceResult:
    ticket_summary: str
    explanation_text: str
    suggested_response: str


class LLMProvider:
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise NotImplementedError
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        raise NotImplementedError


class TemplateLLMProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return f"[Template response for: {prompt[:50]}...]"
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        return text[:max_length-3] + "..." if len(text) > max_length else text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("openai package not installed")
            except Exception as e:
                print(f"OpenAI init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
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
            print(f"OpenAI error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        prompt = f"Summarize this support ticket in {max_length} chars or less:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or GEMINI_MODEL
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ Gemini initialized: {self.model_name}")
            except ImportError:
                print("google-generativeai not installed")
            except Exception as e:
                print(f"Gemini init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.model:
            return TemplateLLMProvider().generate(prompt, max_tokens)
        
        try:
            full_prompt = f"You are a helpful customer support assistant. Be concise and professional.\n\n{prompt}"
            response = self.model.generate_content(
                full_prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.7}
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        prompt = f"Summarize this support ticket in {max_length} chars or less:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or OLLAMA_BASE_URL or "http://localhost:11434"
        self.model = model or OLLAMA_MODEL
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return TemplateLLMProvider().generate(prompt, max_tokens)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        prompt = f"Summarize this briefly:\n\n{text}"
        return self.generate(prompt, max_tokens=150)


def get_llm_provider() -> LLMProvider:
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


_llm_provider: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def generate_ticket_summary(full_text: str) -> str:
    provider = get_provider()
    
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            prompt = f"Summarize this ticket in 2-3 sentences:\n\n{full_text}\n\nSummary:"
            return provider.generate(prompt, max_tokens=150)
        except Exception as e:
            print(f"LLM summary failed: {e}")
    
    text_lower = full_text.lower()
    products = ["laptop", "phone", "tv", "camera", "software", "app", "printer", "router", "tablet", "computer"]
    found_products = [p for p in products if p in text_lower]
    
    issues = {
        "not working": "functionality issue", "broken": "failure", "error": "error",
        "refund": "refund request", "billing": "billing concern", "password": "account issue"
    }
    found_issues = [v for k, v in issues.items() if k in text_lower]
    
    parts = []
    if found_products:
        parts.append(f"Product: {', '.join(found_products)}")
    if found_issues:
        parts.append(f"Issue: {', '.join(found_issues[:2])}")
    parts.append(f"Details: {full_text[:150]}...")
    
    return " | ".join(parts) if parts else "Ticket requires review."


def generate_explanation(predicted_priority: str, final_priority: str, issue_type: Optional[str],
                         sla_hours: int, sla_status: str, was_escalated: bool, handler_type: str) -> str:
    provider = get_provider()
    
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            prompt = f"""Explain this ticket's processing:
- Initial Priority: {predicted_priority}
- Final Priority: {final_priority}
- Issue Type: {issue_type or 'N/A'}
- SLA: {sla_hours}h, Status: {sla_status}
- Escalated: {'Yes' if was_escalated else 'No'}
- Handler: {handler_type}

Write a clear 4-5 sentence explanation."""
            return provider.generate(prompt, max_tokens=300)
        except Exception as e:
            print(f"LLM explanation failed: {e}")
    
    parts = [f"📊 PRIORITY: Classified as '{predicted_priority}' based on content analysis."]
    
    if was_escalated:
        parts.append(f"\n⬆️ ESCALATED: Priority raised from '{predicted_priority}' to '{final_priority}' due to SLA risk.")
    
    parts.append(f"\n⏰ SLA: {sla_hours}-hour window. Status: {sla_status.upper()}")
    
    if issue_type:
        parts.append(f"\n📁 ISSUE: {issue_type} related.")
    
    handler_icon = "🤖" if handler_type == "AI" else "👤"
    parts.append(f"\n{handler_icon} HANDLER: {handler_type}")
    
    if handler_type == "Human":
        reasons = []
        if final_priority in ["High", "Critical"]:
            reasons.append(f"'{final_priority}' priority requires human oversight")
        if issue_type in ["Billing", "Security"]:
            reasons.append(f"'{issue_type}' requires human judgment")
        parts.append(f" ({', '.join(reasons) if reasons else 'Policy requires human review'})")
    else:
        parts.append(" (Standard priority and issue type)")
    
    return "".join(parts)


def generate_suggested_response(full_text: str, issue_type: Optional[str],
                                 final_priority: str, handler_type: str) -> str:
    provider = get_provider()
    
    if not isinstance(provider, TemplateLLMProvider) and USE_LLM:
        try:
            timeline = {'Critical': '6 hours', 'High': '24 hours', 'Medium': '48 hours', 'Low': '72 hours'}
            prompt = f"""Write a professional support response for:

Issue: {full_text[:500]}
Type: {issue_type or 'General'}
Priority: {final_priority}
Handler: {handler_type}

Include: greeting, acknowledgment, timeline ({timeline.get(final_priority, '48 hours')}), next steps.
Keep under 150 words."""
            return provider.generate(prompt, max_tokens=400)
        except Exception as e:
            print(f"LLM response failed: {e}")
    
    response = ["Dear Valued Customer,\n"]
    response.append("Thank you for contacting us. We've received your ticket and are working on it.\n")
    
    if issue_type:
        acks = {
            "Technical issue": "We understand you're having technical difficulties.",
            "Billing": "We take billing concerns seriously and will review your account.",
            "Refund request": "We'll review your case per our refund policy."
        }
        response.append(acks.get(issue_type, "We're reviewing your request.") + "\n")
    
    timelines = {
        "Critical": "A senior specialist will contact you within 6 hours.",
        "High": "We'll respond within 24 hours.",
        "Medium": "We'll respond within 48 hours.",
        "Low": "We'll address this within 72 hours."
    }
    response.append(timelines.get(final_priority, "") + "\n")
    
    if handler_type == "Human":
        response.append("A dedicated agent has been assigned to your case.\n")
    else:
        response.append("Check our FAQ for immediate help.\n")
    
    response.append("\nBest regards,\nCustomer Support Team")
    return "\n".join(response)


def generate_llm_assistance(full_text: str, predicted_priority: str, final_priority: str,
                            issue_type: Optional[str], sla_hours: int, sla_status: str,
                            was_escalated: bool, handler_type: str) -> LLMAssistanceResult:
    return LLMAssistanceResult(
        ticket_summary=generate_ticket_summary(full_text),
        explanation_text=generate_explanation(predicted_priority, final_priority, issue_type,
                                               sla_hours, sla_status, was_escalated, handler_type),
        suggested_response=generate_suggested_response(full_text, issue_type, final_priority, handler_type)
    )


if __name__ == "__main__":
    print("Testing LLM assistance\n")
    
    result = generate_llm_assistance(
        full_text="My laptop screen is flickering and sometimes goes black.",
        predicted_priority="Medium",
        final_priority="High",
        issue_type="Technical issue",
        sla_hours=24,
        sla_status="at_risk",
        was_escalated=True,
        handler_type="Human"
    )
    
    print("SUMMARY:", result.ticket_summary)
    print("\nEXPLANATION:", result.explanation_text)
    print("\nRESPONSE:", result.suggested_response)
