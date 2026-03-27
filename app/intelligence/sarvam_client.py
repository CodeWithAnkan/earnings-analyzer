import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SarvamClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url="https://api.sarvam.ai/v1"
        )
        self.model = "sarvam-m"  # correct model name

    def chat_completion(self, messages, temperature=0.1):
        """Create a chat completion using Sarvam API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Sarvam API error: {e}")
            return None

    def extract_entities(self, text):
        """Extract entities (guidance, risks, metrics) from text."""
        prompt = f"""Extract financial entities from this earnings call transcript segment.
Return ONLY valid JSON — no markdown fences, no explanation:
{{
    "guidance": ["forward-looking statement 1", ...],
    "risks": ["risk factor 1", ...],
    "metrics": ["specific number or KPI 1", ...]
}}

Text: {text[:3000]}"""

        messages = [
            {"role": "system", "content": "You are a financial analyst extracting structured entities from earnings calls. Return only JSON."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)
        if not response:
            return None

        # Strip accidental markdown fences
        clean = re.sub(r"^```(?:json)?\s*", "", response.strip())
        clean = re.sub(r"\s*```$", "", clean)
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        return clean

    def score_confidence(self, text):
        """Score speaker confidence based on hedge words (0.0 = very uncertain, 1.0 = very confident)."""
        prompt = f"""Analyze the confidence level of this speaker based on language used.
Look for hedge words (may, might, could, approximately, we believe, we expect, subject to)
and definitive statements.

Return ONLY a single float between 0.0 and 1.0. No explanation.

Text: {text[:2000]}"""

        messages = [
            {"role": "system", "content": "You are an expert at detecting confidence levels in business communication. Return only a number."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages, temperature=0.0)
        if not response:
            return 0.5

        # 1. Fix the 'think' tag regex (yours had </think> twice)
        clean = re.sub(r"</think>.*?</think>", "", response, flags=re.DOTALL).strip()
    
        # 2. Extract the first float/int found in the response
        match = re.search(r"(\d?\.\d+|\d+)", clean)
    
        try:
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val)) # Clamp between 0 and 1
            return 0.5
        except (ValueError, TypeError):
            return 0.5


# Global client instance
sarvam_client = SarvamClient()
