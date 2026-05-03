"""app/intelligence/sarvam_client.py"""

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class SarvamClient:
    def __init__(self):
        import httpx
        self.client = OpenAI(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url="https://api.sarvam.ai/v1",
            http_client=httpx.Client()
        )
        self.model = "sarvam-m"

    def chat_completion(self, messages, temperature=0.1):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Sarvam API error: {e}")
            return None

    def _strip_think(self, text: str) -> str:
        """
        Remove <think>...</think> blocks that sarvam-m emits before its answer.
        Handles both closed tags and unclosed tags (model cut off mid-think).
        """
        if not text:
            return text

        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # If there's an unclosed <think> tag, drop everything from it onward
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)

        return text.strip()

    def _extract_json(self, text: str) -> str | None:
        """
        Try to extract a valid JSON object from text.
        Handles:
          - Clean JSON responses
          - Markdown fences (```json ... ```)
          - Truncated JSON (model hit max_tokens mid-object) — we attempt repair
        """
        if not text:
            return None

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try parsing as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Find the first { and try to parse from there
        brace_start = text.find("{")
        if brace_start == -1:
            return None

        candidate = text[brace_start:]

        # Try as-is
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        # Truncated JSON repair: close any open arrays and objects
        # Strategy: count open braces/brackets and close what's missing
        candidate = self._repair_truncated_json(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None

    def _repair_truncated_json(self, text: str) -> str:
        """
        Best-effort repair of JSON truncated mid-stream by max_tokens.
        Closes open strings, arrays, and objects in the right order.
        """
        # If we're mid-string (odd number of unescaped quotes after last complete value),
        # truncate to the last complete list item
        # Simpler approach: find the last complete value boundary and close from there

        # Remove trailing partial content after last complete array item
        # Look for the last }, "],  or "value" pattern and truncate there
        
        # Find the last position that looks like end of a complete string value
        # by scanning for the last '",' or '"]' or '"}'
        last_clean = max(
            text.rfind('",'),
            text.rfind('"]'),
            text.rfind('"}'),
        )

        if last_clean > 0:
            text = text[:last_clean + 1]  # keep up to and including the quote

        # Now count open structures
        stack = []
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack:
                    stack.pop()

        # Close open structures in reverse order
        closing = {"{": "}", "[": "]"}
        for opener in reversed(stack):
            text += closing[opener]

        return text

    def extract_entities(self, text: str):
        """Extract entities (guidance, risks, metrics) from text."""
        prompt = f"""Extract financial entities from this earnings call transcript segment.
Return ONLY valid JSON — no markdown fences, no explanation, no reasoning:
{{
    "guidance": ["forward-looking statement 1", ...],
    "risks": ["risk factor 1", ...],
    "metrics": ["specific number or KPI 1", ...]
}}

Keep each item under 100 characters. Include only clearly stated items.

Text: {text[:2000]}"""

        messages = [
            {"role": "system", "content": "You are a financial analyst. Return only a JSON object with keys: guidance, risks, metrics. No explanations, no reasoning traces."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)
        if not response:
            return None

        # Strip think blocks first
        clean = self._strip_think(response)

        # Extract and validate JSON
        result = self._extract_json(clean)

        if result is None:
            print(f"Failed to parse entity extraction response: {response[:80]}")

        return result

    def score_confidence(self, text: str):
        """Score speaker confidence based on hedge words (0.0 = very uncertain, 1.0 = very confident)."""
        prompt = f"""Analyze the speaker's confidence based on their linguistic certainty and commitment to their statements.
Consider the following:

High Confidence (towards 1.0): Use of definitive verbs (will, shall, is, are), concrete data points, direct answers, and assertive phrasing.
Low Confidence (towards 0.0): Frequent use of modal verbs (might, could, may), hedge words (perhaps, we believe, approximately, subject to), stalling, or non-committal phrasing.
Provide a nuanced score between 0.0 and 1.0 that reflects the overall tone of the speaker in this specific segment.

Return ONLY a single float between 0.0 and 1.0. No explanation.

Text: {text[:2000]}"""

        messages = [
            {"role": "system", "content": "Return only a single number between 0.0 and 1.0. No explanation, no reasoning."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages, temperature=0.0)
        if not response:
            return 0.5

        # Strip think blocks
        clean = self._strip_think(response)

        # Extract first float found
        match = re.search(r"(\d?\.\d+|\d+)", clean)
        try:
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            return 0.5
        except (ValueError, TypeError):
            return 0.5


# Global client instance
sarvam_client = SarvamClient()
