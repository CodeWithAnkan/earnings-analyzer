import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SarvamClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url="https://api.sarvam.ai/v1"  # Sarvam API base URL
        )
    
    def chat_completion(self, messages, model="sarvam-1", temperature=0.1):
        """Create a chat completion using Sarvam API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Sarvam API call: {e}")
            return None
    
    def extract_entities(self, text):
        """Extract entities (guidance, risks, metrics) from text."""
        prompt = f"""
        Extract financial entities from this earnings call transcript segment. 
        Return JSON with these categories:
        - guidance: forward-looking statements, forecasts, targets
        - risks: risk factors, concerns, challenges mentioned  
        - metrics: specific numbers, percentages, financial figures
        
        Text: {text}
        
        Response format:
        {{
            "guidance": ["entity1", "entity2"],
            "risks": ["risk1", "risk2"], 
            "metrics": ["metric1", "metric2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a financial analyst extracting structured entities from earnings calls."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages)
    
    def score_confidence(self, text):
        """Score speaker confidence based on hedge words (0-1 scale)."""
        prompt = f"""
        Analyze the confidence level of this speaker based on language used.
        Look for hedge words, uncertainty phrases, and definitive statements.
        Return a confidence score from 0.0 (very uncertain) to 1.0 (very confident).
        
        Text: {text}
        
        Response format: Return only the number, e.g. "0.75"
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at detecting confidence levels in business communication."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages, temperature=0.0)
        
        try:
            return float(response.strip())
        except (ValueError, AttributeError):
            return 0.5  # Default middle confidence

# Global client instance
sarvam_client = SarvamClient()
