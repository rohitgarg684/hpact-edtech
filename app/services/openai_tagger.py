import os
from openai import OpenAI

TAGGING_MODEL = "gpt-3.5-turbo"

class OpenAITagger:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_tags(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=TAGGING_MODEL,
            messages=[
                {"role": "system", "content": "You are an intelligent tagger that reads articles and provides relevant tags or categories."},
                {"role": "user", "content": f"Provide 3-6 relevant tags for this content:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
