import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello, test!"}],
    max_tokens=10
)
print(resp.choices[0].message['content'])
