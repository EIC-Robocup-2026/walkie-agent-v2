import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("MODEL_BASE_URL:", os.getenv("MODEL_BASE_URL"))
print("MODEL_API_KEY:", os.getenv("MODEL_API_KEY"))

model = ChatOpenAI(
    model="qwen3.5-9b",
    api_key="your api key goes here",
    base_url="http://10.0.0.210:8000/v1",
    temperature=0,
)

while True:
    prompt = input("Enter your prompt (or 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    response = model.invoke(prompt)
    print("Response:", response)