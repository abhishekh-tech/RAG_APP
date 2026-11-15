import os 
from dotenv import load_dotenv,find_dotenv
_= load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]

# Completion model

from langchain_groq import ChatGroq

llmModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

response = llmModel.invoke(
    "Write a short 5 line poem about JFK"
)
print(response)