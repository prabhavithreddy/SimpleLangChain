import os
from langchain.output_parsers import DatetimeOutputParser
from src.after.enum_llm_models import LLMModels
from src.after.simple_chat_api import SimpleChatApi

os.environ["REQUESTS_CA_BUNDLE"] = r"../../ca-bundle-full.crt"

api = SimpleChatApi(LLMModels.AzureChatOpenAI)
instruction = "When was Sachin Tendulkar born."
response = api.create_prompt(instruction, DatetimeOutputParser())
print(response)
