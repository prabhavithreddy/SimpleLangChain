import os
import unittest
from datetime import datetime

from langchain.output_parsers import DatetimeOutputParser

from src.after.enum_llm_models import LLMModels
from src.after.simple_chat_api import SimpleChatApi


class SimpleChatApiTest(unittest.TestCase):

    def test_hello_world(self):
        os.environ["REQUESTS_CA_BUNDLE"] = r"../ca-bundle-full.crt"
        api = SimpleChatApi(LLMModels.AzureChatOpenAI)
        instruction = "Hello, World!"
        response = str(api.create_prompt(instruction))
        self.assertEquals(instruction, response)

    def test_datetime(self):
        os.environ["REQUESTS_CA_BUNDLE"] = r"../ca-bundle-full.crt"
        api = SimpleChatApi(LLMModels.AzureChatOpenAI)
        instruction = "When was Sachin Tendulkar born."
        expected = datetime(1973, 4, 24, 0, 0)
        actual = api.create_prompt(instruction, DatetimeOutputParser())
        self.assertEquals(expected, actual)
