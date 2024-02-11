from abc import ABC, abstractmethod

from langchain_core.output_parsers import BaseOutputParser

from src.after.enum_llm_models import LLMModels
from src.after.llm_factory import LLMFactory


class AbstractChatApi(ABC):
    def __init__(self, model: LLMModels):
        self._llm = LLMFactory.create_instance(model)

    @abstractmethod
    def create_prompt(self, instruction: str, output_parser: BaseOutputParser) -> object:
        pass