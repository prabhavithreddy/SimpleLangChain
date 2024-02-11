from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from src.after.abstract_chat_api import AbstractChatApi
from src.after.enum_llm_models import LLMModels


class SimpleChatApi(AbstractChatApi):
    def __init__(self, model: LLMModels):
        AbstractChatApi.__init__(self, model)

    def create_prompt(self, instruction: str, output_parser: BaseOutputParser = None) -> object:
        human_template = "{instruction}\n{format_instructions}"
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message])
        if not output_parser:
            chain = chat_prompt | self._llm
            response = chain.invoke(
                {"instruction": instruction, "format_instructions": "return the response as a string"})
            return response.content

        chain = chat_prompt | self._llm | output_parser
        return chain.invoke(
            {"instruction": instruction, "format_instructions": output_parser.get_format_instructions()})