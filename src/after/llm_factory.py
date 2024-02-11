import os

from langchain.chat_models import AzureChatOpenAI, ChatAnthropic, ChatOpenAI
from src.after.enum_llm_models import LLMModels


class LLMFactory:
    """
        Factory class for ChatOpenAI
    """
    @staticmethod
    def create_instance(llm_model:LLMModels) -> ChatOpenAI:
        """
            Create instance of ChatOpenAI based on the model type
        """
        if llm_model == LLMModels.AzureChatOpenAI:
            return AzureChatOpenAI(
                deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_type=os.environ["OPENAI_API_TYPE"],
            )
        elif llm_model == LLMModels.Anthropic:
            return ChatAnthropic(temperature=0, anthropic_api_key="YOUR_API_KEY", model_name="claude-instant-1.2")
        else:
            raise ValueError(f"Unknown LLM model: {llm_model}")