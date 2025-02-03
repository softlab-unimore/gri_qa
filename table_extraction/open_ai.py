import os
import logging

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.openai")

class OpenAIChatModel:
    def __init__(self, model_name, temperature):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = model_name
        self.temperature = temperature

    def _get_llm(self,):
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )

        return self.llm
    
    def _create_messages(self, system_prompt, human_prompt):
        return [
            (
                "system",
                system_prompt
            ),
            (
                "human",
                human_prompt
            )
        ]

    def invoke(self, system_prompt, human_prompt):
        if not hasattr(self, "llm"):
            self._get_llm()

        with get_openai_callback() as cb:
            result = self.llm.invoke(self._create_messages(system_prompt, human_prompt))
            logger.info(cb)

        return result
    