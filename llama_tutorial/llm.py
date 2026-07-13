# **********************************************************
#                   llama-index-llms-openai 使用教程
# **********************************************************
from llama_index.llms.openai import OpenAI
from cores.config import get_settings, AgentSettings
from llama_index.core.llms import ChatMessage
import asyncio


async def get_llm(settings: AgentSettings):
    """
    创建 llm
    :param settings:
    :return:
    """
    return OpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        api_base=settings.OPENAI_URL
    )


async def main():
    # ====== complete ====== #
    settings = get_settings()
    llm = await get_llm(settings)
    response = llm.complete(
        prompt="RAG是"
    )



if __name__ == "__main__":
    asyncio.run(main())
