# **********************************************************
#                llama-index-llms-openai-like
# **********************************************************
from llama_index.llms.openai_like import OpenAILike
from cores.config import get_settings, AgentSettings
from llama_index.core.llms import ChatMessage
import asyncio
from tools.logs import get_logger

logger = get_logger(__name__)


async def get_llm(settings: AgentSettings):
    """
    创建 llm
    :param settings:
    :return:
    """
    return OpenAILike(
        is_chat_model=True,  # 必须设置才能跳过白名单的检查
        model=settings.OPENAI_MODEL,
        api_base=settings.OPENAI_URL,
        api_key=settings.OPENAI_API_KEY,
        context_window=settings.CONTEXT_WINDOW
    )


async def main():
    settings = get_settings()
    llm = await get_llm(settings)

    response = llm.chat(messages=[ChatMessage(
        content="Rag是什么？"
    )])
    logger.info(response)


if __name__ == "__main__":
    asyncio.run(main())
