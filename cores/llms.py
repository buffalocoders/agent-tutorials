# **********************************************************
#                   LLM 和 Embedding
# **********************************************************
import asyncio

from llama_index.llms.siliconflow import SiliconFlow
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from .config import get_settings, AgentSettings
import structlog

Settings = get_settings()


async def get_llm():
    """
    创建 llm
    :return:
    """
    return SiliconFlow(
        model=Settings.SF_MODEL,
        api_key=Settings.SF_API_KEY,
    )


async def get_embedding():
    """
    创建 embedding 模型
    :return:
    """
    return SiliconFlowEmbedding(
        model=Settings.SF_EMBEDDING,
        api_key=Settings.SF_API_KEY,
    )


async def main():
    logger = structlog.get_logger()

    llm = await get_llm()
    embedding = await get_embedding()

    resp = await llm.acomplete(prompt="RAG是什么？")
    logger.info(f"响应：{resp}")

    embedding_text = embedding.get_text_embedding("rag是什么？")
    logger.info(f"嵌入维度：{len(embedding_text)}")


if __name__ == "__main__":
    asyncio.run(main())
