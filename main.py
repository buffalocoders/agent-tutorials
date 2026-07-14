from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_tutorial.llm import get_llm
from tools.logs import get_logger
import asyncio

logger = get_logger(__name__)


async def demo():
    docs = SimpleDirectoryReader(input_dir="datas").load_data(show_progress=True)
    logger.info(f"Found {len(docs)} documents")
    logger.info(f"大小：{docs[0].metadata['file_size'] / 1000 / 1000 :.2f} MB")
    logger.info(f"meta:{docs[0].metadata}")


if __name__ == "__main__":
    asyncio.run(demo())
