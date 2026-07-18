# **********************************************************
#                   向量数据库
# **********************************************************
import asyncio

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import Settings
from cores.llms import  get_embedding,get_openai_llm
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tools.constants import HOME_DIR
from devtools import debug
from pymilvus import MilvusClient


async def main():
    Settings.llm = await get_openai_llm()
    Settings.embed_model = await get_embedding()

    # 清理旧 collection（如果存在且维度不匹配）
    milvus_client = MilvusClient(uri="http://localhost:19530")
    if "rag_collection" in milvus_client.list_collections():
        milvus_client.drop_collection("rag_collection")
        print("已删除旧 collection: rag_collection")

    milvus = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="rag_collection",
        enable_sparse=True,
        sparse_embedding_function=None,
        enable_dense=True,
        dim=2560,
    )

    storage = StorageContext.from_defaults(vector_store=milvus)

    docs = SimpleDirectoryReader(input_dir=HOME_DIR / "datas").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(
        documents=docs,
        storage_context=storage,
        show_progress=True
    )

    debug(index)


if __name__ == "__main__":
    asyncio.run(main())
    
