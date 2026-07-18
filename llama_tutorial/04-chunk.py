# **********************************************************
#                   分块
# **********************************************************
import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from tools.constants import HOME_DIR
from llama_index.readers.file import PyMuPDFReader, PDFReader
from devtools import debug
from cores.llms import get_embedding

DATA_DIR = HOME_DIR / "datas"
file_extractor = {
    ".pdf": PDFReader()
}


async def main():
    docs = SimpleDirectoryReader(input_dir=DATA_DIR, file_extractor=file_extractor).load_data(show_progress=True)
    debug(f"文档数：{len(docs)}")

    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = await parser.aget_nodes_from_documents(documents=docs, show_progress=True)
    debug(f"chunk 数量：{len(nodes)}")

    parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=await get_embedding()
    )
    nodes = await parser.aget_nodes_from_documents(documents=docs, show_progress=True)
    debug(f"基于语义切割的 chunk数量：{len(nodes)}")


if __name__ == "__main__":
    asyncio.run(main())
