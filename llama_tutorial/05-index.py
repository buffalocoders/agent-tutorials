# **********************************************************
#                   索引 index
# **********************************************************
import asyncio
from devtools import debug

from llama_index.core import SimpleDirectoryReader, KeywordTableIndex
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from tools.constants import HOME_DIR

DATA_DIR = HOME_DIR / "datas"


async def main():
    reader = DoclingReader()

    docs = list(reader.load_data(file_path=DATA_DIR/"docker_practice.pdf"))

    parser = DoclingNodeParser()

    print(f"加载了{len(docs)}个文档")

    nodes = parser.get_nodes_from_documents(documents=docs)
    print(f"分成了{len(nodes)}个节点")

    index = KeywordTableIndex.build_index_from_nodes(nodes=nodes)

    debug(index)


if __name__ == "__main__":
    asyncio.run(main())
