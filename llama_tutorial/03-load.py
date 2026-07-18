# **********************************************************
#                   加载文档
# **********************************************************
import asyncio

from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from tools.constants import HOME_DIR
from devtools import debug
from llama_index.readers.file import PyMuPDFReader

DATA_DIR = HOME_DIR / "datas"

file_extractor = {
    ".pdf": PyMuPDFReader(),
}


async def main():
    docs = SimpleDirectoryReader(input_dir=DATA_DIR, filename_as_id=True, num_files_limit=1,file_extractor=file_extractor).load_data(
        show_progress=True)
    debug("=" * 80)
    debug(f"加载了{len(docs)}个文档")
    debug(docs[0].metadata)
    debug(docs[0].doc_id)
    debug("=" * 80)
    docs = [d for d in docs if d.text.strip()]
    debug(docs[0].text[100:500])


if __name__ == "__main__":
    asyncio.run(main())
