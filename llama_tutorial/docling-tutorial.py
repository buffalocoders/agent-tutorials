# **********************************************************
#                   docling教程
# **********************************************************
from llama_index.readers.docling import DoclingReader
from llama_index.core import SimpleDirectoryReader
from tools.constants import HOME_DIR
from devtools import debug
from docling.document_converter import DocumentConverter
import asyncio

DATA_DIR = HOME_DIR / 'datas'


async def llama_with_docling():
    file_extractor = {
        ".pdf": DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
    }

    docs = SimpleDirectoryReader(input_dir=DATA_DIR, file_extractor=file_extractor).load_data(show_progress=True)

    debug(f"加载了{len(docs)}个文档")


async def main():
    converter = DocumentConverter()
    docs = converter.convert(DATA_DIR / "docker_practice.pdf").document
    debug(f"文本：{len(docs.texts)}")
    debug(f"表格：{len(docs.tables)}")
    debug(f"图片：{len(docs.pictures)}")
    print("分割线".center(50,'='))
    print(docs.pictures[0])
    table = docs.tables[0]
    print(f"类型: {type(table)}")  # TableItem
    print(f"位置: 第{table.prov[0].page_no}页")

    # ⚠️ 这里才是关键！试试导出数据：
    try:
        df = table.export_to_dataframe(doc=docs)
        print(f"DataFrame: {df}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

    # 也试试表格的文本表示
    try:
        md = table.export_to_markdown(doc=docs)
        print(f"Markdown:\n{md}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")


if __name__ == '__main__':
    asyncio.run(main())
