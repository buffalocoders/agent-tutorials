# **********************************************************
#                   ragas 使用
# **********************************************************
import asyncio

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from cores.llms import get_llm, get_embedding
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.integrations.llama_index import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from tools.constants import HOME_DIR
from cores.config import get_settings


APPSettings = get_settings()

DATA_DIR = HOME_DIR / "datas"


async def main():
    # ====== 0、配置 ====== #
    Settings.llm = await get_llm()
    Settings.embed_model = await get_embedding()

    # ====== 1、加载文档 ====== #
    docs = SimpleDirectoryReader(input_dir=DATA_DIR).load_data(show_progress=True)
    print(f"成功加载了{len(docs)}个文档")

    # ====== 2、构建 index ====== #
    index = VectorStoreIndex.from_documents(documents=docs, show_progress=True)
    query_engine = index.as_query_engine()

    # ====== 3、生成测试数据集 ====== #
    from openai import OpenAI
    client = OpenAI(
        api_key=APPSettings.SF_API_KEY,
        base_url=APPSettings.SF_URL
    )
    evalator_llm = llm_factory(
        model=APPSettings.SF_MODEL,
        client=client,
    )

    generator = TestsetGenerator(
        llm=evalator_llm,
        embedding_model=await get_embedding(),
    )

    test_set = generator.generate_with_llamaindex_docs(documents=docs, testset_size=10)
    df = test_set.to_dataframe()
    print("生成的测试数据集".center(50, '='))
    print(df.head())

    # ====== 4、配置评估指标 ====== #


    custom_metrics = [
        Faithfulness(llm=evalator_llm),
        AnswerRelevancy(llm=evalator_llm),
        ContextPrecision(llm=evalator_llm),
        ContextRecall(llm=evalator_llm),
    ]

    # ====== 5、进行评估 ====== #
    print("开始评估".center(50, '='))
    ragas_dataset = test_set.to_evaluation_dataset()

    result = evaluate(
        query_engine=query_engine,
        dataset=ragas_dataset,
        metrics=custom_metrics,
    )

    # ====== 6、分析结果 ====== #
    print("分析结果".center(50, '='))
    for metric_name, score in result.items():
        print(f"  {metric_name}: {score:.4f}")

    # 转为 DataFrame 查看每条数据的评分
    result_df = result.to_pandas()
    print("\n每条数据的详细评分：")
    print(result_df[["user_input", "faithfulness", "answer_relevancy",
                     "context_precision", "context_recall"]].to_string())



if __name__ == "__main__":
    asyncio.run(main())