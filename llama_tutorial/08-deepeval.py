# **********************************************************
#                   deepeval 评测
# **********************************************************
import asyncio
import os
from deepeval import assert_test, evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from dotenv import load_dotenv
from tools.constants import HOME_DIR
from cores.config import get_settings
from deepeval.synthesizer import Synthesizer

APPSettings = get_settings()

env_file = HOME_DIR / ".env"

load_dotenv(env_file)

os.environ["OPENAI_API_KEY"] = APPSettings.SF_API_KEY
os.environ["OPENAI_BASE_URL"] = APPSettings.SF_URL


async def main():
    # ====== 定义评估指标 ====== #
    revlancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=APPSettings.SF_MODEL,
    )

    test_case = LLMTestCase(
        input="什么是 DeepEval？",
        actual_output="DeepEval 是一个开源的 LLM 评测框架。",
        expected_output="DeepEval 是一个开源 Python 库，用于评估大语言模型应用的质量。"
    )

    assert_test(test_case, [revlancy_metric])


async def main2():
    # 创建多个测试用例
    test_cases = [
        LLMTestCase(
            input="什么是 DeepEval？",
            actual_output="DeepEval 是一个开源的 LLM 评测框架。",
            retrieval_context=["DeepEval 是一个开源的 LLM 评估框架，提供 50+ 种评测指标。"]
        ),
        LLMTestCase(
            input="DeepEval 支持哪些指标？",
            actual_output="DeepEval 支持 Faithfulness、Answer Relevancy 等指标。",
            retrieval_context=["DeepEval 提供 50+ 种研究级评测指标，包括 RAG 指标、Agent 指标等。"]
        )
    ]

    # 定义评测指标
    metrics = [
        FaithfulnessMetric(threshold=0.7, model="deepseek-ai/DeepSeek-V3"),
        AnswerRelevancyMetric(threshold=0.5, model="deepseek-ai/DeepSeek-V3")
    ]

    # 批量评估
    results = evaluate(test_cases=test_cases, metrics=metrics, display_config=DisplayConfig(print_results=True,verbose_mode=False))
    print(results)


async def main3():
    synthesizer = Synthesizer(
        model=APPSettings.SF_MODEL,
    )

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[HOME_DIR/"datas"]
    )

    for g in goldens:
        print(f"Q: {g.input}")
        print(f"A: {g.expected_output}\n")

if __name__ == "__main__":
    asyncio.run(main3())
