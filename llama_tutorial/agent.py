import asyncio

from llama_index.core.agent.workflow import FunctionAgent,CodeActAgent

from cores.llms import get_openai_llm, get_embedding
from devtools import debug


# 定义工具
def multiply(a: float, b: float) -> float:
    """计算两个数的乘积"""
    debug("this is multiply")
    return a * b


def add(a: float, b: float) -> float:
    """计算两个数的和"""
    debug("this is add")
    return a + b


def divide(a: float, b: float) -> float:
    """计算两个数的商"""
    debug("this is divide")
    if b == 0:
        return float('inf')
    return a / b


async def main():
    llm = await get_openai_llm()
    agent = FunctionAgent(
        llm=llm,
        tools=[multiply, add, divide],
        system_prompt="你是一个数学计算器，可以使用工具进行数学运算。"
    )

    response = await agent.run(user_msg="计算 (20 + 30) * 2 等于多少？")
    print(response)


def process_fn(code):
    import subprocess
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as e:
        return f"Error: {str(e)}"

async def main2():
    llm = await get_openai_llm()
    agent = CodeActAgent(
        llm=llm,
        tools=[multiply, add, divide],
        system_prompt="你是一个数学计算器....",
        code_execute_fn=process_fn
    )

    response = await agent.run(user_msg="计算 (20 + 31) * 2 等于多少？")
    print(response)

if __name__ == "__main__":
    asyncio.run(main2())
