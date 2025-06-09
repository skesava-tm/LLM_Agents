import os

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from transformers import AutoModel

import asyncio
import dotenv

dotenv.load_dotenv("../.env")
HF_TOKEN=os.getenv("HF_TOKEN_INFERENCE")

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


async def agent_func(query: str):
    # initialize llm
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=HF_TOKEN)
    # llm = AutoModel.from_pretrained("google/gemma-3-1b-it")

    # initialize agent
    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(multiply)],
        llm=llm
    )

    response = await agent.run(query)

    return response


if __name__=="__main__":

    response = asyncio.run(agent_func("What is 2 times 2?"))
    print(response)
