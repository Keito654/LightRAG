import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import aiohttp
import logging
import asyncio
from pathlib import Path
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

WORKING_DIR = "./dickens"


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),
        "top_p": kwargs.get("top_p", 1),
        "n": kwargs.get("n", 1),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def embedding_func(texts: list[str]) -> np.ndarray:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_EMBEDDING_API_VERSION}"

    payload = {"input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)

embedding_dimension = 1536

async def insert():
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    source_dir = Path('./operation/source')
    file_contents = []

    for file_path in source_dir.rglob('*'):  # 再帰的にすべてのファイルを取得
        if file_path.is_file():
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    content = f.read()
                    file_contents.append(content) 
                # ファイルの内容を処理する
                    print(f"読み取ったファイル: {file_path}")
            except Exception as e:
                print(f"ファイル {file_path} の読み取り中にエラーが発生しました: {e}")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: rag.insert(file_contents)
        )
        print("successfully insertion")
    except Exception as e:
        print("failed insertion")
        print(f"{e}")

async def query():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    with open("./operation/prompt/prompt.txt", encoding="utf-8") as f:
        query_text = f.read()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: rag.query(query_text, param=QueryParam(mode="hybrid"))
        )
        with open("./operation/out/out.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("successfully query")
    except Exception as e:
        print("failed query")
        print(f"{e}")

