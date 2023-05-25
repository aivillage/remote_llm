from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server
from remote_llm.gpt_neo_wrap import GPTNeoWrap
from remote_llm.server import LLMService
from remote_llm.keystore import ApiKeystore

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

address = '0.0.0.0'
port = 50055
small = "EleutherAI/gpt-neo-125M"
large = "EleutherAI/gpt-j-6B"
xlarge = "EleutherAI/gpt-neox-20b"
current = small

async def main():
    print("Loading model.")
    keystore = ApiKeystore("sqlite:///./keystore.db")
    keystore.add_admin_key(name="admin", key="482fdd5f-b59c-43de-98b9-4e19a21b4d85") # Test key, don't use this. :)

    model = AutoModelForCausalLM.from_pretrained(current, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(current, torch_dtype=torch.float16)
    model = GPTNeoWrap(model=model, tokenizer=tokenizer)
    server = Server([LLMService(llm=model, keystore=keystore)])
    print('Starting server.')
    with graceful_exit([server]):
        await server.start(address, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())