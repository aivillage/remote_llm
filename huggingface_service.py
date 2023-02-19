from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server
from remote_llm.server import ServiceHuggingFace
from remote_llm.keystore import ApiKeystore

address = '0.0.0.0'
port = 50055
small = "EleutherAI/gpt-neo-125M"
large = "EleutherAI/gpt-j-6B"
xlarge = "EleutherAI/gpt-neox-20b"
current = small

async def main():
    print('starting server')
    keystore = ApiKeystore("sqlite:///./keystore.db")
    #keystore.add_admin_key(name="admin", key="482fdd5f-b59c-43de-98b9-4e19a21b4d85")

    model = AutoModelForCausalLM.from_pretrained(current, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(current, torch_dtype=torch.float16)
    server = Server([ServiceHuggingFace(model=model, tokenizer=tokenizer, keystore=keystore)])
    with graceful_exit([server]):
        await server.start(address, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())