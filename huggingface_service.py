from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server
import remote_llm

import ssl

address = '0.0.0.0'
port = 50055
server_cert = 'ssl_config/server_keys/server.crt'
server_key = 'ssl_config/server_keys/server.key'
client_certs = 'ssl_config/client_keys/'

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.verify_mode = ssl.CERT_REQUIRED
context.load_cert_chain(certfile=server_cert, keyfile=server_key)
context.load_verify_locations(capath=client_certs)

small = "EleutherAI/gpt-neo-125M"
large = "EleutherAI/gpt-j-6B"
xlarge = "EleutherAI/gpt-neox-20b"
current = small

async def main():
    print('starting server')
    model = AutoModelForCausalLM.from_pretrained(current, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(current, torch_dtype=torch.float16)
    server = Server([remote_llm.ServiceHuggingFace(model=model, tokenizer=tokenizer)])
    with graceful_exit([server]):
        await server.start(address, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())