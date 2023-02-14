from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import GPTNeoForCausalLM, AutoTokenizer, TextGenerationPipeline

import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server
import remote_llm


async def main(host='127.0.0.1', port=50055):
    print('starting server')
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3b")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3b")
    server = Server([remote_llm.ServiceGPTNeo(model=model, tokenizer=tokenizer)])
    with graceful_exit([server]):
        await server.start(host, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())