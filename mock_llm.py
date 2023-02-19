from typing import List, Optional
from pydantic import BaseModel
from remote_llm.server import LLMService
from remote_llm.base_llm import AbstractLLM
from remote_llm.schema import LLMResult, Generation
import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server


class MockLLM(AbstractLLM):
    def __init__(self):
        pass

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        print('generating')
        return LLMResult(generations=[[Generation(text='world')]])
    
    def llm_name(self) -> str:
        return "mock"
    
async def main(host='127.0.0.1', port=50051):
    print('starting server')
    server = Server([LLMService(llm=MockLLM())])
    with graceful_exit([server]):
        await server.start(host, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())