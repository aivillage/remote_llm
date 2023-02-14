from typing import List, Optional
from pydantic import BaseModel
from langchain.llms import BaseLLM
from remote_llm import ServiceLLM
from langchain.schema import LLMResult, Generation
import asyncio
from grpclib.utils import graceful_exit
from grpclib.server import Server


class MockLLM(BaseLLM, BaseModel):
    def __init__(self):
        pass

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        print('generating')
        return LLMResult(generations=[[Generation(text='world')]])
    
    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        print('agenerating')
        return LLMResult(generations=[[Generation(text='world')]])
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
async def main(host='127.0.0.1', port=50051):
    print('starting server')
    server = Server([ServiceLLM(MockLLM())])
    with graceful_exit([server]):
        await server.start(host, port)
        await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())