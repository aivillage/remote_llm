"""Wrapper around Huggingface."""
import logging
from typing import (
    List,
    Optional,
)
from grpclib.client import Channel

from .llm_rpc.api import RemoteLLMStub, GenerateReplyGenerationList
from .schema import Generation, LLMResult

import asyncio
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if str(e).startswith('There is no current event loop in thread'):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        raise
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

def unpack_generation_list(generations: GenerateReplyGenerationList) -> List[Generation]:
    return [Generation(text=g.text, generation_info=g.generation_info) for g in generations.generations]

class ClientLLM:
    """
    Remote LLM. Uses a GRPC server to generate text.
    """
    client: RemoteLLMStub  #: :meta private:
    
    def __init__(self, channel: Channel, api_key: str = None):
        self.client = RemoteLLMStub(channel)
        self.api_key = api_key

    def generate_text(
        self, prompts: List[str],
    ) -> LLMResult:
        return loop.run_until_complete(self.async_generate_text(prompts))

    async def async_generate_text(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Generate text using the remote llm."""
        result = await self.client.generate(prompts=prompts, stop=stop, api_key=self.api_key)
        return LLMResult(generations=[unpack_generation_list(g) for g in result.generations])
