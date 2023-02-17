"""Wrapper around OpenAI APIs."""
import logging
from typing import (
    Any,
    List,
    Optional,
)

from langchain.llms.base import BaseLLM
from .llm_rpc.api import GenerateReplyGenerationList
from .schema import Generation, LLMResult
from .client import ClientLLM

import asyncio
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

def unpack_generation_list(generations: GenerateReplyGenerationList) -> List[Generation]:
    return [Generation(text=g.text, generation_info=g.generation_info) for g in generations.generations]

class ClientLangchain(BaseLLM):
    """
    Remote LLM. Uses a GRPC server to generate text.
    """
    client: ClientLLM
    
    def __init__(self, client: ClientLLM):
        super().__init__(client=client)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.client.async_generate_text(prompts, stop))

    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Generate text using the remote llm."""
        result = await self.client.async_generate_text(prompts=prompts, stop=stop)
        return LLMResult(generations=[unpack_generation_list(g) for g in result.generations])

    @property
    def _llm_type(self) -> str:
        loop = asyncio.get_event_loop()
        remote_type = loop.run_until_complete(self.client.get_llm_type())
        return f"remote:{remote_type.llm_type}"
    
    def save(self, file_path: Any) -> None:
        raise NotImplementedError("Cannot save remote LLMs.")
