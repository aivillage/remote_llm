"""Wrapper around Huggingface."""
import logging
from typing import (
    List,
    Optional,
)
from grpclib.client import Channel

from .llm_rpc.api import RemoteLLMStub, GenerateReplyGenerationList
from .schema import Generation, LLMResult, unpack_generational_guts, GenerationalGuts

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
    host: str
    port: int
    api_key: str = None
    channel_kwargs: Optional[dict] = None
    
    def __init__(
        self,
        host: str,
        port: int,
        api_key: str = None,
        channel_kwargs: Optional[dict] = None,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.channel_kwargs = channel_kwargs or {}

    async def generate_text(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Generate text using the remote llm."""
        async with Channel(self.host, self.port, **self.channel_kwargs) as channel:
            client = RemoteLLMStub(channel)
            result = await client.generate(prompts=prompts, stop=stop, api_key=self.api_key)
            return LLMResult(generations=[unpack_generation_list(g) for g in result.generations])
    
    async def llm_name(self) -> str:
        """Get model info."""
        async with Channel(self.host, self.port, **self.channel_kwargs) as channel:
            client = RemoteLLMStub(channel)
            result = await client.get_llm_type(api_key=self.api_key)
            return result.llm_type
        
    async def generational_guts(
        self,
        text: str, *,
        top_k_logits: int = 5,
        fft_embeddings: bool = True,
        embedding_trunkation: Optional[int] = 25,
        response_type: Optional[str] = None,
    ) -> GenerationalGuts:
        """Get model info."""
        async with Channel(self.host, self.port, **self.channel_kwargs) as channel:
            client = RemoteLLMStub(channel)
            result = await client.generational_guts(
                prompt=text,
                api_key=self.api_key,
                top_k_logits=top_k_logits,
                fft_embeddings=fft_embeddings,
                embedding_trunkation=embedding_trunkation,
            )
            if response_type == 'json':
                return result.to_json()
            elif response_type == 'dict':
                return result.to_dict()
            else:
                return unpack_generational_guts(result)
    
    def sync_generate_text(
        self, prompts: List[str],
    ) -> LLMResult:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except:
            raise
        else:
            pass

        return loop.run_until_complete(self.generate_text(prompts))
    
    def sync_llm_name(self) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except:
            raise
        else:
            pass
        return loop.run_until_complete(self.llm_name())
    
    def sync_generational_guts(self, text: str) -> GenerationalGuts:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except:
            raise
        else:
            pass
        return loop.run_until_complete(self.generational_guts(text))