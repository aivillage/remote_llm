"""Wrapper around Huggingface."""
import logging
from typing import (
    List,
    Dict,
    Optional,
)
from grpclib.client import Channel
import grpclib
from transformers import TextGenerationPipeline

from .llm_rpc.api import RemoteLLMStub, GenerateRequest, GenerateReply, GenerateReplyGeneration, GenerateReplyGenerationList, LLMTypeRequest, LLMTypeReply
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

'''
This shouldn't exist, but for some reason the TextGenerationPipeline doesn't work when it's in the normal service class.
'''
class ServiceHuggingFace(): 
    generator: TextGenerationPipeline
    max_length: int 
    num_sequences: int
    api_key: Optional[str]

    def __init__(self, *, model, tokenizer, max_length: int = 100, num_sequences: int = 1, api_key: Optional[str] = None):
        self.generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
        self.max_length = max_length
        self.num_sequences = num_sequences
        self.api_key = api_key

    def check_key(self, api_key: Optional[str]) -> bool:
        if self.api_key is None:
            return True
        return self.api_key == api_key

    async def Generate(self, stream: "grpclib.server.Stream[GenerateRequest, GenerateReply]") -> None:
        request = await stream.recv_message()
        print(request.prompts)
        if not self.check_key(request.api_key):
            return None
        generations = []
        for prompt in request.prompts:
            
            generated = self.generator(
                prompt,
                max_length=self.max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=self.num_sequences,
                repetition_penalty=1.2,
                temperature=1.0,
                no_repeat_ngram_size=3,
            )
            generated = [GenerateReplyGeneration(text=gen['generated_text'][len(prompt):]) for gen in generated]
            generations.append(GenerateReplyGenerationList(generations=generated))
        reply = GenerateReply(generations=generations)
        await stream.send_message(reply)

    async def GetLlmType(self, stream: "grpclib.server.Stream[LLMTypeRequest, LLMTypeReply]") -> None:
        request = await stream.recv_message()
        if not self.check_key(request.api_key):
            return None
        msg = LLMTypeReply(llm_type=self.llm._llm_type)
        await stream.send_message(msg)

    def __mapping__(self) -> Dict[str, "grpclib.const.Handler"]:
        return {
            "/llm_rpc.api.RemoteLLM/Generate": grpclib.const.Handler(
                self.Generate,
                grpclib.const.Cardinality.UNARY_UNARY,
                GenerateRequest,
                GenerateReply,
            ),
            "/llm_rpc.api.RemoteLLM/GetLlmType": grpclib.const.Handler(
                self.GetLlmType,
                grpclib.const.Cardinality.UNARY_UNARY,
                LLMTypeRequest,
                LLMTypeReply,
            )
        }

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
        self, prompts: List[str],
    ) -> LLMResult:
        """Generate text using the remote llm."""
        result = await self.client.generate(prompts=prompts)
        print(result)
        return LLMResult(generations=[unpack_generation_list(g) for g in result.generations], api_key=self.api_key)
