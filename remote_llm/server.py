"""Wrapper around Huggingface."""
import logging
from typing import (
    Dict,
    Optional,
)
import grpclib
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM

from .llm_rpc.api import GenerateRequest, GenerateReply, GenerateReplyGeneration, GenerateReplyGenerationList, LLMTypeRequest, LLMTypeReply
from .keystore import ApiKeystore
logger = logging.getLogger(__name__)

'''
This shouldn't exist, but for some reason the TextGenerationPipeline doesn't work when it's in the normal service class.
'''
class ServiceHuggingFace():
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generator: TextGenerationPipeline
    max_length: int 
    num_sequences: int
    keystore: Optional[ApiKeystore]

    def __init__(self, *, model, tokenizer, max_length: int = 100, num_sequences: int = 1, keystore: Optional[ApiKeystore] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
        self.max_length = max_length
        self.num_sequences = num_sequences
        self.keystore = keystore

    def check_key(self, api_key: Optional[str]) -> Optional[str]:
        if self.keystore is None:
            return ""
        if api_key is None:
            return None
        return self.keystore.check_key(key=api_key)

    async def Generate(self, stream: "grpclib.server.Stream[GenerateRequest, GenerateReply]") -> None:
        request = await stream.recv_message()
        user = self.check_key(request.api_key)
        if user is None:
            return stream.send_message(GenerateReply())
        
        generations = []
        for prompt in request.prompts:
            logging.info(f"Generating text for {user} with prompt: {prompt}")
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
        user = self.check_key(request.api_key)
        if user is None:
            return stream.send_message(LLMTypeReply())
        logging.info(f"Getting LLM type for {user}")
        model_name = self.model.config._name_or_path
        msg = LLMTypeReply(llm_type=model_name)
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