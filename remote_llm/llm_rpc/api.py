# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: llm_rpc.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import List

import betterproto
import grpclib


@dataclass
class GenerateRequest(betterproto.Message):
    api_key: str = betterproto.string_field(1)
    prompts: List[str] = betterproto.string_field(2)
    stop: List[str] = betterproto.string_field(3)


@dataclass
class GenerateReply(betterproto.Message):
    generations: List["GenerateReplyGenerationList"] = betterproto.message_field(1)


@dataclass
class GenerateReplyGeneration(betterproto.Message):
    text: str = betterproto.string_field(1)
    # JSON object with additional information about the generation.
    generation_info: str = betterproto.string_field(2)


@dataclass
class GenerateReplyGenerationList(betterproto.Message):
    generations: List["GenerateReplyGeneration"] = betterproto.message_field(1)


@dataclass
class LLMTypeRequest(betterproto.Message):
    api_key: str = betterproto.string_field(1)


@dataclass
class LLMTypeReply(betterproto.Message):
    llm_type: str = betterproto.string_field(1)


class RemoteLLMStub(betterproto.ServiceStub):
    async def generate(
        self, *, api_key: str = "", prompts: List[str] = [], stop: List[str] = []
    ) -> GenerateReply:
        request = GenerateRequest()
        request.api_key = api_key
        request.prompts = prompts
        request.stop = stop

        return await self._unary_unary(
            "/llm_rpc.api.RemoteLLM/Generate",
            request,
            GenerateReply,
        )

    async def get_llm_type(self, *, api_key: str = "") -> LLMTypeReply:
        request = LLMTypeRequest()
        request.api_key = api_key

        return await self._unary_unary(
            "/llm_rpc.api.RemoteLLM/GetLlmType",
            request,
            LLMTypeReply,
        )
