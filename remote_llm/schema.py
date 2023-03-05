from dataclasses import dataclass
from typing import (
    Any,
    List,
    Dict,
    Optional,
)
from .llm_rpc.api import (
    GenerateReply,
    GenerateReplyGeneration,
    GenerateReplyGenerationList,
    GenerationalGutsReply,
    GenerationalGutsReplyTokenStack, 
    GenerationalGutsReplyGeneration,
)
import numpy

try:
    from langchain.schema import Generation, LLMResult
    __all__ = ['Generation', 'LLMResult']
except ImportError:
    @dataclass
    class Generation:
        text: str
        generation_info: Optional[Dict[str, Any]] = None

    @dataclass
    class LLMResult:
        generations: List[List[Generation]]


@dataclass
class GenerationalGuts:
    positional_embeddings: numpy.ndarray
    token_embeddings: numpy.ndarray
    hidden_states: numpy.ndarray

    sentence_tokens: List[str]
    sentence_ids: List[int]

    top_k_generated_token: List[str]
    top_k_generated_token_id: List[int]
    top_k_generated_token_logits: List[float]

    def pack(self) -> GenerationalGutsReply:
        tokens = []
        for pe, te, hs, st, si in zip(
            self.positional_embeddings,
            self.token_embeddings,
            self.hidden_states,
            self.sentence_tokens,
            self.sentence_ids,
        ):
            tokens.append(GenerationalGutsReplyTokenStack(
                token=st,
                token_id=si,
                positional_embedding=list(pe),
                token_embedding=list(te),
                hidden_state=list(hs),
            ))
        generations = []
        for t, i, l in zip(
            self.top_k_generated_token,
            self.top_k_generated_token_id,
            self.top_k_generated_token_logits,
        ):
            generations.append(GenerationalGutsReplyGeneration(
                token=t,
                id=i,
                logit=l,
            ))
        return GenerationalGutsReply(
            tokens=tokens,
            generations=generations,
        )
    
def unpack_generational_guts(guts: GenerationalGutsReply) -> GenerationalGuts:
    positional_embeddings = numpy.array([t.positional_embedding for t in guts.tokens])
    token_embeddings = numpy.array([t.token_embedding for t in guts.tokens])
    hidden_states = numpy.array([t.hidden_state for t in guts.tokens])
    sentence_tokens = [t.token for t in guts.tokens]
    sentence_ids = [t.token_id for t in guts.tokens]
    top_k_generated_token = [g.token for g in guts.generations]
    top_k_generated_token_id = [g.id for g in guts.generations]
    top_k_generated_token_logits = [g.logit for g in guts.generations]
    return GenerationalGuts(
        positional_embeddings=positional_embeddings,
        token_embeddings=token_embeddings,
        hidden_states=hidden_states,
        sentence_tokens=sentence_tokens,
        sentence_ids=sentence_ids,
        top_k_generated_token=top_k_generated_token,
        top_k_generated_token_id=top_k_generated_token_id,
        top_k_generated_token_logits=top_k_generated_token_logits,
    )


def unpack_result(result: GenerateReply) -> LLMResult:
    return LLMResult(generations=[[Generation(text=gg.text, generation_info=gg.generation_info) for gg in g.generations] for g in result.generations])

def pack_result(result: LLMResult) -> GenerateReply:
    return GenerateReply(generations=[GenerateReplyGenerationList(generations=[GenerateReplyGeneration(text=gg.text, generation_info=gg.generation_info) for gg in g]) for g in result.generations])