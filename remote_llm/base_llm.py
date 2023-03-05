
import abc
from typing import List, Optional

import torch

from remote_llm.schema import GenerationalGuts, LLMResult

class AbstractLLM(object):
    @abc.abstractmethod
    def llm_name(self) -> str:
        pass

    @abc.abstractmethod
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        pass

class EducationalLLM(AbstractLLM):
    @abc.abstractmethod
    def tokens(self, text: str) -> List[str]:
        pass

    @abc.abstractmethod
    def tokens(self, text: str) -> List[int]:
        pass

    @abc.abstractmethod
    def token_embeddings(self, text: str) -> torch.tensor:
        pass
    
    @abc.abstractmethod
    def position_embeddings(self, text: str) -> torch.tensor:
        pass

    @abc.abstractmethod
    def forward(self, text: str) -> torch.tensor:
        pass

    @abc.abstractmethod
    def logits(self, text: str) -> torch.tensor:
        pass

    @abc.abstractmethod
    def displayable_tokens(self, tokens: List[int]) -> List[str]:
        pass

def get_generational_guts(llm: EducationalLLM, text: str, top_k_logits: int = 5, fft: bool = True, embedding_trunkation: Optional[int] = 100) -> GenerationalGuts:
    sentence_ids = llm.tokens(text)
    sentence_tokens = llm.displayable_tokens(sentence_ids)
    token_embeddings = llm.token_embeddings(text).to(torch.float32).detach().cpu()
    position_embeddings = llm.position_embeddings(text).to(torch.float32).detach().cpu()
    hidden_states = llm.forward(text).to(torch.float32).detach().cpu()

    logits = llm.logits(text)[-1,:].to(torch.float32).detach().cpu()
    logits = torch.topk(logits, k=top_k_logits, dim=-1)

    if fft:
        token_embeddings = torch.fft.fft(token_embeddings, dim=1).real.numpy()
        position_embeddings = torch.fft.fft(position_embeddings, dim=1).real.numpy()
        hidden_states = torch.fft.fft(hidden_states, dim=1).real.numpy()

    if embedding_trunkation is not None:
        token_embeddings = token_embeddings[:, :embedding_trunkation]
        position_embeddings = position_embeddings[:, :embedding_trunkation]
        hidden_states = hidden_states[:, :embedding_trunkation]

    top_k_generated_token_id = logits.indices
    top_k_generated_token = llm.displayable_tokens(top_k_generated_token_id)
    top_k_generated_token_logits = logits.values

    return GenerationalGuts(
        positional_embeddings=position_embeddings,
        token_embeddings=token_embeddings,
        hidden_states=hidden_states,

        sentence_ids=sentence_ids,
        sentence_tokens=sentence_tokens,

        top_k_generated_token=top_k_generated_token,
        top_k_generated_token_id=top_k_generated_token_id,
        top_k_generated_token_logits=top_k_generated_token_logits,
    )