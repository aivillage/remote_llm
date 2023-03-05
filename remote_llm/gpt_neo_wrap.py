"""Wrapper around Huggingface."""
import torch
import logging
from typing import (
    List,
    Optional,
)
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM

from .base_llm import EducationalLLM, get_generational_guts
from .schema import Generation, GenerationalGuts, LLMResult
logger = logging.getLogger(__name__)

class GPTNeoWrap(EducationalLLM):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generator: TextGenerationPipeline
    max_new_tokens: int 
    num_sequences: int

    def __init__(self, *, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens = 20, num_sequences = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
        self.max_new_tokens = max_new_tokens
        self.num_sequences = num_sequences

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            generated = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.0,
            )
            generated = [Generation(text=gen['generated_text'][len(prompt):]) for gen in generated]
            generations.append(generated)
        return LLMResult(generations=generations)
    
    def llm_name(self) -> str:
        return self.model.config._name_or_path
    
    def tokens(self, text: str) -> List[int]:
        return self.tokenizer(text)["input_ids"]
    
    def token_embeddings(self, text: str) -> torch.tensor:
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens.to(self.model.device)
        return self.model._modules["transformer"].wte(tokens["input_ids"])[0]

    def position_embeddings(self, text: str) -> torch.tensor:
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
        input_shape = tokens.shape
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=self.model.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.model._modules["transformer"].wpe(position_ids)[0]

    def forward(self, text: str) -> torch.tensor:
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens.to(self.model.device)
        return self.model.transformer(**tokens, output_hidden_states=True).last_hidden_state[0]

    def logits(self, text: str) -> torch.tensor:
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens.to(self.model.device)
        return torch.sigmoid(self.model.lm_head(self.forward(text)))

    def displayable_tokens(self, tokens: List[int]) -> List[str]:
        def raw(string: str, replace: bool = False) -> str:
            """Returns the raw representation of a string. If replace is true, replace a single backslash's repr \\ with \."""
            r = repr(string)[1:-1]  # Strip the quotes from representation
            if replace:
                r = r.replace('\\\\', '\\')
            return r
        return [raw(self.tokenizer.convert_tokens_to_string([token])) for token in self.tokenizer.convert_ids_to_tokens(tokens)]
    
    def get_generational_guts(self, text: str, *, top_k_logits: int = 5, fft: bool = True, embedding_trunkation: Optional[int] = 25) -> GenerationalGuts:
        return get_generational_guts(self, text, top_k_logits=top_k_logits, fft=fft, embedding_trunkation=embedding_trunkation)