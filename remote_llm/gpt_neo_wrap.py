"""Wrapper around Huggingface."""

import logging
from typing import (
    List,
    Optional,
)
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM

from .base_llm import AbstractLLM
from .schema import Generation, LLMResult
logger = logging.getLogger(__name__)

class GPTNeoWrap(AbstractLLM):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generator: TextGenerationPipeline
    max_length: int 
    num_sequences: int

    def __init__(self, *, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            generated = self.generator(
                prompt,
            )
            generated = [Generation(text=gen['generated_text'][len(prompt):]) for gen in generated]
            generations.append(generated)
        return LLMResult(generations=generations)
    
    def llm_name(self) -> str:
        return self.model.config._name_or_path