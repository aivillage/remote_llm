try:
    from langchain.schema import Generation, LLMResult
    __all__ = ['Generation', 'LLMResult']
except ImportError:
    from dataclasses import dataclass
    from typing import (
        Any,
        List,
        Dict,
        Optional,
    )
    @dataclass
    class Generation:
        text: str
        generation_info: Optional[Dict[str, Any]] = None

    @dataclass
    class LLMResult:
        generations: List[List[Generation]]