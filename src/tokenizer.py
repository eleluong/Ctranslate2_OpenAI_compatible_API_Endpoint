from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Optional, Union

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
