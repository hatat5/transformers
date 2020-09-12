from .configuration_utils import PretrainedConfig
from .utils import logging
from typing import Dict
from .tokenization_gpt2 import GPT2Tokenizer

logger = logging.get_logger(__name__)

class BasicTMConfig(PretrainedConfig):
    model_type = "basic_tm"
    def __init__(self,
                 hparams: Dict,
                 tgt_tokenizer: GPT2Tokenizer):
        super(BasicTMConfig, self).__init__(bos_token_id=tgt_tokenizer.bos_token_id,
                                            pad_token_id=tgt_tokenizer.pad_token_id,
                                            eos_token_id=tgt_tokenizer.eos_token_id,
                                            is_encoder_decoder=True)

        self.vocab_size = tgt_tokenizer.vocab_size
        self.hparams = hparams
