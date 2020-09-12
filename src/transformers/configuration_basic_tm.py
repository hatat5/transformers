from .configuration_utils import PretrainedConfig
from .utils import logging
from typing import Dict
from .tokenization_gpt2 import GPT2Tokenizer

logger = logging.get_logger(__name__)

class BasicTMEncoderConfig(PretrainedConfig):
    model_type = "basic_tm_encoder"
    def __init__(self,
                 encoder_params: Dict,
                 tokenizer,
                 **common_kwargs):
        super(BasicTMEncoderConfig, self).__init__(bos_token_id=tokenizer.bos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id)

        self.vocab_size = tokenizer.vocab_size
        self.encoder_params = encoder_params
        self.tokenizer = tokenizer


class BasicTMDecoderConfig(PretrainedConfig):
    model_type = "basic_tm_decoder"
    def __init__(self,
                 decoder_params: Dict,
                 tokenizer: GPT2Tokenizer,
                 **common_kwargs):
        super(BasicTMDecoderConfig, self).__init__(bos_token_id=tokenizer.bos_token_id,
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=tokenizer.eos_token_id,
                                                   is_decoder=True)

        self.vocab_size = tokenizer.vocab_size
        self.decoder_params = decoder_params
        self.tokenizer = tokenizer
