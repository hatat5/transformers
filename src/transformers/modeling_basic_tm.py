import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from typing import Optional
from typing import Dict
from .modeling_utils import PreTrainedModel
from .configuration_basic_tm import BasicTMEncoderConfig, BasicTMDecoderConfig
from .modeling_bart import SinusoidalPositionalEmbedding


class PretrainedTranslationModelEncoder(PreTrainedModel):
    config_class = BasicTMEncoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class PretrainedTranslationModelDecoder(PreTrainedModel):
    config_class = BasicTMDecoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TranslationModel(nn.Module):
    def __init__(self,
                 hparams: Dict,
                 src_vocab_size: int,
                 src_padding_idx: int,
                 tgt_vocab_size: int,
                 tgt_padding_idx: int):

        super(TranslationModel, self).__init__()

        self.decoder_type = hparams['decoder_type']

        self.encoder = NMTEncoder(encoder_params=hparams['encoder'],
                                  src_vocab_size=src_vocab_size,
                                  src_padding_idx=src_padding_idx)

        if self.decoder_type == 'transformer_decoder':
            self.decoder = NMTDecoder(decoder_params=hparams['decoder'],
                                      tgt_vocab_size=tgt_vocab_size,
                                      tgt_padding_idx=tgt_padding_idx)
        elif self.decoder_type == 'gpt2':
            raise NotImplementedError('TODO: need to implement gpt2')

        else:
            raise ValueError('make sure hparams["decoder_type"] == transformer_decoder or gpt2')

        self.init_weights()

    def forward(self,
                src_sentences: torch.Tensor,
                src_mask: Optional[torch.Tensor],
                src_key_padding_mask: Optional[torch.Tensor],
                tgt_sentences: torch.Tensor,
                tgt_mask: Optional[torch.Tensor],
                encoded_output_mask: Optional[torch.Tensor],
                tgt_key_padding_mask: Optional[torch.Tensor],
                encoded_output_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        encoded_outputs = self.encoder(src_sentences=src_sentences,
                               src_mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)

        decoded = self.decoder(tgt_sentences=tgt_sentences,
                               encoded_output=encoded_outputs,
                               tgt_mask=tgt_mask,
                               encoded_output_mask=encoded_output_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               encoded_output_key_padding_mask=encoded_output_key_padding_mask)

        return decoded

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float=0.1,
                 max_len: int=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class NMTEncoder(PretrainedTranslationModelEncoder, nn.Module):
    def __init__(self,
                 encoder_params: Dict,
                 src_vocab_size: int,
                 src_padding_idx: int):

        super(NMTEncoder, self).__init__()

        self.encoder_params = encoder_params
        self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size,
                                          embedding_dim=self.encoder_params['embedding_dim'],
                                          padding_idx=src_padding_idx)

        self.pos_encoding = PositionalEncoding(d_model=self.encoder_params['hidden_size'],
                                               dropout=self.encoder_params['pos_encoding_dropout'])

        self.encoder_layer = TransformerEncoderLayer(d_model=self.encoder_params['hidden_size'],
                                                     nhead=self.encoder_params['nheads'],
                                                     dim_feedforward=self.encoder_params['ff_size'],
                                                     dropout=self.encoder_params['dropout'])

        self.layer_norm = nn.LayerNorm(normalized_shape=self.encoder_params['hidden_size'],
                                       eps=1e-6)

        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                          num_layers=self.encoder_params['nlayers'],
                                          norm=self.layer_norm)

    def forward(self,
                src_sentences: torch.Tensor,
                src_mask: Optional[torch.Tensor],
                src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        embedded = self.src_embedding(input=src_sentences)
        positional_encoded = self.pos_encoding(x=embedded)

        #Transformers don't have batch first
        positional_encoded = positional_encoded.permute(1,0,2)

        encoded = self.encoder(src=positional_encoded,
                               mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)
        return encoded

class NMTDecoder(PretrainedTranslationModelDecoder, nn.Module):
    def __init__(self,
                 decoder_params: Dict,
                 tgt_vocab_size: int,
                 tgt_padding_idx: int):

        super(NMTDecoder, self).__init__()
        self.decoder_params = decoder_params

        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size+1,
                                          embedding_dim=self.decoder_params['embedding_dim'],
                                          padding_idx=tgt_padding_idx)

        self.pos_encoding = PositionalEncoding(d_model=self.decoder_params['hidden_size'],
                                               dropout=self.decoder_params['pos_encoding_dropout'])

        self.decoder_layer = TransformerDecoderLayer(d_model=self.decoder_params['hidden_size'],
                                                     nhead=self.decoder_params['nheads'],
                                                     dim_feedforward=self.decoder_params['ff_size'],
                                                     dropout=self.decoder_params['dropout'])

        self.layer_norm = nn.LayerNorm(normalized_shape=self.decoder_params['hidden_size'],
                                       eps=1e-6)

        self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer,
                                          num_layers=self.decoder_params['nlayers'],
                                          norm=self.layer_norm)

        self.latent2vocab = nn.Linear(self.decoder_params['hidden_size'],
                                      tgt_vocab_size)

        self.drop = nn.Dropout(p=self.decoder_params['fc_dropout'])


    def forward(self,
                tgt_sentences: torch.Tensor,
                encoded_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor],
                encoded_output_mask: Optional[torch.Tensor],
                tgt_key_padding_mask: Optional[torch.Tensor],
                encoded_output_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        embedded = self.tgt_embedding(input=tgt_sentences)
        positional_encoded = self.pos_encoding(x=embedded)

        if tgt_mask is None:
            tgt_mask = self.generate_square_autoregressive_mask(embedded.size(1))
            #tgt_mask = tgt_mask.unsqueeze(0).repeat(embedded.size(0), 1, 1)

        #Transformers don't have batch first
        positional_encoded = positional_encoded.permute(1,0,2)

        decoded = self.decoder(tgt=positional_encoded,
                               memory=encoded_output,
                               tgt_mask=tgt_mask,
                               memory_mask=encoded_output_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=encoded_output_key_padding_mask)

        output = self.latent2vocab(self.drop(decoded))
        return output

    def generate_square_autoregressive_mask(self,
                                            sequence_length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length)) == 1
        return mask.bool()

#from transformers import T5Config, EncoderDecoderConfig, EncoderDecoderModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#decoder = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#config = EncoderDecoderConfig.from_encoder_decoder_configs(T5Config(), decoder.config)
#model = EncoderDecoderModel(config=config)
#model.decoder = decoder
#print(model)