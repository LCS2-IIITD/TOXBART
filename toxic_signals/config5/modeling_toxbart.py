from transformers import (
    BartPretrainedModel,
    BartConfig,
)

from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
import copy

from typing import Optional
import math


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class CoDA(nn.Module):
    def __init__(self,
                 dim_model: int,
                 multi_head: Optional[bool]=False,
                 num_heads: Optional[int]=1,
                 alpha: Optional[float]=1.0,
                 beta: Optional[float]=1.0,
                 scaling: Optional[bool]=False,
                 centering_E: Optional[bool]=False,
                 centering_N: Optional[bool]=False):
        super(CoDA, self).__init__()
        
        assert (
            dim_model % num_heads == 0
        ), f"dim_model must be divisible by num_heads (got `embed_dim`: {dim_model} and `num_heads`: {num_heads})."

        self.dim_model = dim_model
        self.multi_head = multi_head
        self.num_heads = num_heads
        
        if self.multi_head and self.num_heads > 1:
            self.head_dim = self.dim_model // self.num_heads
            self.query_transform = nn.Linear(self.dim_model, self.num_heads * self.head_dim, bias=False)
            self.key_transform = nn.Linear(self.dim_model, self.num_heads * self.head_dim, bias=False)
            self.value_transform = nn.Linear(self.dim_model, self.num_heads * self.head_dim, bias=False)
        else:
            self.query_transform = nn.Linear(self.dim_model, self.dim_model, bias=False)
            self.key_transform = nn.Linear(self.dim_model, self.dim_model, bias=False)
            self.value_transform = nn.Linear(self.dim_model, self.dim_model, bias=False)

        self.alpha = alpha
        self.beta = beta
        self.scaling = scaling
        self.centering_E = centering_E
        self.centering_N = centering_N
        
        self.fc = nn.Linear(self.dim_model, self.dim_model)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(self.dim_model)
    
    def coda_attention(self,
                       q: torch.Tensor, 
                       k: torch.Tensor,
                       v: torch.Tensor):  
        # KEY AND VALUE ARE FROM THE DIFFERENT MODALITY
        # QUERY REMAINS THE SAME
        if self.multi_head and self.num_heads > 1:
            E = torch.mul(self.alpha, torch.matmul(q, k.transpose(-2, -1)))
            if self.centering_E:
                E = E - torch.mean(E)

            N = None
            q = q.permute(1, 0, 2, 3)
            k = k.permute(1, 0, 2, 3)
            for head_q, head_k in list(zip(q, k)):
                head_N = torch.cdist(head_q, head_k, p=1)

                head_N = head_N.unsqueeze(0)
                if N is None:
                    N = head_N
                else:
                    N = torch.cat([N, head_N], dim=0)

            q = q.permute(1, 0, 2, 3)
            k = k.permute(1, 0, 2, 3)
            N = N.permute(1, 0, 2, 3)
            
            if self.centering_N:
                N = N - torch.mean(N)

            if self.scaling:
                E  = E / math.sqrt(k.shape[-1])
                N = N / math.sqrt(k.shape[-1])

            if self.centering_N:
                coda = torch.mul(F.tanh(E), F.sigmoid(N))
            else:
                coda = torch.mul(F.tanh(E), F.sigmoid(N))
            output = torch.matmul(coda, v)
            return output
        
        else:
            E = torch.mul(self.alpha, torch.matmul(q, k.transpose(-2, -1)))
            if self.centering_E:
                E = E - torch.mean(E)

            N = torch.mul(-self.beta, torch.cdist(q, k, p=1))
            if self.centering_N:
                N = N - torch.mean(N)

            if self.scaling:
                E  = E / math.sqrt(k.shape[-1])
                N = N / math.sqrt(k.shape[-1])

            if self.centering_N:
                coda = torch.mul(F.tanh(E), F.sigmoid(N))
            else:
                coda = torch.mul(F.tanh(E), F.sigmoid(N))
            output = torch.matmul(coda, v)
            return output

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor,
                value: torch.Tensor):
        
        batch_size = query.shape[0]
        residual = query
        
        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)

        if self.multi_head and self.num_heads > 1:
            query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            coda_output = self.coda_attention(q=query,
                                              k=key,
                                              v=value)
            coda_output = coda_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        else:
            coda_output = self.coda_attention(q=query,
                                              k=key,
                                              v=value)
        coda_output = self.fc(self.dropout(coda_output))
        coda_output = self.layer_norm(coda_output + residual)
        return coda_output

class ToxModelC4(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.coda = CoDA(dim_model=config.d_model, multi_head=False, num_heads=1)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids,
        attention_mask=None,
        toxic_input_ids=None,
        toxic_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        toxic_enc_outputs = self.encoder(
            input_ids=toxic_input_ids,
            attention_mask=toxic_attention_mask
        )
        
        if not self.training:
            toxic_enc_outputs["last_hidden_state"] = toxic_enc_outputs["last_hidden_state"].repeat(encoder_outputs["last_hidden_state"].shape[0]//toxic_enc_outputs["last_hidden_state"].shape[0], 1, 1)

        coda_outputs = self.coda(query = encoder_outputs["last_hidden_state"], key = toxic_enc_outputs["last_hidden_state"], value = toxic_enc_outputs["last_hidden_state"])
        encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"] + coda_outputs

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ToxBARTC4(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = ToxModelC4(config)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        toxic_input_ids=None,
        toxic_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):  
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        use_cache = self.config.use_cache
        return_dict = self.config.use_return_dict
        
        generation_outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            toxic_input_ids=toxic_input_ids,
            toxic_attention_mask=toxic_attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pre_logits = generation_outputs[0]
        generation_logits = self.lm_head(generation_outputs[0]) + self.final_logits_bias

        generation_masked_lm_loss = 0
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # paraphrase_masked_lm_loss = loss_fct(paraphrase_logits.view(-1, self.config.vocab_size), labels.view(-1))
            generation_masked_lm_loss = loss_fct(generation_logits.view(-1, self.config.vocab_size), labels.view(-1))

        final_loss = generation_masked_lm_loss if labels is not None else None
        
        return Seq2SeqLMOutput(
            loss=final_loss,
            logits=generation_logits,
            past_key_values=generation_outputs.past_key_values,
            decoder_hidden_states=generation_outputs.decoder_hidden_states,
            decoder_attentions=generation_outputs.decoder_attentions,
            cross_attentions=generation_outputs.cross_attentions,
            encoder_last_hidden_state=generation_outputs.encoder_last_hidden_state,
            encoder_hidden_states=generation_outputs.encoder_hidden_states,
            encoder_attentions=generation_outputs.encoder_attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "toxic_input_ids": torch.stack([torch.LongTensor(each) for each in kwargs["toxic_input_ids"]]).cuda(),
            "toxic_attention_mask": torch.stack([torch.LongTensor(each) for each in kwargs["toxic_attention_mask"]]).cuda(),
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past