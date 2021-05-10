import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration
# from transformers.models.t5.modeling_t5 import shift_tokens_right

from utils import label_smoothed_nll_loss

class MyT5(T5ForConditionalGeneration):
    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_training=False,
        ):

        # if is_training:
        #     _decoder_input_ids = self._shift_right(decoder_input_ids)
        # else:
        #     _decoder_input_ids = decoder_input_ids

        if is_training:
            labels = decoder_input_ids
            _decoder_input_ids = None
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = T5ForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print(type(outputs))
        # print(outputs[0])

        # According to https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/modeling_t5.py
        # outputs[0] should be lm_logits

        lm_logits = outputs[1]
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
            # return outputs[0]
        return outputs
        # return (lm_logits, ) + outputs[1:]

