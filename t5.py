import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration
# from transformers.models.t5.modeling_t5 import shift_tokens_right

from utils import label_smoothed_nll_loss

class MyT5(T5ForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None,
            use_cache=False, is_training=False):

        if is_training:
            _decoder_input_ids = self._shift_right(decoder_input_ids)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = T5ForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
        )

        # According to https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/modeling_t5.py
        # outputs[0] should be lm_logits

        lm_logits = outputs[0]
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]

