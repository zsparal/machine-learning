import torch

from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from typing import Optional, Union
from torch.nn import BCEWithLogitsLoss


class ConvNextV2ForImbalancedImageClassification(ConvNextV2ForImageClassification):
    def __init__(self, config: ConvNextV2Config, label_weights: torch.Tensor):
        super().__init__(config)
        self.label_weights = label_weights
        self.config.problem_type = "multi_label_classification"
        assert (
            self.num_labels > 0
        ), "This class should only be used for multi-label classification"
        assert self.num_labels == len(
            label_weights
        ), "Label weights must be the same length as the number of labels"

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.convnextv2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.label_weights.to(logits.device)
            )
            loss = loss_fct(logits, torch.Tensor(labels).to(logits.device))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
