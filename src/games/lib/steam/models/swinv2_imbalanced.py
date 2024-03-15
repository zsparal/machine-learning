import torch

from transformers import Swinv2ForImageClassification, Swinv2Config
from transformers.models.swinv2.modeling_swinv2 import Swinv2ImageClassifierOutput
from typing import Optional, Union
from torch.nn import BCEWithLogitsLoss


class Swinv2ForImbalancedImageClassification(Swinv2ForImageClassification):
    def __init__(self, config: Swinv2Config, label_weights: torch.Tensor):
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
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Swinv2ImageClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.swinv2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        # move labels to correct device to enable model parallelism
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.label_weights.to(logits.device)
            )
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Swinv2ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
