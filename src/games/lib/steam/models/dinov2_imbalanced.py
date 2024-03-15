import torch

from transformers import Dinov2ForImageClassification, Dinov2Config
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union
from torch.nn import BCEWithLogitsLoss


class DinoV2ForImbalancedImageClassification(Dinov2ForImageClassification):
    """
    An extension of the DinoV2 model for multi-label, multi-class image classification that handles imbalanced datasets
    by applying label weights in the BCE loss function.
    """

    def __init__(self, config: Dinov2Config, label_weights: torch.Tensor):
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
    ) -> Union[tuple, ImageClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.dinov2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]

        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.classifier(linear_input)

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

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
