# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch
import re

import numpy as np
from torch.nn import CrossEntropyLoss

from pet.utils import InputFeatures, InputExample, get_verbalization_ids, chunks, trim_input_ids, remove_final_punc, \
    lowercase_first


class TaskHelper(ABC):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.
    """

    def __init__(self, wrapper):
        """
        Create a new task helper.
        :param wrapper: The wrapper for the language model being used.
        """
        self.wrapper = wrapper
        self.output = None

    def train_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the train step for this task.
        :param batch: a batch of examples
        :return: a scalar loss tensor
        """
        pass

    def eval_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the eval step for this task.
        :param batch: a batch of examples
        :return: a tensor of logits
        """
        pass

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:
        """
        Add special features to the ``meta`` dictionary of a feature set
        :param input_example: the input example considered
        :param input_features: the set of features corresponding to this example
        """

        pass

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:
        """
        Add special features from the ``meta`` dictionary of a sequence of features to the corresponding dictionary
        :param features: the sequence of features
        :param feature_dict: the dictionary that stores aggregated feature views as tensors
        """
        pass

    def get_sequence_classifier_inputs(self, example: InputExample) -> Dict[str, Any]:
        """
        Get the inputs for sequence classification. Override this method if the input for the task considered is of a
        more complicated form than `text_a` or `text_a [SEP] text_b`.
        :param example: the input example
        :return: the dictionary of inputs
        """
        pass


class MultiRcTaskHelper(TaskHelper):
    """A custom task helper for the MultiRC dataset."""

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:
        input_features.meta['question_idx'] = input_example.meta['question_idx']

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:
        feature_dict['question_idx'] = torch.tensor([f.meta['question_idx'] for f in features], dtype=torch.long)


class CopaTaskHelper(TaskHelper):
    """A custom task helper for the COPA dataset."""

    def train_step(self, batch, **kwargs) -> Optional[torch.Tensor]:

        inputs = self.wrapper.generate_default_inputs(batch)
        mask = batch['labels'].unsqueeze(1)
        correct_targets = batch['choice1_token_ids'] * (1 - mask) + batch['choice2_token_ids'] * mask
        wrong_targets = batch['choice1_token_ids'] * mask + batch['choice2_token_ids'] * (1 - mask)

        prediction_scores = self.wrapper.model(**inputs)[0][..., self.wrapper.config.prompt_length:, :].reshape(-1,
                                                                                                                self.wrapper.model.model.config.vocab_size)
        loss_fct = CrossEntropyLoss()

        loss_correct_label = loss_fct(prediction_scores, correct_targets.view(-1))
        loss_wrong_label = loss_fct(prediction_scores, wrong_targets.view(-1))
        loss = 1 + loss_correct_label - loss_wrong_label
        loss[loss < 0] = 0
        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor], decoding_strategy: str = 'default', **kwargs):

        assert batch['input_ids'].shape[0] == 1, 'eval_step() for COPA is only implemented for batch_size=1'
        log_probs = []
        for choice in ['choice1', 'choice2']:
            labels = batch[f'{choice}_token_ids']
            log_prob = self._get_choice_log_probability(batch, labels, decoding_strategy=decoding_strategy)
            log_probs.append(log_prob)

        return torch.tensor([log_probs])

    def _get_choice_log_probability(self, batch, target_sequence, decoding_strategy: str = 'default'):
        # adjust the number of masks
        num_masks = sum(1 for tok_id in target_sequence[0] if tok_id != -100)
        # print(num_masks)
        # print(batch['input_ids'])

        input_ids = trim_input_ids(batch['input_ids'], num_masks=num_masks,
                                   pad_token_id=self.wrapper.tokenizer.pad_token_id,
                                   mask_token_id=self.wrapper.tokenizer.mask_token_id)
        # input_ids = batch['input_ids']
        # print(input_ids)
        # print(batch['input_ids'])
        # print(target_sequence)
        log_probabilities = []
        origin_batch = {}
        origin_batch['input_ids'] = input_ids
        origin_batch["attention_mask"] = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long).cuda()
        while True:
            masks = [(idx, tok_id) for idx, tok_id in enumerate(target_sequence[0]) if tok_id != -100]
            if not masks:  # there are no masks left to process, we are done
                break

            origin_batch["input_ids"] = input_ids
            # origin_batch["attention_mask"] = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long).cuda()
            inputs = self.wrapper.generate_default_inputs(origin_batch)
            # print(masks)
            outputs = self.wrapper.model(**inputs)
            next_token_logits = outputs[0][..., self.wrapper.config.prompt_length:, :]
            next_token_logits = torch.nn.Softmax(dim=2)(next_token_logits)
            # print(next_token_logits[0].size())
            mask_pos, masked_id = None, None
            max_prob = None
            for m_pos, m_id in masks:
                m_prob = next_token_logits[0][m_pos][m_id].item()
                if max_prob is None or m_prob > max_prob:
                    max_prob = m_prob
                    mask_pos, masked_id = m_pos, m_id
            max_prob += 1e-8
            log_probabilities.append(math.log(max_prob))
            input_ids[0][mask_pos] = masked_id
            target_sequence[0][mask_pos] = -100
        matrix = outputs[1][-1].squeeze(0).sum(dim=0).data.cpu().numpy()
        np.save("output.npy", matrix)
        return sum(log_probabilities)

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:

        mask_start = input_features.input_ids.index(
            self.wrapper.tokenizer.mask_token_id)
        # print(mask_start)
        for choice in ['choice1', 'choice2']:
            choice_text = input_example.meta[choice]
            choice_token_ids = get_verbalization_ids(choice_text, self.wrapper.tokenizer, force_single_token=False)
            mask_end = mask_start + len(choice_token_ids)
            input_features.meta[f'{choice}_token_ids'] = [-100] * len(input_features.input_ids)
            input_features.meta[f'{choice}_token_ids'][mask_start:mask_end] = choice_token_ids

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:
        for choice in ['choice1', 'choice2']:
            feature_dict[f'{choice}_token_ids'] = torch.tensor(
                [f.meta[f'{choice}_token_ids'] for f in features], dtype=torch.long)


class WscTaskHelper(TaskHelper):
    """A custom task helper for the Wsc dataset."""

    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.id_to_target = []

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:

        mask_start = input_features.input_ids.index(
            self.wrapper.tokenizer.mask_token_id)
        num_masks = input_features.input_ids.count(self.wrapper.tokenizer.mask_token_id)
        mask_end = mask_start + num_masks

        target = input_example.meta['span1_text']
        input_features.meta['target'] = target
        target_token_ids = get_verbalization_ids(target, self.wrapper.tokenizer, force_single_token=False)
        input_features.meta['target_token_ids'] = [-100] * len(input_features.input_ids)

        # we also predict <pad> tokens at the missing positions
        target_token_ids += [self.wrapper.tokenizer.pad_token_id] * (num_masks - len(target_token_ids))
        input_features.meta['target_token_ids'][mask_start:mask_end] = target_token_ids

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:

        feature_dict['target_id'] = torch.tensor([len(self.id_to_target) + idx for idx, f in enumerate(features)],
                                                 dtype=torch.long)
        self.id_to_target += [f.meta['target'] for f in features]
        feature_dict['target_token_ids'] = torch.tensor([f.meta['target_token_ids'] for f in features],
                                                        dtype=torch.long)

    def train_step(self, batch, **kwargs) -> Optional[torch.Tensor]:

        inputs = self.wrapper.generate_default_inputs(batch)
        labels = batch['target_token_ids']
        prediction_scores = self.wrapper.model(**inputs)[0][..., self.wrapper.config.prompt_length:, :].reshape(-1,
                                                                                                                self.wrapper.model.model.config.vocab_size)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(prediction_scores, labels.view(-1).contiguous())
        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor], decoding_strategy: str = 'default', **kwargs):

        assert batch['input_ids'].shape[0] == 1, 'eval_step() for COPA is only implemented for batch_size=1'
        num_masks = sum(1 for tok_id in batch['target_token_ids'][0] if tok_id != -100)
        input_ids = trim_input_ids(batch['input_ids'], num_masks=num_masks,
                                   pad_token_id=self.wrapper.tokenizer.pad_token_id,
                                   mask_token_id=self.wrapper.tokenizer.mask_token_id)
        print(input_ids)
        input_length = input_ids.size()[-1] - 1
        input_ids = batch["input_ids"]
        origin_batch = batch

        orig_mask_positions = [
            idx for idx, input_id in enumerate(input_ids[0]) if
            input_id == self.wrapper.tokenizer.mask_token_id
        ]

        while True:
            mask_positions = [
                idx for idx, input_id in enumerate(input_ids[0]) if input_id == self.wrapper.tokenizer.mask_token_id
            ]
            if not mask_positions:  # there are no masks left to process, we are done
                input_ids = input_ids[0].detach().cpu().tolist()
                output_actual = self.wrapper.tokenizer.decode([
                    input_id for idx, input_id in enumerate(input_ids)
                    if idx in orig_mask_positions and input_id not in self.wrapper.tokenizer.all_special_ids
                ])

                output_expected = self.id_to_target[batch["target_id"][0].item()]

                # transform both outputs as described in the T5 paper
                output_actual = output_actual.lower().strip()
                output_actual = [w for w in re.split('[^a-zA-Z]', output_actual) if w]
                output_expected = output_expected.lower().strip()
                output_expected = [w for w in re.split('[^a-zA-Z]', output_expected) if w]

                # compare outputs
                if all(x in output_expected for x in output_actual) or all(
                        x in output_actual for x in output_expected):
                    return torch.tensor([[0, 1]])
                return torch.tensor([[1, 0]])

            origin_batch["input_ids"] = input_ids
            # origin_batch["attention_mask"] = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long).cuda()
            inputs = self.wrapper.generate_default_inputs(origin_batch)

            outputs = self.wrapper.model(**inputs)
            next_token_logits = outputs[0][..., self.wrapper.config.prompt_length:, :]
            next_token_logits = torch.nn.Softmax(dim=2)(next_token_logits)
            next_token_logits = next_token_logits[0].detach().cpu().numpy()

            most_confident = ()
            most_confident_score = -1

            for mask_position in mask_positions:
                ntl = next_token_logits[mask_position]
                top_token_id = np.argmax(ntl)
                top_score = ntl[top_token_id]

                if top_score > most_confident_score:
                    most_confident_score = top_score
                    most_confident = (mask_position, top_token_id)

            input_ids[0][most_confident[0]] = most_confident[1]
            matrix = outputs[1][1].squeeze(0).sum(dim=0)[:self.wrapper.config.prompt_length,
                     self.wrapper.config.prompt_length:input_length + self.wrapper.config.prompt_length].data.cpu().numpy()
            print(matrix.shape)
            np.save("output.npy", matrix)


class RecordTaskHelper(TaskHelper):
    """A custom task helper for the ReCoRD dataset."""

    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.output = []
        self.original_choices = {}

    def train_step(self, batch, **kwargs) -> Optional[torch.Tensor]:
        inputs = self.wrapper.generate_default_inputs(batch)

        prediction_scores = self.wrapper.model(**inputs)[0][..., self.wrapper.config.prompt_length:, :].reshape(-1,
                                                                                                                self.wrapper.model.model.config.vocab_size)
        loss_fct = CrossEntropyLoss()

        # all_candidate_token_ids.shape() == batch_size x max_num_candidates x max_seq_len
        all_candidate_token_ids = batch['candidate_token_ids']

        # all_candidate_labels.shape() == batch_size x max_num_candidates
        all_candidate_labels = batch['candidate_labels']

        all_candidate_token_ids = all_candidate_token_ids.permute(1, 0, 2)
        all_candidate_labels = all_candidate_labels.permute(1, 0)

        total_loss = 0
        loss_correct_label = loss_fct(prediction_scores, all_candidate_token_ids[0].reshape(-1))

        # compute hinge loss
        for candidate_token_ids, candidate_labels in zip(all_candidate_token_ids[1:], all_candidate_labels[1:]):
            loss_wrong_label = loss_fct(prediction_scores, candidate_token_ids.reshape(-1))
            hinge_loss = 1 + loss_correct_label - loss_wrong_label
            hinge_loss[hinge_loss < 0] = 0
            total_loss += hinge_loss

        return total_loss

    def eval_step(self, batch: Dict[str, torch.Tensor], batch_size: int = 8, decoding_strategy: str = 'default'):
        assert batch['input_ids'].shape[0] == 1, "eval_step() for ReCoRD is only implemented for batch_size=1"

        best_choice_correct, best_choice, max_prob = False, None, None
        question_idx = batch['question_idx'][0].item()
        output_line = {'idx': question_idx, 'choices': {}}

        # group choices by length to speed up decoding
        choices_grouped_by_length = defaultdict(list)

        for idx, (choice_ids, label) in enumerate(zip(batch['candidate_token_ids'][0], batch['candidate_labels'][0])):
            if label < 0:
                continue
            num_masks = sum(1 for x in choice_ids if x != -100)
            choice = self.original_choices[question_idx][idx]
            choices_grouped_by_length[num_masks].append((choice, choice_ids, label))

        input_ids = {}
        initial_outputs = {}
        origin_batch = {}
        for num_masks in choices_grouped_by_length.keys():
            # modify the input ids to contain the correct number of masks
            input_ids[num_masks] = trim_input_ids(batch['input_ids'], num_masks=num_masks,
                                                  pad_token_id=self.wrapper.tokenizer.pad_token_id,
                                                  mask_token_id=self.wrapper.tokenizer.mask_token_id)
            # print(input_ids[num_masks].size())
            origin_batch['input_ids'] = input_ids[num_masks]
            origin_batch["attention_mask"] = torch.tensor([[1] * len(input_ids[num_masks][0])], dtype=torch.long).cuda()
            inputs = self.wrapper.generate_default_inputs(origin_batch)

            initial_outputs[num_masks] = self.wrapper.model(**inputs)

        for num_masks, choices_with_labels in choices_grouped_by_length.items():

            for batch in chunks(choices_with_labels, batch_size):
                batch_input_ids = input_ids[num_masks].repeat(len(batch), 1)
                choice_ids = torch.stack([choice_id for choice, choice_id, label in batch])

                probs = self._get_choice_probabilities_batched(choice_ids, batch_input_ids, initial_outputs[num_masks],
                                                               decoding_strategy=decoding_strategy)

                for idx, (choice, choice_ids, label) in enumerate(batch):
                    prob = probs[idx]
                    output_line['choices'][choice] = prob

                    if max_prob is None or prob > max_prob:
                        best_choice_correct, max_prob = (label == 1), prob

        self.output.append(output_line)

        if best_choice_correct:
            return torch.tensor([[0, 1]])
        return torch.tensor([[1, 0]])

    def _get_choice_probabilities_batched(self, target_sequences, input_ids, initial_output, decoding_strategy):

        log_probabilities = defaultdict(list)
        first_call = True
        original_batch = {}
        original_batch['input_ids'] = input_ids
        # print(input_ids.size())
        # original_batch["attention_mask"] = torch.tensor([[1] * len(input_ids)], dtype=torch.long).cuda()
        # print(original_batch["attention_mask"].size())
        while True:
            masks = {batch_idx: [(idx, tok) for idx, tok in enumerate(target_sequences[batch_idx]) if tok >= 0] for
                     batch_idx in range(len(target_sequences))}

            if not masks[0]:  # there are no masks left to process, we are done
                break
            original_batch['input_ids'] = input_ids
            if first_call:
                outputs = initial_output
                # next_token_logits = outputs[0]
            else:
                inputs = self.wrapper.generate_default_inputs(original_batch)
                outputs = self.wrapper.model(**inputs)
            next_token_logits = outputs[0][..., self.wrapper.config.prompt_length:, :]
            next_token_logits = torch.nn.Softmax(dim=2)(next_token_logits)

            for batch_idx in range(len(target_sequences)):

                ntl = next_token_logits[batch_idx] if not first_call else next_token_logits[0]

                if decoding_strategy == 'parallel':
                    for m_pos, m_id in masks[batch_idx]:
                        log_probabilities[batch_idx].append(math.log(ntl[m_pos][m_id].item()))
                        target_sequences[batch_idx][m_pos] = -100

                else:
                    mask_pos, masked_id = None, None
                    highest_prob = None
                    for m_pos, m_id in masks[batch_idx]:
                        m_prob = ntl[m_pos][m_id]
                        if highest_prob is None or m_prob > highest_prob:
                            highest_prob = m_prob
                            mask_pos, masked_id = m_pos, m_id

                    log_probabilities[batch_idx].append(math.log(ntl[mask_pos][masked_id].item()))
                    input_ids[batch_idx][mask_pos] = masked_id
                    target_sequences[batch_idx][mask_pos] = -100

            first_call = False

        return {batch_idx: sum(log_prob for log_prob in log_probabilities[batch_idx]) for batch_idx in
                range(len(target_sequences))}

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:
        mask_start = input_features.input_ids.index(self.wrapper.tokenizer.mask_token_id)

        choices = input_example.meta['candidates']
        question_idx = input_example.meta['question_idx']

        input_features.meta['candidate_token_ids'] = []
        input_features.meta['candidate_labels'] = []
        input_features.meta['question_idx'] = question_idx

        self.original_choices[question_idx] = []

        for idx, choice_text in enumerate(choices):
            choice_token_ids = get_verbalization_ids(choice_text, self.wrapper.tokenizer, force_single_token=False)
            choice_label = 1 if choice_text in input_example.meta['answers'] else 0

            mask_end = mask_start + len(choice_token_ids)
            candidate_token_ids = [-100] * len(input_features.input_ids)
            candidate_token_ids[mask_start:mask_end] = choice_token_ids

            input_features.meta['candidate_token_ids'].append(candidate_token_ids)
            input_features.meta['candidate_labels'].append(choice_label)
            self.original_choices[question_idx].append(choice_text)

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:
        # apply padding if necessary
        max_num_candidates = max(len(f.meta['candidate_token_ids']) for f in features)
        for feature in features:
            while len(feature.meta['candidate_token_ids']) < max_num_candidates:
                feature.meta['candidate_token_ids'].append([-100] * len(feature.input_ids))
                feature.meta['candidate_labels'].append(-100)

        feature_dict['candidate_token_ids'] = \
            torch.tensor([f.meta['candidate_token_ids'] for f in features], dtype=torch.long)
        feature_dict['candidate_labels'] = \
            torch.tensor([f.meta['candidate_labels'] for f in features], dtype=torch.long)

        feature_dict['question_idx'] = torch.tensor([f.meta['question_idx'] for f in features], dtype=torch.long)
