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

from pt_fewshot.pet.utils import InputFeatures, InputExample, get_verbalization_ids, chunks, trim_input_ids, remove_final_punc, \
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

        prediction_scores = self.wrapper.model(**inputs)[0].view(-1, self.wrapper.model.model.config.vocab_size)
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
        input_ids = trim_input_ids(batch['input_ids'], num_masks=num_masks,
                                   pad_token_id=self.wrapper.tokenizer.pad_token_id,
                                   mask_token_id=self.wrapper.tokenizer.mask_token_id)

        log_probabilities = []
        original_batch = {}
        while True:
            masks = [(idx, tok_id) for idx, tok_id in enumerate(target_sequence[0]) if tok_id != -100]
            if not masks:  # there are no masks left to process, we are done
                break

            original_batch["input_ids"] = input_ids
            original_batch["attention_mask"] = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long).cuda()
            original_batch["block_flag"] = batch["block_flag"]
            inputs = self.wrapper.generate_default_inputs(original_batch)

            outputs = self.wrapper.model(**inputs)
            next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]

            mask_pos, masked_id = None, None
            max_prob = None
            for m_pos, m_id in masks:
                m_prob = next_token_logits[m_pos][m_id].item()
                if max_prob is None or m_prob > max_prob:
                    max_prob = m_prob
                    mask_pos, masked_id = m_pos, m_id

            log_probabilities.append(math.log(max_prob))
            input_ids[0][mask_pos] = masked_id
            target_sequence[0][mask_pos] = -100

        return sum(log_probabilities)


    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:

        mask_start = input_features.input_ids.index(self.wrapper.tokenizer.mask_token_id)

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

        mask_start = input_features.input_ids.index(self.wrapper.tokenizer.mask_token_id)
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

        # 为当前这批 features 中的每个样本分配一个唯一的整数 ID（target_id），并存入 feature_dict
        # feature_dict = {
        #             'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
        #             'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        #             'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        #             'labels': torch.tensor([f.label for f in features], dtype=torch.long),
        #             'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
        #             'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
        #             'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
        #             'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long)
        #         }
        # 应该叫 feature_id：代表该 feature 在数据集中的全局索引
        # len(self.id_to_target) 是已处理过的批次的 feature 的最大索引，在此基础上计算当前批次 feature 的全局 id
        feature_dict['target_id'] = torch.tensor([len(self.id_to_target) + idx for idx, f in enumerate(features)],
                                                 dtype=torch.long)
        # self.id_to_target 包含整个数据集 feature 的 target：
        #   列表的索引为 feature_id（feature 的全局索引），对应该 feature 的 target 文本
        self.id_to_target += [f.meta['target'] for f in features]
        feature_dict['target_token_ids'] = torch.tensor([f.meta['target_token_ids'] for f in features],
                                                        dtype=torch.long)



    def train_step(self, batch, **kwargs) -> Optional[torch.Tensor]:

        inputs = self.wrapper.generate_default_inputs(batch)
        inputs['labels'] = batch['target_token_ids']
        loss = self.wrapper.model(**inputs)[0]
        return loss


    def eval_step(self, batch: Dict[str, torch.Tensor], decoding_strategy: str = 'default', **kwargs):
        """
        给定一个包含多个 [MASK] token 的输入（如 T5 风格的 prompt），逐步用模型预测最可能的词替换每个 [MASK]，
          直到所有 mask 被填满，然后将填充结果与标准答案比较，判断是否正确。

        Notes:
              解码时一般批次中只有一个样本，并且通过 assert batch['input_ids'].shape[0] == 1 确保批次中只有一个样本，
                故该方法中的操作都是针对单样本的，比如：input_ids[0]、batch["target_id"][0]
        Args:
            batch: featuredict
            feature_dict = {
                'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
                'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                'labels': torch.tensor([f.label for f in features], dtype=torch.long),
                'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
                'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
                'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
                'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long),
                'target_id': tensor([...]),
                'target_token_ids': tensor([...]),
            }
            decoding_strategy:
            **kwargs:

        Returns:
            一个形状为 [1, 2] 的张量，表示预测结果（[[wrong, correct]]）

        """

        assert batch['input_ids'].shape[0] == 1, 'eval_step() for COPA is only implemented for batch_size=1'

        # 将 input_ids 提取出来用于填充 mask
        input_ids = batch["input_ids"]
        origin_batch = batch

        # 记录原始输入中所有 [MASK] token 的位置索引：这些位置是最终要提取预测词的位置，用于与标准答案比较
        orig_mask_positions = [
            idx for idx, input_id in enumerate(input_ids[0]) if input_id == self.wrapper.tokenizer.mask_token_id
        ]

        # 开始迭代填空循环：只要还有 [MASK]，就继续预测并填充
        while True:
            # 当前还剩哪些 mask：每次循环都重新查找当前剩余的 [MASK] 位置（因为会不断填充）
            mask_positions = [
                idx for idx, input_id in enumerate(input_ids[0]) if input_id == self.wrapper.tokenizer.mask_token_id
            ]
            # 没有剩余的 MASK 标记，进行结果对比并退出循环
            if not mask_positions:  # there are no masks left to process, we are done
                # 处理模型输入：提取填充后的 MASK 位置的 token，解码成文本
                # 将最终的 input_ids [1, seq_len] 转为 Python 列表（[seq_len]），便于后续处理
                input_ids = input_ids[0].detach().cpu().tolist()

                # 构建实际输出 output_actual:
                #   - 只取原始 mask 位置的预测 token id，忽略其他部分,
                #   - 排除特殊 token（如 [PAD], [CLS] 等）；
                #   - 用 tokenizer 将这些 ID 解码为字符串（如 "store"）。
                output_actual = self.wrapper.tokenizer.decode([
                    input_id for idx, input_id in enumerate(input_ids)
                    if idx in orig_mask_positions and input_id not in self.wrapper.tokenizer.all_special_ids
                ])
                # 获取标准答案：
                #   - 从 id_to_target 映射中获取 标准答案字符串（如 "grocery store"）。
                #   - batch["target_id"] 是当前批次 feature 的全局 id
                #     [0]：因为验证阶段每批次只有一个 feature，所以取该 feature 的全局 id
                #   - id_to_target[feature_id]: 得到该 feature 对应的 target 文本
                output_expected = self.id_to_target[batch["target_id"][0].item()]

                # 与标准答案比较（宽松匹配）：避免因大小写、标点、冠词（a/the）导致误判。
                # 标准化处理：转小写、去空格、按非字母字符分割成单词列表（比如 "hello, world!" 会变成 ["hello", "world"]）。
                # transform both outputs as described in the T5 paper
                output_actual = output_actual.lower().strip()
                output_actual = [w for w in re.split('[^a-zA-Z]', output_actual) if w]
                output_expected = output_expected.lower().strip()
                output_expected = [w for w in re.split('[^a-zA-Z]', output_expected) if w]

                # 判断：是否互为子集（任意方向）
                # 比对预测和标准答案：只要一方的所有单词都包含在另一方中，就判定为正确
                # 比对规则：只要预测的所有单词都在标准答案中，或标准答案的所有单词都在预测中，就判定为正确。
                # compare outputs
                if all(x in output_expected for x in output_actual) or all(
                        x in output_actual for x in output_expected):
                    return torch.tensor([[0, 1]]) # 正确返回 [0,1] （错误数 0，正确数 1）
                return torch.tensor([[1, 0]]) # 错误返回 [1,0]

            # 还有未填充的 MASK，继续生成
            origin_batch["input_ids"] = input_ids # 更新批次的 input_ids 为当前待填充的版本
            inputs = self.wrapper.generate_default_inputs(origin_batch) # 生成模型所需的输入（如 attention_mask 等）

            # 模型前向推理，获取每个位置的 token 预测概率
            outputs = self.wrapper.model(**inputs)
            next_token_logits = outputs[0]
            next_token_logits = torch.nn.Softmax(dim=2)(next_token_logits) # 对最后一维做 softmax，转成概率
            next_token_logits = next_token_logits[0].detach().cpu().numpy() # 转成 numpy 数组，方便后续计算

            # 找出所有剩余 MASK 位置中，模型最有信心（概率最高）的那个位置和对应的 token
            most_confident = ()
            most_confident_score = -1
            for mask_position in mask_positions:
                ntl = next_token_logits[mask_position] # 当前 MASK 位置的所有 token 概率
                top_token_id = np.argmax(ntl) # 概率最高的 token id
                top_score = ntl[top_token_id] # 对应的概率值

                # 更新最有信心的位置和 token
                if top_score > most_confident_score:
                    most_confident_score = top_score
                    most_confident = (mask_position, top_token_id)

            # 用模型最有信心的 token 填充该 MASK 位置，进入下一次循环
            input_ids[0][most_confident[0]] = most_confident[1]