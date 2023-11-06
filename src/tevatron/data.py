import numpy as np
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Callable, Dict, NewType, Union
from random import random, shuffle, choice, sample, randrange, Random
import collections

import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import TevatronTrainer

import json
import logging
logger = logging.getLogger(__name__)


def load_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def truncate_seq(tokens_a, tokens_b, max_num_tokens):
    """
    tokens_a: text域
    tokens_b: 模板域
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        tokens_a.pop()

def padding_for_max_len(token_lists):
    max_len = max(len(tokens) for tokens in token_lists)  # 最长长度
    padding_token_lists = []
    masks = []
    for tokens in token_lists:
        padded_tokens = torch.zeros(max_len, dtype=torch.long)
        mask = torch.zeros(max_len, dtype=torch.long)

        padded_tokens[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mask[:len(tokens)] = 1

        padding_token_lists.append(padded_tokens)
        masks.append(mask)
    padding_token_lists = torch.stack(padding_token_lists)
    masks = torch.stack(masks)
    return padding_token_lists,  masks     


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    tokens = tokens.copy()
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[SEP]":
            continue
        if token == "[CLS]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(
        1, int(round(len(cand_indices) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()

    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index]))  # 存储原始label
            tokens[index] = masked_token  # 替换为被mask后的

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]
    return tokens, mask_indices, masked_token_labels


def generate_for_prompt(tokenizer, new_aspect_template, prompt_dict, text_dict, max_seq_len):
    pos_list = []  # [[start_pos, end_pos, aspect_name], [..], ..] 存放各个aspect 的位置
    for a_name, txt in text_dict.items():
        if txt == "":
            new_aspect_template = new_aspect_template.replace(
                prompt_dict[a_name], '')
    for a_name, txt in text_dict.items():
        if txt != "":
            pos_list.append([new_aspect_template.index(prompt_dict[a_name]), new_aspect_template.index(
                prompt_dict[a_name])+len(prompt_dict[a_name]), a_name])
    # 按开始位置排序, 方便从前开始替换
    pos_list = sorted(pos_list, key=lambda x: x[0])

    last_idx = 0  # 待tokenize的开始idx

    all_tokenize_list = []  # 按顺序预存储tokenize后的文本
    all_type_list = []
    type_id = {  # 转为数字
        'text': 0,
        'prompt': 1,
        'aspect': 2
    }

    for pos_ in pos_list:
        # 模板，不进行mlm
        before_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
            new_aspect_template[last_idx:pos_[0]]))
        if len(before_token) != 0:
            all_type_list.append(type_id['prompt'])
            all_tokenize_list.append(before_token)

        mask_content = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
            text_dict[pos_[2]]))  # slot 内容填进去

        if pos_[2] == 'text':  # 不需要mask
            # 针对纯文本进行mlm
            # 由于text出现在模板中间，需要提前对过长的text进行裁剪
            truncate_seq(mask_content, [], max_seq_len)

            if len(mask_content) != 0:
                all_type_list.append(type_id['text'])
                all_tokenize_list.append(mask_content)

        else:
            if len(mask_content) != 0:
                all_type_list.append(type_id['aspect'])
                all_tokenize_list.append(mask_content)

        last_idx = pos_[1]

    # 多余的模板
    if pos_list != []:
        end_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
            new_aspect_template[pos_[1]:]))
        if len(end_token) != 0:
            all_type_list.append(type_id['prompt'])
            all_tokenize_list.append(end_token)
    features = {
        'all_tokenize_list': all_tokenize_list,
        'all_type_list': all_type_list
    }
    return features


def merge_dict(dict_list):
    all_dict = {}
    for num in range(len(dict_list)):
        cate_dict = dict_list[num]
        cate_dict = sorted(cate_dict.items(), key=lambda x: x[1])  # 根据id排序，固定遍历次序
        dict_len = len(cate_dict)
        idx = 0
        for cate, cate_id in cate_dict:
            assert idx == cate_id
            idx += 1
            if cate not in all_dict:
                all_dict[cate] = len(all_dict)
        assert dict_len == idx
    return all_dict


class PretrainPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' ', data_args=None):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.cat_level = data_args.cat_level  # 用多少级的cat
        self.separator = separator
        self.data_args = data_args
        self.bert_vocab_list = list(tokenizer.vocab.keys())
        self.is_concat = data_args.is_concat  # 默认 false ，不拼接多aspect
        if data_args.brand2id_path is not None:  # 加载有监督字典
            self.brand2id_dict = load_dict(data_args.brand2id_path)
        else:
            self.brand2id_dict = None
        if data_args.color2id_path is not None:
            self.color2id_dict = load_dict(data_args.color2id_path)
        else:
            self.color2id_dict = None

        if data_args.cate1_2id_path is not None:
            self.cate1_2id_dict = load_dict(data_args.cate1_2id_path)
        else:
            self.cate1_2id_dict = None

        if data_args.cate2_2id_path is not None:
            self.cate2_2id_dict = load_dict(data_args.cate2_2id_path)
        else:
            self.cate2_2id_dict = None

        if data_args.cate3_2id_path is not None:
            self.cate3_2id_dict = load_dict(data_args.cate3_2id_path)
        else:
            self.cate3_2id_dict = None

        if data_args.cate4_2id_path is not None:
            self.cate4_2id_dict = load_dict(data_args.cate4_2id_path)
        else:
            self.cate4_2id_dict = None

        if data_args.cate5_2id_path is not None:
            self.cate5_2id_dict = load_dict(data_args.cate5_2id_path)
        else:
            self.cate5_2id_dict = None
        
        self.pretrain_type = data_args.pretrain_type
        if self.pretrain_type in ['prompt']:
            self.cate_word_vocab_dict = tokenizer.get_vocab()  # 原始词典
        elif self.pretrain_type in ['mgrain']:
            self.cate_wordpiece_dict = load_dict(data_args.cate_wordpiece_vocab_file)  # wordpiece 词典
            self.cate_word_dict = load_dict(data_args.cate_word_vocab_file)
            self.cate_wholeword_dict = load_dict(data_args.whole_cate_vocab_file)

            self.brand_wordpiece_dict = load_dict(data_args.brand_wordpiece_vocab_file)  # wordpiece 词典
            self.brand_word_dict = load_dict(data_args.brand_word_vocab_file)
            self.brand_wholeword_dict = load_dict(data_args.whole_brand_vocab_file)

            self.color_wordpiece_dict = load_dict(data_args.color_wordpiece_vocab_file)  # wordpiece 词典
            self.color_word_dict = load_dict(data_args.color_word_vocab_file)
            self.color_wholeword_dict = load_dict(data_args.whole_color_vocab_file)

        dict_list = [
            self.cate1_2id_dict,
            self.cate2_2id_dict,
            self.cate3_2id_dict,
            self.cate4_2id_dict,
            self.cate5_2id_dict,
        ]
        self.all_cate_dict = merge_dict(dict_list[:self.cat_level])
        # self.all_cate_dict = load_dict(data_args.whole_cate_vocab_file)

        # self.mgrain_cate = 9
        self.mgrain_cate_level = self.cat_level  # 调试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        print('=======mgrain_cate_level,', self.mgrain_cate_level)

    def __call__(self, example):
        text_max_length = self.text_max_length
        title = example['product_title']
        description = example['product_description']
        bullet_point = example['product_bullet_point']

        text = title + '[SEP]' + description + '[SEP]' + bullet_point

        brand = example['product_brand'].lower()
        color = example['product_color'].lower()
        cate1_text = example['product_cate'][0] if len(
            example['product_cate']) >= 1 else ''
        cate2_text = example['product_cate'][1] if len(
            example['product_cate']) >= 2 else ''
        cate3_text = example['product_cate'][2] if len(
            example['product_cate']) >= 3 else ''
        cate4_text = example['product_cate'][3] if len(
            example['product_cate']) >= 4 else ''
        cate5_text = example['product_cate'][4] if len(
            example['product_cate']) >= 5 else ''

        cat = ','.join(example['product_cate'][:self.cat_level]).strip(',')

        aspect_text = {
            'p_brand': brand,
            'p_color': color,
            'p_cate1': cate1_text,
            'p_cate2': cate2_text,
            'p_cate3': cate3_text,
            'p_cate4': cate4_text,
            'p_cate5': cate5_text,
        }

        aspect_fc = {
            'p_brand': self.brand2id_dict,
            'p_color': self.color2id_dict,
            'p_cate1': self.cate1_2id_dict,
            'p_cate2': self.cate2_2id_dict,
            'p_cate3': self.cate3_2id_dict,
            'p_cate4': self.cate4_2id_dict,
            'p_cate5': self.cate5_2id_dict,
        }
        aspect_id = {}
        for apsect_name, aspect_txt in aspect_text.items():
            aspect_id[apsect_name] = aspect_fc[apsect_name][''] \
                if aspect_txt not in aspect_fc[apsect_name] else aspect_fc[apsect_name][aspect_txt]

        # cate_mul_label = np.zeros(len(self.all_cate_dict))
        for_cate_mul = [str(len(self.all_cate_dict))]
        for cate_text in example['product_cate'][:self.cat_level]:
            if cate_text in self.all_cate_dict and cate_text != '':
                mul_id = self.all_cate_dict[cate_text]
                for_cate_mul.append(str(mul_id))
        string_cate_mul = '#'.join(for_cate_mul)  # 第一个是长度
        aspect_id['string_cate_mul'] = string_cate_mul

        if self.pretrain_type in ['wordpredict', 'wordpredict_no', 'wordpredict_shared_dynamic']:
            for_cate_vocab = [str(len(self.cate_word_vocab_dict))]
            regex = r'[^\w\s]|[\s]'  # 空格和标点符号切分
            for cate_text in example['product_cate']:
                words = re.split(regex, cate_text)
                for word in words:
                    if word.strip():
                        if word.strip() in self.cate_word_vocab_dict:
                            word_id = self.cate_word_vocab_dict[word.strip()]  # 所有的词都在词典中出现
                        else:
                            word_id =  self.cate_word_vocab_dict['UNK']  # 预留的UNK
                            raise ValueError('not exist word in category vocab!')
                        for_cate_vocab.append(str(word_id))
            string_cate_vocab = '#'.join(for_cate_vocab)  # 第一个是长度
            aspect_id['p_cate_vocab'] = string_cate_vocab
        elif self.pretrain_type in ['prompt']:
            for_cate_vocab = [str(len(self.cate_word_vocab_dict))]
            for cate_text in example['product_cate']:
                token_ids = self.tokenizer.encode(cate_text)[1:-1]  # 在原始分词器中对应的token id list，去掉cls和sep
                for_cate_vocab.extend([str(i) for i in token_ids])
            string_cate_vocab = '#'.join(for_cate_vocab)  # 第一个是长度
            aspect_id['p_cate_vocab'] = string_cate_vocab
        
        elif self.pretrain_type in ['mgrain']:
            regex = r'[^\w\s]|[\s]'  # 空格和标点符号切分

            # wordpiece
            for_cate_wordpiece = [str(len(self.cate_wordpiece_dict))]
            for cate_text in example['product_cate'][:self.mgrain_cate_level]:
                words = re.split(regex, cate_text)
                for word in words:
                    if word.strip():
                        for w in  self.tokenizer.tokenize(word):
                            for_cate_wordpiece.append(str(self.cate_wordpiece_dict[w]))
            string_cate_vocab = '#'.join(for_cate_wordpiece)  # 第一个是长度
            aspect_id['p_cate_wordpiece'] = string_cate_vocab

            # wordpiece
            for_brand_wordpiece = [str(len(self.brand_wordpiece_dict))]
            for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:
                words = re.split(regex, brand_text)
                for word in words:
                    if word.strip():
                        for w in  self.tokenizer.tokenize(word):
                            for_brand_wordpiece.append(str(self.brand_wordpiece_dict[w]))
            string_brand_vocab = '#'.join(for_brand_wordpiece)  # 第一个是长度
            aspect_id['p_brand_wordpiece'] = string_brand_vocab

            # wordpiece
            for_color_wordpiece = [str(len(self.color_wordpiece_dict))]
            for color_text in [color if color in aspect_fc['p_color'] else '']:
                words = re.split(regex, color_text)
                for word in words:
                    if word.strip():
                        for w in  self.tokenizer.tokenize(word):
                            for_color_wordpiece.append(str(self.color_wordpiece_dict[w]))
            string_color_vocab = '#'.join(for_color_wordpiece)  # 第一个是长度
            aspect_id['p_color_wordpiece'] = string_color_vocab
            
            # word
            for_cate_word = [str(len(self.cate_word_dict))]
            for cate_text in example['product_cate'][:self.mgrain_cate_level]:
                words = re.split(regex, cate_text)
                for word in words:
                    if word.strip():
                        word_id = self.cate_word_dict[word.strip()]  # 所有的词都在词典中出现
                        for_cate_word.append(str(word_id))
            string_cate_vocab = '#'.join(for_cate_word)  # 第一个是长度
            aspect_id['p_cate_word'] = string_cate_vocab

            # word
            for_brand_word = [str(len(self.brand_word_dict))]
            for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:
                words = re.split(regex, brand_text)
                for word in words:
                    if word.strip():
                        word_id = self.brand_word_dict[word.strip()]  # 所有的词都在词典中出现
                        for_brand_word.append(str(word_id))
            string_brand_vocab = '#'.join(for_brand_word)  # 第一个是长度
            aspect_id['p_brand_word'] = string_brand_vocab

            # word
            for_color_word = [str(len(self.color_word_dict))]
            for color_text in [color if color in aspect_fc['p_color'] else '']:
                words = re.split(regex, color_text)
                for word in words:
                    if word.strip():
                        word_id = self.color_word_dict[word.strip()]  # 所有的词都在词典中出现
                        for_color_word.append(str(word_id))
            string_color_vocab = '#'.join(for_color_word)  # 第一个是长度
            aspect_id['p_color_word'] = string_color_vocab

            # wholeword
            for_cate_wholeword = [str(len(self.cate_wholeword_dict))]
            for cate_text in example['product_cate'][:self.mgrain_cate_level]:  # 完整的保持，不切词
                word_id = self.cate_wholeword_dict[cate_text]
                for_cate_wholeword.append(str(word_id))
            string_cate_vocab = '#'.join(for_cate_wholeword)  # 第一个是长度
            aspect_id['p_cate_wholeword'] = string_cate_vocab

            # wholeword
            for_brand_wholeword = [str(len(self.brand_wholeword_dict))]
            for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:  # 完整的保持，不切词
                word_id = self.brand_wholeword_dict[brand_text]
                for_brand_wholeword.append(str(word_id))
            string_brand_vocab = '#'.join(for_brand_wholeword)  # 第一个是长度
            aspect_id['p_brand_wholeword'] = string_brand_vocab

            # wholeword
            for_color_wholeword = [str(len(self.color_wholeword_dict))]
            for color_text in [color if color in aspect_fc['p_color'] else '']:  # 完整的保持，不切词
                word_id = self.color_wholeword_dict[color_text]
                for_color_wholeword.append(str(word_id))
            string_color_vocab = '#'.join(for_color_wholeword)  # 第一个是长度
            aspect_id['p_color_wholeword'] = string_color_vocab

        # example['text'] = text  # for debug

        if self.data_args.pretrain_type in ['mlm', 'condenser']:
            if self.is_concat == 'yes':
                text = '[unused0]' + cat \
                    + '[unused1]' + brand \
                    + '[unused2]' + color \
                    + '[SEP]' + '[unused3]' + title \
                    + '[SEP]' + description \
                    + '[SEP]' + bullet_point

            tokenized_text = self.tokenizer.encode(
                text, add_special_tokens=False, max_length=text_max_length, truncation=True)
            example['tokenized_text'] = tokenized_text
            return example
        elif self.data_args.pretrain_type == 'prompt':
            prompt_type = self.data_args.prompt_type  # 每次只预测一个

            if prompt_type == 'three-sep':
                aspect_template = '[unused0][text_a][unused1][text_b][unused2][text_c][SEP][unused3][text_d]'
                prompt_dict = {
                    'cat': '[text_a]',
                    'brand': '[text_b]',
                    'color': '[text_c]',
                    'text': '[text_d]',
                }  # 每个字段，在aspect_template中对应的占位符
                text_dict = {
                    'text': text,
                    'brand': brand,
                    'color': color,
                    'cat': cat,
                }
            features = generate_for_prompt(
                self.tokenizer, aspect_template, prompt_dict, text_dict, text_max_length)
            features['aspect_id'] = aspect_id
            return features

        elif self.data_args.pretrain_type in ['mtbert', 'madr', 'mgrain']:
            if self.is_concat == 'yes':
                text = '[unused0]' + cat \
                    + '[unused1]' + brand \
                    + '[unused2]' + color \
                    + '[SEP]' + '[unused3]' + title \
                    + '[SEP]' + description \
                    + '[SEP]' + bullet_point
            elif self.is_concat == 'first-aspect-num':  # 添加aspect num-1个token，
                text = '[unused0]' + '[unused1]' + '[unused2]' + '[SEP]' + '[unused3]' + title + '[SEP]' + description + '[SEP]' + bullet_point
            elif self.is_concat == 'first-aspect-num-nine':
                text = '[unused0]' + '[unused1]' + '[unused2]' + '[unused3]' + '[unused4]' + '[unused5]' + '[unused6]' + '[unused7]' + '[unused8]' \
                    + '[SEP]' + '[unused9]' + title + '[SEP]' + description + '[SEP]' + bullet_point
            elif self.is_concat == 'first-aspect-num-concat':
                text = '[unused4]' + '[unused5]' + '[unused6]'\
                    + '[unused0]' + brand + '[unused1]' + color + '[unused2]' + cat \
                     + '[SEP]' + '[unused3]' + title + '[SEP]' + description + '[SEP]' + bullet_point
            tokenized_text = self.tokenizer.encode(
                text, add_special_tokens=False, max_length=text_max_length, truncation=True)
            example['tokenized_text'] = tokenized_text
            example['aspect_id'] = aspect_id
            return example
        else:
            raise ValueError(
                f"pretrain_type ({self.data_args.pretrain_type}) is not valid!")


class HFPretrainDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_file
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = PretrainPreProcessor
        self.tokenizer = tokenizer
        self.max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(
            self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)
        self.data_args = data_args

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            remove_columns = []
            # remove_columns = self.dataset.column_names
            save_list = ['is_doc', 'doc_id']
            for i in save_list:
                if i in remove_columns:
                    remove_columns.remove(i)
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.max_len,
                                  self.separator, self.data_args),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=remove_columns,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class PretrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer
        self.data_args = data_args
        self.total_len = len(self.train_data)
        self.pretrain_type = data_args.pretrain_type

    def create_one_example(self, text_encoding: List[int], max_length, is_query=False):

        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        instance = self.train_data[item]
        max_length = self.data_args.p_max_len

        if self.pretrain_type in ['mlm', 'condenser']:
            tok_text = instance["tokenized_text"]
            inputs = self.create_one_example(tok_text, max_length=max_length)

            all_tokenize_list = inputs['input_ids'][1:-1]
            if len(all_tokenize_list) != 0:
                all_tokenize_list = [all_tokenize_list]
            return all_tokenize_list, [0]

        elif self.pretrain_type == 'prompt':
            s = {
                'all_tokenize_list': instance['all_tokenize_list'],
                'all_type_list': instance['all_type_list'],
            }
            return s['all_tokenize_list'], s['all_type_list'], instance['aspect_id']

        elif self.pretrain_type in ['mtbert', 'madr', 'mgrain']:
            tok_text = instance["tokenized_text"]
            inputs = self.create_one_example(tok_text, max_length=max_length)
            all_tokenize_list = inputs['input_ids'][1:-1]
            if len(all_tokenize_list) != 0:
                all_tokenize_list = [all_tokenize_list]
            return all_tokenize_list, [0], instance['aspect_id']
        else:
            raise ValueError(
                f"pretrain_type ({self.pretrain_type}) is not valid!")


@dataclass
class PreTrainCollator:
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    """

    def __init__(self, tokenizer, data_args, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.bert_vocab_list = list(tokenizer.vocab.keys())
        self.masked_lm_prob = data_args.mlm_probability
        self.aspect_mlm_prob = data_args.aspect_mlm_prob
        self.mask_aspect_num = data_args.mask_aspect_num
        # Maximum number of tokens to mask in each sequence
        self.max_predictions_per_seq = 60
        self.pad_to_multiple_of = pad_to_multiple_of
        self.p_max_len = data_args.p_max_len
        self.pretrain_type = data_args.pretrain_type
        self.multi_have_weight = data_args.multi_have_weight

    def handle_for_one_example(self, idx, tokenize_list, type_list, text_mlm_prob, aspect_mlm_prob):
        all_masked_pos = [[], []]
        all_masked_label = []

        max_prompt_length = self.p_max_len
        type_id = {  # 转为数字
            'text': 0,
            'prompt': 1,
            'aspect': 2
        }

        # batch逐个样本处理
        mask_seq = [101]  # 计算pos的时候需要加上cls
        all_masked_lm_positions = []
        all_masked_lm_labels = []

        for part in range(len(tokenize_list)):
            if type_list[part] == type_id['prompt']:
                # 模板不进行处理
                mask_seq += tokenize_list[part]
            elif type_list[part] == type_id['text']:
                mask_content = tokenize_list[part]
                tokens, masked_lm_positions, masked_lm_labels = self.torch_mask_tokens(
                    torch.tensor([mask_content]), mlm_prob=text_mlm_prob)
                # 计算送入 create_masked_lm_predictions 的tokens 在mask_seq中的开始idx
                begin_pos_idx = len(mask_seq)
                masked_lm_positions = [
                    i+begin_pos_idx for i in masked_lm_positions]
                all_masked_lm_positions.extend(masked_lm_positions)
                all_masked_lm_labels.extend(masked_lm_labels)
                mask_seq += tokens
            elif type_list[part] == type_id['aspect']:
                mask_content = tokenize_list[part]
                # 针对aspect 内容进行mlm，方便设置不同的mask 比例, masked_lm_prob=1 代表全部mask
                tokens, masked_lm_positions, masked_lm_labels = self.torch_mask_tokens(
                    torch.tensor([mask_content]), mlm_prob=aspect_mlm_prob)
                # 计算送入 create_masked_lm_predictions 的tokens 在mask_seq中的开始idx
                begin_pos_idx = len(mask_seq)
                masked_lm_positions = [
                    i+begin_pos_idx for i in masked_lm_positions]
                all_masked_lm_positions.extend(masked_lm_positions)
                all_masked_lm_labels.extend(masked_lm_labels)
                mask_seq += tokens
        mask_seq += [102]
        text = self.tokenizer.prepare_for_model(
            mask_seq[1:-1],
            truncation='only_first',
            max_length=max_prompt_length,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # 把被裁剪掉的 mask pos 去掉
        for i, pos in enumerate(all_masked_lm_positions):
            label = all_masked_lm_labels[i]
            if pos >= max_prompt_length:
                continue
            all_masked_label.append(label)
            all_masked_pos[0].append(idx)  # 第几个样本
            all_masked_pos[1].append(pos)  # 第几个token
        return text, all_masked_pos, all_masked_label

    def __call__(self, features):
        all_tokenize_list = [x[0] for x in features]  # len=123
        all_type_list = [x[1] for x in features]
        aspect_dict = None
        if len(features) != 0 and len(features[0]) >= 3:
            # ['p_brand', 'p_cat1', 'p_cat2'..]
            aspect_names = features[0][2].keys()
            aspect_dict = {}
            for _name in aspect_names:
                if _name == 'string_cate_mul':
                    cate_mul_all = []
                    for f in features:
                        string_cate_mul = f[2][_name]
                        cate_mul_info = string_cate_mul.split('#')
                        cate_mul_label = np.zeros(int(cate_mul_info[0]))
                        for cate_id in cate_mul_info[1:]:
                            cate_mul_label[int(cate_id)] = 1
                        cate_mul_all.append(cate_mul_label)
                    cate_mul_all = np.array(cate_mul_all)
                    aspect_dict['p_cate_mul'] = cate_mul_all
                elif _name in ['p_cate_vocab', 'p_cate_wordpiece', 'p_cate_word', 'p_cate_wholeword', \
                    'p_brand_wordpiece', 'p_brand_word', 'p_brand_wholeword',\
                    'p_color_wordpiece', 'p_color_word', 'p_color_wholeword']:
                    cate_vocab_all = []  # 每个样本的category的word id list
                    for f in features:
                        string_cate_vocab = f[2][_name]
                        cate_vocab_info = string_cate_vocab.split('#')
                        cate_vocab_label = np.zeros(int(cate_vocab_info[0]))
                        for word_id in cate_vocab_info[1:]:  # 根据出现次数给予权重
                            if self.multi_have_weight:
                                cate_vocab_label[int(word_id)] += 1  # 考虑出现的次数
                            else:
                                cate_vocab_label[int(word_id)] = 1  # 暂时不考虑出现的次数
                        cate_vocab_all.append(cate_vocab_label)
                    cate_vocab_all = np.array(cate_vocab_all)
                    aspect_dict[_name] = cate_vocab_all
                else:
                    aspect_dict[_name] = np.array(
                        [f[2][_name] for f in features], dtype=np.int64)
                    
        type_id = {  # 转为数字
            'text': 0,
            'prompt': 1,
            'aspect': 2
        }
        input_ids = []
        all_masked_pos = [[], []]
        all_masked_label = []
        all_aspect_id = {}
        max_prompt_length = self.p_max_len

        cnt = 0  # 样本数量

        for idx in range(len(all_tokenize_list)):
            # batch逐个样本处理
        
            tokenize_list = all_tokenize_list[idx]  # [[1], [12,321,342], [2],[23, 222, 33,22]]
            type_list = all_type_list[idx] # [1,2,1,0] -> [1,2,1,2,1,2,1,1,0]
 
            aspect_pos = np.where(np.array(type_list) == type_id['aspect'])[0]  # [1,2,1,2,1,2,1,1,0]
            if self.aspect_mlm_prob == 1.0:  # 全mask的时候，随机选择一个aspect
                mask_num = min(self.mask_aspect_num, len(aspect_pos))
                choose_aspect_list = np.random.choice(aspect_pos, mask_num, replace=False)  # 随机选取一个进行全部mask
            else:
                choose_aspect_list = aspect_pos

            text_mlm_prob = self.masked_lm_prob
            aspect_mlm_prob = self.aspect_mlm_prob

            ori_tokenized_text = [tokenize_list[-1]]  # text field
            ori_all_aspect_text = []  # aspect field
            for x in tokenize_list[:-1]:
                ori_all_aspect_text.append(x)

            truncate_seq(ori_tokenized_text, ori_all_aspect_text,
                         max_prompt_length-2)

            mlm_token_list = ori_tokenized_text  # 只有content
            text, masked_pos, masked_label = self.handle_for_one_example(
                cnt, mlm_token_list, [0], text_mlm_prob, 0.0)
            cnt += 1
            input_ids.append(text)
            all_masked_pos[0].extend(masked_pos[0])
            all_masked_pos[1].extend(masked_pos[1])
            all_masked_label.extend(masked_label)

            if aspect_dict is not None:
                for aspect_name, aspect_rep in aspect_dict.items():
                    if aspect_name not in all_aspect_id:
                        all_aspect_id[aspect_name] = []
                    all_aspect_id[aspect_name].append(aspect_rep[idx])

            if self.pretrain_type in ['prompt']:
                aspect_mlm_token_list = ori_all_aspect_text + \
                    ori_tokenized_text  # aspect(masked) + content

                new_type_list = []
                for pos_id, t_id in enumerate(type_list):
                    if t_id == type_id['aspect']:
                        if pos_id in choose_aspect_list:
                            new_type_list.append(type_id['aspect'])
                        else:  # 被选中的aspect进行全比例mask，其余的替换成prompt类型，不参与mask
                            new_type_list.append(type_id['prompt'])
                    else:
                        new_type_list.append(t_id)

                text, masked_pos, masked_label = self.handle_for_one_example(
                    cnt, aspect_mlm_token_list, new_type_list, 0.0, aspect_mlm_prob)
                cnt += 1

                input_ids.append(text)
                all_masked_pos[0].extend(masked_pos[0])
                all_masked_pos[1].extend(masked_pos[1])
                all_masked_label.extend(masked_label)

                if aspect_dict is not None:
                    for aspect_name, aspect_rep in aspect_dict.items():
                        if aspect_name not in all_aspect_id:
                            all_aspect_id[aspect_name] = []
                        all_aspect_id[aspect_name].append(aspect_rep[idx])

                aspect_text_mlm_token_list = ori_all_aspect_text + \
                    ori_tokenized_text  # aspect + content(masked)
                text, masked_pos, masked_label = self.handle_for_one_example(
                    cnt, aspect_text_mlm_token_list, type_list, text_mlm_prob, 0.0)
                cnt += 1

                input_ids.append(text)
                all_masked_pos[0].extend(masked_pos[0])
                all_masked_pos[1].extend(masked_pos[1])
                all_masked_label.extend(masked_label)

                if aspect_dict is not None:
                    for aspect_name, aspect_rep in aspect_dict.items():
                        if aspect_name not in all_aspect_id:
                            all_aspect_id[aspect_name] = []
                        all_aspect_id[aspect_name].append(aspect_rep[idx])

        batch = self.tokenizer.pad(
            input_ids, return_tensors="pt", padding='max_length', max_length=max_prompt_length)
        lm_label_array = np.full(
            batch['input_ids'].shape, dtype=np.int64, fill_value=-100)  # [bs, seq_len]
        lm_label_array[all_masked_pos[0], all_masked_pos[1]] = all_masked_label
        # all_masked_pos[0]: bs row
        # all_masked_pos[1]: seq_len ,col
        batch["labels"] = torch.tensor(lm_label_array)
        for _name in all_aspect_id.keys():
            all_aspect_id[_name] = torch.tensor(np.array(all_aspect_id[_name]))
        if self.pretrain_type not in ['mlm']:
            batch['aspect_dict'] = all_aspect_id
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, mlm_prob=0.15) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.masked_lm_prob`)
        probability_matrix = torch.full(
            labels.shape, mlm_prob)  # [bs, seq_len] 全0.15
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()  # 1的位置代表是特殊的token

        probability_matrix.masked_fill_(
            special_tokens_mask, value=0.0)  # 把特殊位置的概率填上0
        # True 代表被mask
        # input (Tensor)-对于伯努利分布的输入概率值。取1的概率是p
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 把没有被选中的位置，设置为-100，其余位置为真实的token id

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # [bs, seq_len] 全0.8，每个位置都有0.8的概率是否被真正选中替换为[MASK] & 同时被选中mask
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # 被选中随机替换 & 被选中mask & 没有已经被选中替换为[MASK]
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        pos = torch.where(masked_indices[0])[0].numpy().tolist()
        labels = labels[torch.where(masked_indices)].numpy().tolist()
        inputs = inputs[0].numpy().tolist()
        return inputs, pos, labels


def make_aspect_tokens(all_aspect_list, aspect_level, sep_token):
    """for finetune, model_type=prompt"""
    # same with pretrain
    aspect_list = []
    # if aspect_level > len(all_aspect_list):
    #     raise ValueError('aspect_level error!', aspect_level)
    for aspect_l in all_aspect_list[:aspect_level]:
        if len(aspect_list) != 0 and len(aspect_l) != 0:
            aspect_list.extend(sep_token)
        aspect_list.extend(aspect_l)
    return aspect_list


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

        self.is_concat = data_args.is_concat  # 默认 false ，不拼接多aspect
        self.is_concat_query = data_args.is_concat_query
        self.cat_level = data_args.cat_level  # 用多少级的cat

        if data_args.brand2id_path is not None:  # 加载有监督字典
            self.brand2id_dict = load_dict(data_args.brand2id_path)
        else:
            self.brand2id_dict = None
        if data_args.color2id_path is not None:
            self.color2id_dict = load_dict(data_args.color2id_path)
        else:
            self.color2id_dict = None

        if data_args.cate1_2id_path is not None:
            self.cate1_2id_dict = load_dict(data_args.cate1_2id_path)
        else:
            self.cate1_2id_dict = None

        if data_args.cate2_2id_path is not None:
            self.cate2_2id_dict = load_dict(data_args.cate2_2id_path)
        else:
            self.cate2_2id_dict = None

        if data_args.cate3_2id_path is not None:
            self.cate3_2id_dict = load_dict(data_args.cate3_2id_path)
        else:
            self.cate3_2id_dict = None

        if data_args.cate4_2id_path is not None:
            self.cate4_2id_dict = load_dict(data_args.cate4_2id_path)
        else:
            self.cate4_2id_dict = None

        if data_args.cate5_2id_path is not None:
            self.cate5_2id_dict = load_dict(data_args.cate5_2id_path)
        else:
            self.cate5_2id_dict = None

        self.model_type = data_args.model_type
        dict_list = [
            self.cate1_2id_dict,
            self.cate2_2id_dict,
            self.cate3_2id_dict,
            self.cate4_2id_dict,
            self.cate5_2id_dict,
        ]
        self.all_cate_dict = merge_dict(dict_list[:self.cat_level])
        # self.all_cate_dict = load_dict(data_args.whole_cate_vocab_file)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        """
        p1 = create_one_example(pos_psg)
        tokenizer.convert_ids_to_tokens(p1['input_ids'])
        """
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )  # {'input_ids': [101, 999,...]} 不padding，只truncation, 最开始加上cls，最后加sep
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        encoded_query: {'input_ids': [101, 999,...]}
        encoded_passages: [{'input_ids': [101, 999,...]}, {'input_ids': [101, 999,...]}] 第一个是pos的，其余的是neg的
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['q_text_tokenized']
        if self.is_concat_query == 'first-aspect-num':  # 添加aspect num-1个token，
            qry = [1] + [2] + [3] + [102] + [4] + qry
        elif self.is_concat_query == 'first-aspect-num-nine':  # 添加aspect num-1个token，
            qry = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] \
                + [102] + [10] + qry
        elif self.is_concat_query == 'first-aspect-num-concat':  # 添加aspect num-1个token，
            qry = [5] + [6] + [7] \
                + [1] + []  \
                + [2] +  [] \
                + [3] + [] \
                + [102] + [4] + qry
        elif self.is_concat_query == 'first-aspect-num-concat-less':  # 添加aspect num-1个token，
            qry = [5] + [6] + [7] \
                + [102] + [4] + qry
        
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        encoded_passages_brand_id = []
        encoded_passages_color_id = []
        encoded_passages_cat1_id = []
        encoded_passages_cat2_id = []
        encoded_passages_cat3_id = []
        encoded_passages_cat4_id = []
        encoded_passages_cat5_id = []
        encoded_passages_cat_mul_id = []

        group_text_E = group['E_detail_list']
        group_text_S = group['S_detail_list']  # 原始文本
        group_text_C = group['C_detail_list']  # 原始文本
        group_text_I = group['I_detail_list']  # 原始文本

        group_E_detail = group['E_tokenize_list']
        group_S_detail = group['S_tokenize_list']
        group_C_detail = group['C_tokenize_list']
        group_I_detail = group['I_tokenize_list']

        if self.data_args.positive_passage_no_shuffle:  # 如果不shuffle，直接选第一个
            pos_text = group_text_E[0]
            pos_detail = group_E_detail[0]
        else:  # 进行shuffle
            pos_text = group_text_E[(
                _hashed_seed + epoch) % len(group_E_detail)]
            pos_detail = group_E_detail[(
                _hashed_seed + epoch) % len(group_E_detail)]

        # same with pretrain
        sep_token = self.tok.convert_tokens_to_ids(self.tok.tokenize(','))

        cat_list = make_aspect_tokens(
            pos_detail['product_cate'], self.cat_level, sep_token)

        if self.is_concat == 'yes':
            pos_psg = [1] + cat_list \
                + [2] + pos_detail['product_brand'] \
                + [3] + pos_detail['product_color'] \
                + [102] + [4] + pos_detail['product_title'] \
                + [102] + pos_detail['product_description'] \
                + [102] + pos_detail['product_bullet_point']
        elif self.is_concat == 'first-aspect-num':  # 添加aspect num-1个token，
            pos_psg = [1] + [2] + [3] + [102] + [4] + pos_detail['product_title'] + [102] + pos_detail['product_description'] + [102] + pos_detail['product_bullet_point']
        elif self.is_concat == 'first-aspect-num-nine':  # 添加aspect num-1个token，
            pos_psg = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] \
                + [102] + [10] + pos_detail['product_title'] + [102] + pos_detail['product_description'] + [102] + pos_detail['product_bullet_point']
        elif self.is_concat == 'first-aspect-num-concat':  # 添加aspect num-1个token，
            pos_psg = [5] + [6] + [7] \
                + [1] + pos_detail['product_brand']  \
                + [2] +  pos_detail['product_color'] \
                + [3] + cat_list \
                + [102] + [4] + pos_detail['product_title'] \
                + [102] + pos_detail['product_description'] \
                + [102] + pos_detail['product_bullet_point']
        else:
            pos_psg = pos_detail['product_title'] + [102] + \
                pos_detail['product_description'] + [102] + \
                pos_detail['product_bullet_point']

        pos_psg_seq = self.create_one_example(pos_psg)

        encoded_passages.append(pos_psg_seq)

        if self.brand2id_dict is not None:
            # brand 和 color dict 已经小写
            pos_doc_brand_id = self.brand2id_dict[''] if pos_text[
                'product_brand'].lower() not in self.brand2id_dict else self.brand2id_dict[pos_text['product_brand'].lower()]

            pos_doc_color_id = self.color2id_dict[''] if pos_text[
                'product_color'].lower() not in self.color2id_dict else self.color2id_dict[pos_text['product_color'].lower()]

            cate1_text = pos_text['product_cate'][0] if len(
                pos_text['product_cate']) >= 1 else ''
            cate2_text = pos_text['product_cate'][1] if len(
                pos_text['product_cate']) >= 2 else ''
            cate3_text = pos_text['product_cate'][2] if len(
                pos_text['product_cate']) >= 3 else ''
            cate4_text = pos_text['product_cate'][3] if len(
                pos_text['product_cate']) >= 4 else ''
            cate5_text = pos_text['product_cate'][4] if len(
                pos_text['product_cate']) >= 5 else ''

            pos_doc_cate1_id = self.cate1_2id_dict[
                ''] if cate1_text not in self.cate1_2id_dict else self.cate1_2id_dict[cate1_text]

            pos_doc_cate2_id = self.cate2_2id_dict[
                ''] if cate2_text not in self.cate2_2id_dict else self.cate2_2id_dict[cate2_text]

            pos_doc_cate3_id = self.cate3_2id_dict[
                ''] if cate3_text not in self.cate3_2id_dict else self.cate3_2id_dict[cate3_text]

            pos_doc_cate4_id = self.cate4_2id_dict[
                ''] if cate4_text not in self.cate4_2id_dict else self.cate4_2id_dict[cate4_text]

            pos_doc_cate5_id = self.cate5_2id_dict[
                ''] if cate5_text not in self.cate5_2id_dict else self.cate5_2id_dict[cate5_text]

            encoded_passages_brand_id.append(
                np.array(pos_doc_brand_id, dtype=np.int64))
            encoded_passages_color_id.append(
                np.array(pos_doc_color_id, dtype=np.int64))

            encoded_passages_cat1_id.append(
                np.array(pos_doc_cate1_id, dtype=np.int64))
            encoded_passages_cat2_id.append(
                np.array(pos_doc_cate2_id, dtype=np.int64))
            encoded_passages_cat3_id.append(
                np.array(pos_doc_cate3_id, dtype=np.int64))
            encoded_passages_cat4_id.append(
                np.array(pos_doc_cate4_id, dtype=np.int64))
            encoded_passages_cat5_id.append(
                np.array(pos_doc_cate5_id, dtype=np.int64))

            cate_mul_label = np.zeros(len(self.all_cate_dict))
            for cate_text in pos_text['product_cate'][:self.cat_level]:
                if cate_text in self.all_cate_dict and cate_text != '':
                    mul_id = self.all_cate_dict[cate_text]
                    cate_mul_label[mul_id] = 1
            encoded_passages_cat_mul_id.append(cate_mul_label)

        negative_size = self.data_args.train_n_passages - 1  # train_n_passages-1个负样本

        all_neg_tokenized_list = group_S_detail + \
            group_C_detail + group_I_detail  # 3个等级组成neg
        all_neg_text_list = group_text_S + group_text_C + group_text_I  # 3个等级组成neg

        if len(all_neg_tokenized_list) < negative_size:  # 负样本数量 比需要的少，需要循环选取
            # negs = random.choices(group_negatives, k=negative_size)
            negs_idx = [np.random.choice(len(all_neg_tokenized_list))
                        for _ in range(negative_size)]  # 重复选取的idx
            negs_text = [all_neg_text_list[idx] for idx in negs_idx]
            negs_detail = [all_neg_tokenized_list[idx] for idx in negs_idx]
        elif self.data_args.train_n_passages == 1:  # 不使用自己的负样本，只有一个正样本
            negs_text = []
            negs_detail = []
        elif self.data_args.negative_passage_no_shuffle:  # 如果不对负样本shuffle，直接选取
            negs_text = all_neg_text_list[:negative_size]
            negs_detail = all_neg_tokenized_list[:negative_size]
        else:  # 负样本多，且需要shuffle

            # _offset = epoch * negative_size % len(group_negatives)
            # negs = [x for x in group_negatives]
            # random.Random(_hashed_seed).shuffle(negs)
            # negs = negs * 2
            # negs = negs[_offset: _offset + negative_size]

            _offset = epoch * negative_size % len(all_neg_text_list)
            negs_idx = [i for i in range(len(all_neg_text_list))]
            Random(_hashed_seed).shuffle(negs_idx)
            negs_idx = negs_idx * 2
            negs_idx = negs_idx[_offset: _offset + negative_size]
            negs_text = [all_neg_text_list[idx] for idx in negs_idx]
            negs_detail = [all_neg_tokenized_list[idx] for idx in negs_idx]

        for idx, neg_psg in enumerate(negs_text):
            neg_detail = negs_detail[idx]
            cat_list = make_aspect_tokens(
                neg_detail['product_cate'], self.cat_level, sep_token)

            if self.is_concat == 'yes':
                neg_txt = [1] + cat_list \
                    + [2] + neg_detail['product_brand'] \
                    + [3] + neg_detail['product_color'] \
                    + [102] + [4] + neg_detail['product_title'] \
                    + [102] + neg_detail['product_description'] \
                    + [102] + neg_detail['product_bullet_point']
            elif self.is_concat == 'first-aspect-num':  # 添加aspect num-1个token，
                neg_txt = [1] + [2] + [3] + [102] + [4] + neg_detail['product_title'] + [102] + neg_detail['product_description'] + [102] + neg_detail['product_bullet_point']
            elif self.is_concat == 'first-aspect-num-nine':
                neg_txt = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] \
                    + [102] + [10] + neg_detail['product_title'] + [102] + neg_detail['product_description'] + [102] + neg_detail['product_bullet_point']
            elif self.is_concat == 'first-aspect-num-concat':  # 添加aspect num-1个token，
                neg_txt = [5] + [6] + [7] \
                    + [1] + neg_detail['product_brand']  \
                    + [2] +  neg_detail['product_color'] \
                    + [3] + cat_list \
                    + [102] + [4] + neg_detail['product_title'] \
                    + [102] + neg_detail['product_description'] \
                    + [102] + neg_detail['product_bullet_point']
            else:
                neg_txt = neg_detail['product_title'] + [102] + \
                    neg_detail['product_description'] + [102] + \
                    neg_detail['product_bullet_point']

            neg_txt_seq = self.create_one_example(neg_txt)
            encoded_passages.append(neg_txt_seq)

            if self.brand2id_dict is not None:
                neg_doc_brand_id = self.brand2id_dict[''] if neg_psg[
                    'product_brand'].lower() not in self.brand2id_dict else self.brand2id_dict[neg_psg['product_brand'].lower()]

                neg_doc_color_id = self.color2id_dict[''] if neg_psg[
                    'product_color'].lower() not in self.color2id_dict else self.color2id_dict[neg_psg['product_color'].lower()]

                cate1_text = neg_psg['product_cate'][0] if len(
                    neg_psg['product_cate']) >= 1 else ''
                cate2_text = neg_psg['product_cate'][1] if len(
                    neg_psg['product_cate']) >= 2 else ''
                cate3_text = neg_psg['product_cate'][2] if len(
                    neg_psg['product_cate']) >= 3 else ''
                cate4_text = neg_psg['product_cate'][3] if len(
                    neg_psg['product_cate']) >= 4 else ''
                cate5_text = neg_psg['product_cate'][4] if len(
                    neg_psg['product_cate']) >= 5 else ''

                neg_doc_cate1_id = self.cate1_2id_dict[
                    ''] if cate1_text not in self.cate1_2id_dict else self.cate1_2id_dict[cate1_text]

                neg_doc_cate2_id = self.cate2_2id_dict[
                    ''] if cate2_text not in self.cate2_2id_dict else self.cate2_2id_dict[cate2_text]

                neg_doc_cate3_id = self.cate3_2id_dict[
                    ''] if cate3_text not in self.cate3_2id_dict else self.cate3_2id_dict[cate3_text]

                neg_doc_cate4_id = self.cate4_2id_dict[
                    ''] if cate4_text not in self.cate4_2id_dict else self.cate4_2id_dict[cate4_text]

                neg_doc_cate5_id = self.cate5_2id_dict[
                    ''] if cate5_text not in self.cate5_2id_dict else self.cate5_2id_dict[cate5_text]

                encoded_passages_brand_id.append(
                    np.array(neg_doc_brand_id, dtype=np.int64))
                encoded_passages_color_id.append(
                    np.array(neg_doc_color_id, dtype=np.int64))

                encoded_passages_cat1_id.append(
                    np.array(neg_doc_cate1_id, dtype=np.int64))
                encoded_passages_cat2_id.append(
                    np.array(neg_doc_cate2_id, dtype=np.int64))
                encoded_passages_cat3_id.append(
                    np.array(neg_doc_cate3_id, dtype=np.int64))
                encoded_passages_cat4_id.append(
                    np.array(neg_doc_cate4_id, dtype=np.int64))
                encoded_passages_cat5_id.append(
                    np.array(neg_doc_cate5_id, dtype=np.int64))

                cate_mul_label = np.zeros(len(self.all_cate_dict))
                for cate_text in neg_psg['product_cate'][:self.cat_level]:
                    if cate_text in self.all_cate_dict and cate_text != '':
                        mul_id = self.all_cate_dict[cate_text]
                        cate_mul_label[mul_id] = 1
                encoded_passages_cat_mul_id.append(cate_mul_label)
        return encoded_query, encoded_passages, encoded_passages_brand_id, encoded_passages_color_id,\
            encoded_passages_cat1_id, encoded_passages_cat2_id, encoded_passages_cat3_id, \
            encoded_passages_cat4_id, encoded_passages_cat5_id, encoded_passages_cat_mul_id


class EncodeDataset(Dataset):

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128, is_query=None, data_args=None):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len
        self.is_query = is_query
        self.model_type = data_args.model_type
        self.is_concat = data_args.is_concat  # 默认 false ，不拼接多aspect
        self.is_concat_query = data_args.is_concat_query
        self.cat_level = data_args.cat_level  # 用多少级的cat
        if data_args.brand2id_path is not None:  # 加载有监督字典
            self.brand2id_dict = load_dict(data_args.brand2id_path)
        else:
            self.brand2id_dict = None
        if data_args.color2id_path is not None:
            self.color2id_dict = load_dict(data_args.color2id_path)
        else:
            self.color2id_dict = None

        if data_args.cate1_2id_path is not None:
            self.cate1_2id_dict = load_dict(data_args.cate1_2id_path)
        else:
            self.cate1_2id_dict = None

        if data_args.cate2_2id_path is not None:
            self.cate2_2id_dict = load_dict(data_args.cate2_2id_path)
        else:
            self.cate2_2id_dict = None

        if data_args.cate3_2id_path is not None:
            self.cate3_2id_dict = load_dict(data_args.cate3_2id_path)
        else:
            self.cate3_2id_dict = None

        if data_args.cate4_2id_path is not None:
            self.cate4_2id_dict = load_dict(data_args.cate4_2id_path)
        else:
            self.cate4_2id_dict = None

        if data_args.cate5_2id_path is not None:
            self.cate5_2id_dict = load_dict(data_args.cate5_2id_path)
        else:
            self.cate5_2id_dict = None

        if data_args.cate_wordpiece_vocab_file is not None:
            self.cate_wordpiece_dict = load_dict(data_args.cate_wordpiece_vocab_file)  # wordpiece 词典
            self.cate_word_dict = load_dict(data_args.cate_word_vocab_file)
            self.cate_wholeword_dict = load_dict(data_args.whole_cate_vocab_file)

            self.brand_wordpiece_dict = load_dict(data_args.brand_wordpiece_vocab_file)  # wordpiece 词典
            self.brand_word_dict = load_dict(data_args.brand_word_vocab_file)
            self.brand_wholeword_dict = load_dict(data_args.whole_brand_vocab_file)

            self.color_wordpiece_dict = load_dict(data_args.color_wordpiece_vocab_file)  # wordpiece 词典
            self.color_word_dict = load_dict(data_args.color_word_vocab_file)
            self.color_wholeword_dict = load_dict(data_args.whole_color_vocab_file)
        else:
            self.cate_wordpiece_dict = None  # wordpiece 词典
            self.cate_word_dict = None
            self.cate_wholeword_dict = None

            self.brand_wordpiece_dict = None  # wordpiece 词典
            self.brand_word_dict = None
            self.brand_wholeword_dict = None

            self.color_wordpiece_dict = None  # wordpiece 词典
            self.color_word_dict = None
            self.color_wholeword_dict = None

        dict_list = [
            self.cate1_2id_dict,
            self.cate2_2id_dict,
            self.cate3_2id_dict,
            self.cate4_2id_dict,
            self.cate5_2id_dict,
        ]
        self.all_cate_dict = merge_dict(dict_list[:self.cat_level])
        # self.all_cate_dict = load_dict(data_args.whole_cate_vocab_file)

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        # same with pretrain
        sep_token = self.tok.convert_tokens_to_ids(self.tok.tokenize(','))

        if self.is_query is False:  # for doc
            text_info = self.encode_data[item]['doc_info']
            tokenized_info = self.encode_data[item]['doc_info_tokenize']

            doc_id = text_info['product_id']
            product_title = tokenized_info['product_title']
            product_description = tokenized_info['product_description']
            product_bullet_point = tokenized_info['product_bullet_point']
            product_brand = tokenized_info['product_brand']
            product_color = tokenized_info['product_color']
            # cate1_tokens = tokenized_info['product_cate'][0] if len(
            #     tokenized_info['product_cate']) >= 1 else []
            # cate2_tokens = tokenized_info['product_cate'][1] if len(
            #     tokenized_info['product_cate']) >= 2 else []
            # cate3_tokens = tokenized_info['product_cate'][2] if len(
            #     tokenized_info['product_cate']) >= 3 else []
            # cate4_tokens = tokenized_info['product_cate'][3] if len(
            #     tokenized_info['product_cate']) >= 4 else []
            # cate5_tokens = tokenized_info['product_cate'][4] if len(
            #     tokenized_info['product_cate']) >= 5 else []

            brand_id = self.brand2id_dict[''] if text_info[
                'product_brand'].lower() not in self.brand2id_dict else self.brand2id_dict[text_info['product_brand'].lower()]

            color_id = self.color2id_dict[''] if text_info[
                'product_color'].lower() not in self.color2id_dict else self.color2id_dict[text_info['product_color'].lower()]

            cate1_text = text_info['product_cate'][0] if len(
                text_info['product_cate']) >= 1 else ''
            cate2_text = text_info['product_cate'][1] if len(
                text_info['product_cate']) >= 2 else ''
            cate3_text = text_info['product_cate'][2] if len(
                text_info['product_cate']) >= 3 else ''
            cate4_text = text_info['product_cate'][3] if len(
                text_info['product_cate']) >= 4 else ''
            cate5_text = text_info['product_cate'][4] if len(
                text_info['product_cate']) >= 5 else ''

            cate1_id = self.cate1_2id_dict[
                ''] if cate1_text not in self.cate1_2id_dict else self.cate1_2id_dict[cate1_text]

            cate2_id = self.cate2_2id_dict[
                ''] if cate2_text not in self.cate2_2id_dict else self.cate2_2id_dict[cate2_text]

            cate3_id = self.cate3_2id_dict[
                ''] if cate3_text not in self.cate3_2id_dict else self.cate3_2id_dict[cate3_text]

            cate4_id = self.cate4_2id_dict[
                ''] if cate4_text not in self.cate4_2id_dict else self.cate4_2id_dict[cate4_text]

            cate5_id = self.cate5_2id_dict[
                ''] if cate5_text not in self.cate5_2id_dict else self.cate5_2id_dict[cate5_text]

            cate_mul_label = np.zeros(len(self.all_cate_dict))
            for cate_text in text_info['product_cate'][:self.cat_level]:
                if cate_text in self.all_cate_dict and cate_text != '':
                    mul_id = self.all_cate_dict[cate_text]
                    cate_mul_label[mul_id] = 1

            aspect_id = {}
            if self.model_type == 'mgrain' and self.cate_wordpiece_dict is not None:
                brand = text_info['product_brand'].lower()
                color = text_info['product_color'].lower()

                
                aspect_fc = {
                    'p_brand': self.brand2id_dict,
                    'p_color': self.color2id_dict
                }

                regex = r'[^\w\s]|[\s]'  # 空格和标点符号切分
                # wordpiece
                for_cate_wordpiece = [str(len(self.cate_wordpiece_dict))]
                for cate_text in text_info['product_cate'][:self.cat_level]:
                    words = re.split(regex, cate_text)
                    for word in words:
                        if word.strip():
                            for w in  self.tok.tokenize(word):
                                for_cate_wordpiece.append(str(self.cate_wordpiece_dict[w]))
                string_cate_vocab = '#'.join(for_cate_wordpiece)  # 第一个是长度
                aspect_id['p_cate_wordpiece'] = string_cate_vocab

                for_brand_wordpiece = [str(len(self.brand_wordpiece_dict))]
                for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:
                    words = re.split(regex, brand_text)
                    for word in words:
                        if word.strip():
                            for w in  self.tok.tokenize(word):
                                for_brand_wordpiece.append(str(self.brand_wordpiece_dict[w]))
                string_brand_vocab = '#'.join(for_brand_wordpiece)  # 第一个是长度
                aspect_id['p_brand_wordpiece'] = string_brand_vocab

                # wordpiece
                for_color_wordpiece = [str(len(self.color_wordpiece_dict))]
                for color_text in [color if color in aspect_fc['p_color'] else '']:
                    words = re.split(regex, color_text)
                    for word in words:
                        if word.strip():
                            for w in  self.tok.tokenize(word):
                                for_color_wordpiece.append(str(self.color_wordpiece_dict[w]))
                string_color_vocab = '#'.join(for_color_wordpiece)  # 第一个是长度
                aspect_id['p_color_wordpiece'] = string_color_vocab
                
                # word
                for_cate_word = [str(len(self.cate_word_dict))]
                for cate_text in text_info['product_cate'][:self.cat_level]:
                    words = re.split(regex, cate_text)
                    for word in words:
                        if word.strip():
                            word_id = self.cate_word_dict[word.strip()]  # 所有的词都在词典中出现
                            for_cate_word.append(str(word_id))
                string_cate_vocab = '#'.join(for_cate_word)  # 第一个是长度
                aspect_id['p_cate_word'] = string_cate_vocab

                # word
                for_brand_word = [str(len(self.brand_word_dict))]
                for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:
                    words = re.split(regex, brand_text)
                    for word in words:
                        if word.strip():
                            word_id = self.brand_word_dict[word.strip()]  # 所有的词都在词典中出现
                            for_brand_word.append(str(word_id))
                string_brand_vocab = '#'.join(for_brand_word)  # 第一个是长度
                aspect_id['p_brand_word'] = string_brand_vocab

                # word
                for_color_word = [str(len(self.color_word_dict))]
                for color_text in [color if color in aspect_fc['p_color'] else '']:
                    words = re.split(regex, color_text)
                    for word in words:
                        if word.strip():
                            word_id = self.color_word_dict[word.strip()]  # 所有的词都在词典中出现
                            for_color_word.append(str(word_id))
                string_color_vocab = '#'.join(for_color_word)  # 第一个是长度
                aspect_id['p_color_word'] = string_color_vocab

                # wholeword
                for_cate_wholeword = [str(len(self.cate_wholeword_dict))]
                for cate_text in text_info['product_cate'][:self.cat_level]:  # 完整的保持，不切词
                    word_id = self.cate_wholeword_dict[cate_text]
                    for_cate_wholeword.append(str(word_id))
                string_cate_vocab = '#'.join(for_cate_wholeword)  # 第一个是长度
                aspect_id['p_cate_wholeword'] = string_cate_vocab

                # wholeword
                for_brand_wholeword = [str(len(self.brand_wholeword_dict))]
                for brand_text in [brand if brand in aspect_fc['p_brand'] else '']:  # 完整的保持，不切词
                    word_id = self.brand_wholeword_dict[brand_text]
                    for_brand_wholeword.append(str(word_id))
                # string_brand_vocab = '#'.join(for_brand_wholeword)  # 第一个是长度
                # aspect_id['p_brand_wholeword'] = string_brand_vocab
                assert len(for_brand_wholeword) == 2
                aspect_id['p_brand_wholeword'] = for_brand_wholeword[1]
                

                # wholeword
                for_color_wholeword = [str(len(self.color_wholeword_dict))]
                for color_text in [color if color in aspect_fc['p_color'] else '']:  # 完整的保持，不切词
                    word_id = self.color_wholeword_dict[color_text]
                    for_color_wholeword.append(str(word_id))
                # string_color_vocab = '#'.join(for_color_wholeword)  # 第一个是长度
                # aspect_id['p_color_wholeword'] = string_color_vocab
                assert len(for_color_wholeword) == 2
                aspect_id['p_color_wholeword'] = for_color_wholeword[1]


            text_id = doc_id

            cat_list = make_aspect_tokens(
                tokenized_info['product_cate'], self.cat_level, sep_token)
            if self.is_concat == 'yes':
                text = [1] + cat_list \
                    + [2] + product_brand \
                    + [3] + product_color \
                    + [102] + [4] + product_title \
                    + [102] + product_description \
                    + [102] + product_bullet_point
            elif self.is_concat == 'first-aspect-num':  # 添加aspect num-1个token，
                text = [1] + [2] + [3] + [102] + [4] + product_title + [102] + product_description + [102] + product_bullet_point
            elif self.is_concat == 'first-aspect-num-nine':
                text = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] \
                    + [102] + [10] + product_title + [102] + product_description + [102] + product_bullet_point
            elif self.is_concat == 'first-aspect-num-concat':  # 添加aspect num-1个token，
                text = [5] + [6] + [7] \
                    + [1] + product_brand  \
                    + [2] +  product_color \
                    + [3] + cat_list \
                    + [102] + [4] + product_title \
                    + [102] + product_description \
                    + [102] + product_bullet_point
            else:
                text = product_title + \
                    [102] + product_description + [102] + product_bullet_point

            encoded_text = self.tok.prepare_for_model(
                text,
                max_length=self.max_len,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
            return text_id, encoded_text, brand_id, color_id, \
                cate1_id, cate2_id, cate3_id, cate4_id, cate5_id, cate_mul_label, aspect_id

        else:  # 默认 + query
            text_id = self.encode_data[item]['q_id']
            text = self.encode_data[item]['q_text_tokenized']

            if self.is_concat_query == 'first-aspect-num':  # 添加aspect num-1个token，
                text = [1] + [2] + [3] + [102] + [4] + text
            elif self.is_concat_query == 'first-aspect-num-nine':
                text = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] \
                    + [102] + [10] + text
            elif self.is_concat_query == 'first-aspect-num-concat':  # 添加aspect num-1个token，
                text = [5] + [6] + [7] \
                    + [1] + []  \
                    + [2] +  [] \
                    + [3] + [] \
                    + [102] + [4] + text
            elif self.is_concat_query == 'first-aspect-num-concat-less':  # 添加aspect num-1个token，
                text = [5] + [6] + [7] \
                    + [102] + [4] + text

            encoded_text = self.tok.prepare_for_model(
                text,
                max_length=self.max_len,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
            return text_id, encoded_text


@ dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        dd_brand_id = [f[2] for f in features]
        dd_color_id = [f[3] for f in features]
        dd_cat1_id = [f[4] for f in features]
        dd_cat2_id = [f[5] for f in features]
        dd_cat3_id = [f[6] for f in features]
        dd_cat4_id = [f[7] for f in features]
        dd_cat5_id = [f[8] for f in features]
        dd_cat_mul_id = [f[9] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(dd_brand_id[0], list):
            dd_brand_id = sum(dd_brand_id, [])
            dd_brand_id = np.array(dd_brand_id)
        if isinstance(dd_color_id[0], list):
            dd_color_id = sum(dd_color_id, [])
            dd_color_id = np.array(dd_color_id)
        if isinstance(dd_cat1_id[0], list):
            dd_cat1_id = sum(dd_cat1_id, [])
            dd_cat1_id = np.array(dd_cat1_id)
        if isinstance(dd_cat2_id[0], list):
            dd_cat2_id = sum(dd_cat2_id, [])
            dd_cat2_id = np.array(dd_cat2_id)
        if isinstance(dd_cat3_id[0], list):
            dd_cat3_id = sum(dd_cat3_id, [])
            dd_cat3_id = np.array(dd_cat3_id)
        if isinstance(dd_cat4_id[0], list):
            dd_cat4_id = sum(dd_cat4_id, [])
            dd_cat4_id = np.array(dd_cat4_id)
        if isinstance(dd_cat5_id[0], list):
            dd_cat5_id = sum(dd_cat5_id, [])
            dd_cat5_id = np.array(dd_cat5_id)
        if isinstance(dd_cat_mul_id[0], list):
            dd_cat_mul_id = sum(dd_cat_mul_id, [])
            dd_cat_mul_id = np.array(dd_cat_mul_id)

        """
        tokenizer.pad(
            {'input_ids': [101, 7160, 8139, 1006, 102]}, padding='max_length',max_length=10, return_tensors="pt",)

        {'input_ids': tensor([ 101, 7160, 8139, 1006,  102,    0,    0,    0,    0,    0]),
        'attention_mask': tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])}
        """
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )  # 只进行padding，返回attention_mask
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        return q_collated, d_collated, dd_brand_id, dd_color_id,\
            dd_cat1_id, dd_cat2_id, dd_cat3_id, dd_cat4_id, dd_cat5_id, dd_cat_mul_id


@ dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        if len(features) != 0 and len(features[0]) > 2:
            brand_id = [x[2] for x in features]
            color_id = [x[3] for x in features]
            cate1_id = [x[4] for x in features]
            cate2_id = [x[5] for x in features]
            cate3_id = [x[6] for x in features]
            cate4_id = [x[7] for x in features]
            cate5_id = [x[8] for x in features]
            cate_mul_id = [x[9] for x in features]

            aspect_names = features[0][10].keys()
            aspect_dict = {}
            for _name in aspect_names:
                if _name in ['p_cate_vocab', 'p_cate_wordpiece', 'p_cate_word', 'p_cate_wholeword', \
                    'p_brand_wordpiece', 'p_brand_word',\
                    'p_color_wordpiece', 'p_color_word']:
                    cate_vocab_all = []  # 每个样本的category的word id list
                    for f in features:
                        string_cate_vocab = f[10][_name]
                        cate_vocab_info = string_cate_vocab.split('#')
                        cate_vocab_label = np.zeros(int(cate_vocab_info[0]))
                        for word_id in cate_vocab_info[1:]:  # 根据出现次数给予权重
                            cate_vocab_label[int(word_id)] = 1  # 暂时不考虑出现的次数
                        cate_vocab_all.append(cate_vocab_label)
                    cate_vocab_all = np.array(cate_vocab_all)
                    aspect_dict[_name] = cate_vocab_all
                elif _name in ['p_brand_wholeword', 'p_color_wholeword']:
                    aspect_dict[_name] = np.array([f[10][_name] for f in features], dtype=np.int64)


            brand_id = torch.tensor(brand_id)
            color_id = torch.tensor(color_id)
            cate1_id = torch.tensor(cate1_id)
            cate2_id = torch.tensor(cate2_id)
            cate3_id = torch.tensor(cate3_id)
            cate4_id = torch.tensor(cate4_id)
            cate5_id = torch.tensor(cate5_id)
            cate_mul_id = torch.tensor(cate_mul_id)

            return text_ids, collated_features, brand_id, color_id,\
                cate1_id, cate2_id, cate3_id, cate4_id, cate5_id, cate_mul_id, aspect_dict
        else:
            return text_ids, collated_features
