import logging
import os
import pickle
import sys
# from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeDataset, EncodeCollator, load_dict, merge_dict
from tevatron.modeling import EncoderOutput, DenseModel, MtBertModel, MADRModel, MULGRAIN_Model
from tevatron.datasets import HFQueryDataset, HFCorpusDataset
import os
import random
from sklearn.metrics import accuracy_score
from tevatron.config import get_cfg


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    print("sys.argv:", sys.argv)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if data_args.config_path is not None:
        cfg = get_cfg(data_args.config_path)

        need_cfg = [cfg.common, cfg.encode_query] if data_args.encode_is_qry else [
            cfg.common, cfg.encode_doc]
        for cfg_dict in need_cfg:
            for k, v in cfg_dict.items():
                if hasattr(model_args, k):
                    setattr(model_args, k, v)
                elif hasattr(data_args, k):
                    setattr(data_args, k, v)
                elif hasattr(training_args, k):
                    setattr(training_args, k, v)
                else:
                    raise ValueError('not find args!', k, v)
        data_args.__post_init__()

    print("training_args.n_gpu:", training_args.n_gpu)
    print("training_args.local_rank:", training_args.local_rank)

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # if training_args.local_rank > 0 or training_args.n_gpu > 1:
    #     raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]", "[SEP]", "[unused4]", \
            "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]", "[unused10]", "[unused11]"]})

    if data_args.brand2id_path is not None:  # 加载有监督字典
        brand2id_dict = load_dict(data_args.brand2id_path)
        training_args.brand2id_dict_len = len(brand2id_dict)
        training_args.brand_zero_id = brand2id_dict['']

    if data_args.color2id_path is not None:
        color2id_dict = load_dict(data_args.color2id_path)
        training_args.color2id_dict_len = len(color2id_dict)
        training_args.color_zero_id = color2id_dict['']

    if data_args.cate1_2id_path is not None:
        cat1_2id_dict = load_dict(data_args.cate1_2id_path)
        training_args.cat1_2id_dict_len = len(cat1_2id_dict)
        training_args.cat1_zero_id = cat1_2id_dict['']

    if data_args.cate2_2id_path is not None:
        cat2_2id_dict = load_dict(data_args.cate2_2id_path)
        training_args.cat2_2id_dict_len = len(cat2_2id_dict)
        training_args.cat2_zero_id = cat2_2id_dict['']

    if data_args.cate3_2id_path is not None:
        cat3_2id_dict = load_dict(data_args.cate3_2id_path)
        training_args.cat3_2id_dict_len = len(cat3_2id_dict)
        training_args.cat3_zero_id = cat3_2id_dict['']

    if data_args.cate4_2id_path is not None:
        cat4_2id_dict = load_dict(data_args.cate4_2id_path)
        training_args.cat4_2id_dict_len = len(cat4_2id_dict)
        training_args.cat4_zero_id = cat4_2id_dict['']

    if data_args.cate5_2id_path is not None:
        cat5_2id_dict = load_dict(data_args.cate5_2id_path)
        training_args.cat5_2id_dict_len = len(cat5_2id_dict)
        training_args.cat5_zero_id = cat5_2id_dict['']

    all_cat_dict = [
        cat1_2id_dict,
        cat2_2id_dict,
        cat3_2id_dict,
        cat4_2id_dict,
        cat5_2id_dict
    ]
    cat_level = data_args.cat_level
    cat_mul2id_dict = merge_dict(all_cat_dict[:cat_level])
    training_args.cat_mul2id_dict_len = len(cat_mul2id_dict)
    training_args.cat_mul_zero_id = cat_mul2id_dict['']

    if model_args.model_type == 'mgrain' \
        and model_args.pretrain_model_name_or_path is not None \
        and model_args.pretrain_model_name_or_path != '':
        complete_model = torch.load(model_args.pretrain_model_name_or_path+'/pytorch_model.bin')

        if 'cate_wordpiece_emb.weight' in complete_model.keys():
            cate_wordpiece_emb = complete_model['cate_wordpiece_emb.weight']
            brand_wordpiece_emb = complete_model['brand_wordpiece_emb.weight']
            color_wordpiece_emb = complete_model['color_wordpiece_emb.weight']

            cate_word_emb = complete_model['cate_word_emb.weight']
            brand_word_emb = complete_model['brand_word_emb.weight']
            color_word_emb = complete_model['color_word_emb.weight']

            cate_wholeword_emb = complete_model['cate_wholeword_emb.weight']
            brand_wholeword_emb = complete_model['brand_wholeword_emb.weight']
            color_wholeword_emb = complete_model['color_wholeword_emb.weight']

            aspect_emb_dict = {
                'p_cate_wholeword': cate_wholeword_emb,
                'p_cate_word': cate_word_emb,
                'p_cate_wordpiece': cate_wordpiece_emb,

                'p_brand_wholeword': brand_wholeword_emb,
                'p_brand_word': brand_word_emb,
                'p_brand_wordpiece': brand_wordpiece_emb,

                'p_color_wholeword': color_wholeword_emb,
                'p_color_word': color_word_emb,
                'p_color_wordpiece': color_wordpiece_emb,
            }
            model_args.aspect_emb_dict = aspect_emb_dict

    model_dict = {
        'mtbert': MtBertModel,
        'bibert': DenseModel,
        'madr': MADRModel,
        'mgrain': MULGRAIN_Model,
    }

    if model_args.model_type in model_dict:
        model = model_dict[model_args.model_type].load(
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            model_args=model_args,
            train_args=training_args,
            cache_dir=model_args.cache_dir,
        )
    else:
        assert 'not valid model_type!'
        raise ValueError

    data_args.model_type = model_args.model_type

    aspect2id_dict = {
        'p_brand': brand2id_dict,
        'p_color': color2id_dict,
        'p_cate1': cat1_2id_dict,
        'p_cate2': cat2_2id_dict,
        'p_cate3': cat3_2id_dict,
        'p_cate4': cat4_2id_dict,
        'p_cate5': cat5_2id_dict,
        'p_cate_mul': cat_mul2id_dict,
        'p_brand_wholeword': brand2id_dict,
        'p_color_wholeword': color2id_dict,
    }

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if data_args.dataset_name in ['json', 'amazon']:  # pre-tokenized
        if data_args.encode_is_qry:
            encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                           tokenizer, max_len=text_max_length, is_query=True, data_args=data_args)
        else:
            encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                           tokenizer, max_len=text_max_length, is_query=False, data_args=data_args)
    else:
        encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                       tokenizer, max_len=text_max_length, data_args=data_args)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    doc_aspect_label = {}
    doc_avg_acc = {}

    for sampled_batched in tqdm(encode_loader):
        if data_args.encode_is_qry:
            batch_ids, batch = sampled_batched
        else:
            batch_ids, batch, batch_brand_ids, batch_color_ids,\
                batch_cate1_ids, batch_cate2_ids, batch_cate3_ids, \
                batch_cate4_ids, batch_cate5_ids, batch_cate_mul_ids, mgran_aspect_dict = sampled_batched
            aspect_id = {
                'p_brand': batch_brand_ids,
                'p_color': batch_color_ids,
                'p_cate1': batch_cate1_ids,
                'p_cate2': batch_cate2_ids,
                'p_cate3': batch_cate3_ids,
                'p_cate4': batch_cate4_ids,
                'p_cate5': batch_cate5_ids,
                'p_cate_mul': batch_cate_mul_ids,
            }

            for k, v in mgran_aspect_dict.items():
                assert k not in aspect_id
                aspect_id[k] = v
            
        lookup_indices.extend(batch_ids)

        if training_args.fp16:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    if data_args.encode_is_qry:
                        model_output = model(query=batch)
                        encoded.append(
                            model_output.q_reps.cpu().detach().numpy())
                    else:
                        if model_args.model_type == 'bibert':
                            model_output = model(passage=batch)
                        else:
                            model_output = model(passage=batch, aspect_id=aspect_id)
                        encoded.append(
                            model_output.p_reps.cpu().detach().numpy())

                        if 'doc_label_dict' in model_output and model_output.doc_label_dict is not None:
                            for a_name, model_label in model_output.doc_label_dict.items():
                                if a_name not in doc_aspect_label:
                                    doc_aspect_label[a_name] = {
                                        'labels_true': [],
                                        'labels_pred': []
                                    }
                                doc_aspect_label[a_name]['labels_true'].extend(
                                    aspect_id[a_name])
                                doc_aspect_label[a_name]['labels_pred'].extend(
                                    model_label)
                        if 'doc_acc_dict' in model_output and model_output.doc_acc_dict is not None:
                            for a_name,  model_acc in model_output.doc_acc_dict.items():
                                aspect_name = a_name
                                if aspect_name not in doc_avg_acc:
                                    doc_avg_acc[aspect_name] = []
                                if len(model_acc) != 0 and model_acc[0] != -1:
                                    doc_avg_acc[aspect_name].extend(model_acc)
        else:
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    if model_args.model_type == 'bibert':
                        model_output = model(passage=batch)
                    else:
                        model_output = model(passage=batch, aspect_id=aspect_id)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

                    if 'doc_label_dict' in model_output and model_output.doc_label_dict is not None:
                        for a_name, model_label in model_output.doc_label_dict.items():
                            if a_name not in doc_aspect_label:
                                doc_aspect_label[a_name] = {
                                    'labels_true': [],
                                    'labels_pred': []
                                }
                            doc_aspect_label[a_name]['labels_true'].extend(
                                aspect_id[a_name])
                            doc_aspect_label[a_name]['labels_pred'].extend(
                                model_label)
                    if 'doc_acc_dict' in model_output and model_output.doc_acc_dict is not None:
                        for a_name,  model_acc in model_output.doc_acc_dict.items():
                            aspect_name = a_name
                            if aspect_name not in doc_avg_acc:
                                doc_avg_acc[aspect_name] = []
                            if len(model_acc) != 0 and model_acc[0] != -1:
                                doc_avg_acc[aspect_name].extend(model_acc)

    encoded = np.concatenate(encoded)

    def do_acc_cal(true_label, pred_label, ignore_label=None):
        correct_num = 0
        all_num = 0
        for idx in range(len(true_label)):
            # 忽略无意义标签
            if ignore_label is not None and true_label[idx] == ignore_label:
                continue
            if true_label[idx] == pred_label[idx]:
                correct_num += 1
            all_num += 1
        all_num = 1 if all_num == 0 else all_num
        return correct_num / all_num

    for a_name, label_dict in doc_aspect_label.items():
        aspect_id_dict = aspect2id_dict[a_name]
        print('====', a_name)
        print(accuracy_score(
            label_dict['labels_true'], label_dict['labels_pred']))
        print(do_acc_cal(
            label_dict['labels_true'], label_dict['labels_pred'], ignore_label=aspect_id_dict['']))

    for a_name, acc_list in doc_avg_acc.items():
        print('====', a_name)
        acc_np = np.array(acc_list)
        print(np.mean(acc_np), np.max(acc_np), np.min(acc_np))


    if data_args.encode_is_qry:
        if data_args.encoded_save_path != 'no':
            with open(data_args.encoded_save_path, 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)
    else:
        if data_args.encoded_save_path != 'no':
            with open(data_args.encoded_save_path, 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    import random
    os.environ['MASTER_PORT'] = str(random.randint(5000, 6000))
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    main()
