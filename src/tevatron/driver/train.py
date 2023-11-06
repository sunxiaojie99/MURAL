import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import TrainDataset, QPCollator, load_dict, merge_dict
from tevatron.modeling import DenseModel, MtBertModel, MADRModel, MULGRAIN_Model
from tevatron.trainer import TevatronTrainer as Trainer, GCTrainer
from tevatron.datasets import HFTrainDataset
from torch.utils.tensorboard import SummaryWriter
from tevatron.config import get_cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if data_args.config_path is not None:
        cfg = get_cfg(data_args.config_path)
        need_cfg = [cfg.common, cfg.finetune]
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

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
    training_args.model_type = model_args.model_type
    data_args.model_type = model_args.model_type

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

    model_dict = {
        'mtbert': MtBertModel,
        'bibert': DenseModel,
        'madr': MADRModel,
        'mgrain': MULGRAIN_Model,
    }

    if model_args.model_type in model_dict:
        model = model_dict[model_args.model_type].build(
            model_args=model_args,
            train_args=training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        assert 'not valid model_type!' + model_args.model_type
        raise ValueError

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    if training_args.device == torch.device('cuda:0'):
        writer = SummaryWriter(model_args.tensorboard_output_dir)
    else:
        writer = None

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        ttf_writer=writer,
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.device == torch.device('cuda:0'):
        writer.close()


if __name__ == "__main__":
    main()
