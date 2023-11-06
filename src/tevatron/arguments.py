import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pretrain_model_name_or_path: str = field(
        default=None, metadata={"help": ""}
    )
    query_model_name_or_path: str = field(
        default='no', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    doc_model_name_or_path: str = field(
        default='no', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    is_eval_aspect: str = field(
        default='yes', metadata={"help": "is_eval_aspect_acc"}
    )
    is_pretrain: str = field(
        default='no', metadata={"help": "is want to do aspect retrain"}
    )
    is_shared_mgrain: str = field(
        default=None, metadata={"help": ""}
    )
    aspect_loss_type: str = field(
        default='no', metadata={"help": "aspect_loss_type: softmax / no"}
    )
    tensorboard_output_dir: str = field(
        default=None,
        metadata={"help": "Path to save tensorboard"}
    )
    model_type: str = field(
        default='bibert',
        metadata={"help": "bibert or mtbert or madr"}
    )
    ab_type: str = field(
        default='no',
        metadata={"help": "ablation study type"}
    )
    q_pool_type: str = field(
        default='cls',
        metadata={"help": "cls or att"}
    )
    doc_pool_type: str = field(
        default='cls',
        metadata={"help": "cls or att"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    amlm_loss_factor: Optional[float] = field(
        default=1.0, metadata={"help": "loss factor for aspect masked language model"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    gating_type: str = field(
        default='no', metadata={"help": ""}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    pre_layer_num: int = field(default=None)
    aspect_num: int = field(default=None)
    begin_id: int = field(default=None)
    head_num: int = field(default=None)
    skip_from: int = field(default=None)
    tem: float = field(default=1.0)
    thred: float = field(default=3)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    cat_level: int = field(
        default=None, metadata={"help": "from 0 to 3"}
    )
    config_path: str = field(
        default=None, metadata={"help": "config file"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    is_concat: str = field(
        default='no', metadata={"help": "do not concat for doc input"}
    )
    is_concat_query: str = field(
        default='no', metadata={"help": "do not concat for query input"}
    )
    prompt_type: str = field(
        default='no', metadata={"help": "prompt type"}
    )
    pretrain_type: str = field(
        default='mlm', metadata={"help": "can choose in ['prompt', 'mlm']"}
    )
    brand2id_path: str = field(
        default=None, metadata={"help": "Path to brand2id dict"}
    )
    color2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate1_2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate2_2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate3_2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate4_2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate5_2id_path: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate_word_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    whole_cate_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    cate_wordpiece_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    brand_word_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    whole_brand_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    brand_wordpiece_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    color_word_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    whole_color_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    color_wordpiece_vocab_file: str = field(
        default=None, metadata={"help": "Path to color2id dict"}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=2, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)
    mask_aspect_num: int = field(default=None)
    multi_have_weight: int = field(default=0)
    mlm_probability: float = field(
        default=None, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    aspect_mlm_prob: float = field(
        default=0.5, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum text input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    max_prompt_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    prompt_add_length: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    def __post_init__(self):
        if self.dataset_name is not None and self.dataset_name != '':
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files
                    if f.endswith('jsonl') or f.endswith('json')
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    pretrain_aspect_alpha: float = field(
        default=None, metadata={"help": "alpha for aspect loss in mtbert and madr"}
    )
    finetune_aspect_alpha: float = field(
        default=None, metadata={"help": "alpha for aspect loss in finetune"}
    )
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
