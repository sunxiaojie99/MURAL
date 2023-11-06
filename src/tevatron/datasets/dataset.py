from datasets import load_dataset
from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor
from .preprocessor import TrainPreProcessor_a, QueryPreProcessor_a, CorpusPreProcessor_a
from ..arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
AMAZON_PROCESSORS = [TrainPreProcessor_a, QueryPreProcessor_a, CorpusPreProcessor_a]
PROCESSOR_INFO = {
    'Tevatron/wikipedia-nq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-trivia': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-curated': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-wq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-squad': DEFAULT_PROCESSORS,
    'Tevatron/scifact': DEFAULT_PROCESSORS,
    'Tevatron/msmarco-passage': DEFAULT_PROCESSORS,
    'amazon': AMAZON_PROCESSORS,
    'json': [None, None, None]  # 对于不指定dataset_name的默认，不进行处理
}


class HFTrainDataset:
    """
    https://www.cxybb.com/article/qq_56591814/120653752#122_JSON_63
    https://huggingface.co/docs/datasets/loading
    dataset = load_dataset("json", data_files="my_file.json") json格式直接读取
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files} # {'train': 'amazon_data_tokenized/queries_train.json'}

        dataset_name = data_args.dataset_name
        if data_args.dataset_name == 'amazon':
            dataset_name = 'json'
        
        self.dataset = load_dataset(dataset_name, # ‘json'
                                    data_args.dataset_language, # 'default'
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]  # dataset['train']
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO\
            else DEFAULT_PROCESSORS[0]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx) # num_shards：把数据集划分几份，index 要返回哪个分片
        if self.preprocessor is not None:
            # remove_columns = self.dataset.column_names
            remove_columns = []
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=remove_columns,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        
        dataset_name = data_args.dataset_name
        if data_args.dataset_name == 'amazon':
            dataset_name = 'json'

        self.dataset = load_dataset(dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            # remove_columns = self.dataset.column_names
            remove_columns = []
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=remove_columns,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        
        dataset_name = data_args.dataset_name
        if data_args.dataset_name == 'amazon':
            dataset_name = 'json'
        
        self.dataset = load_dataset(dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        # script_prefix = data_args.dataset_name
        # if script_prefix.endswith('-corpus'):
        #     script_prefix = script_prefix[:-7]
        # self.preprocessor = PROCESSOR_INFO[script_prefix][2] \
        #     if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][2] if data_args.dataset_name in PROCESSOR_INFO\
            else DEFAULT_PROCESSORS[2]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            # remove_columns = self.dataset.column_names
            remove_columns = []
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=remove_columns,
                desc="Running tokenization",
            )
        return self.dataset
