# 读取原始的数据集
import pandas as pd
from collections import defaultdict
import json
import jsonlines
from tqdm import tqdm
import numpy as np
from multiprocessing import Manager
from datasets import load_dataset
from transformers import BertTokenizer
import jsonlines
import random


# df_examples = pd.read_parquet(
#     '/path/downloads/amazon_ori/shopping_queries_dataset_examples.parquet')
# df_products = pd.read_parquet(
#     '/path/downloads/amazon_ori/shopping_queries_dataset_products.parquet')
# df_sources = pd.read_csv(
#     "/path/downloads/amazon_ori/shopping_queries_dataset_sources.csv")

# df_examples_products = pd.merge(
#     df_examples,
#     df_products,
#     how='left',
#     left_on=['product_locale', 'product_id'],
#     right_on=['product_locale', 'product_id']
# )
# df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]

# df_task_en = df_task_1[df_task_1["product_locale"] == 'us']

# df_task_1_train = df_task_en[df_task_en["split"] == "train"]
# df_task_1_test = df_task_en[df_task_en["split"] == "test"]


# print('train data:', len(df_task_1_train))
# print('test data:', len(df_task_1_test))
# print('all data:', len(df_task_en))

# step1: preprocess corpus


def make_corpus():
    """
    train data q-d pair: 419653
    test data q-d pair: 181701
    all data q-d pair: 601354
    small版本的train+test 训练数据条数： 601354

    corpus size:  482105
    """
    print('small版本的train+test 训练数据条数：', len(df_task_en))
    product_id_dict = defaultdict(int)
    with jsonlines.open("/path/downloads/amazon_smaller_version/amazon_corpus.jsonl", 'w') as w:
        for index, row in tqdm(df_task_en.iterrows()):
            product_id = row['product_id'].strip()
            if product_id not in product_id_dict:
                product_id_dict[product_id] = 0
            else:
                continue
            product_title = "" if row["product_title"] is None or row["product_title"].lower(
            ) == 'none' else row["product_title"].strip()

            product_description = "" if row["product_description"] is None or row["product_description"].lower(
            ) == 'none' else row["product_description"].strip()

            product_bullet_point = "" if row["product_bullet_point"] is None or row["product_bullet_point"].lower(
            ) == 'none' else row["product_bullet_point"].strip()

            product_brand = "" if row["product_brand"] is None or row["product_brand"].lower(
            ) == 'none' else row["product_brand"].strip()

            product_color = "" if row["product_color"] is None or row["product_color"].lower(
            ) == 'none' else row["product_color"].strip()

            instance = {
                'product_id': product_id,
                "product_title": product_title,
                "product_description": product_description,
                "product_bullet_point": product_bullet_point,
                "product_brand": product_brand,
                "product_color": product_color
            }
            w.write(instance)
    print("corpus size: ", len(product_id_dict))


def makde_corpus_with_cate():
    file_list = [
        '/path/downloads/craw/doc_cat/part0_with_cat.jsonl',
        '/path/downloads/craw/doc_cat/part1_with_cat.jsonl',
        '/path/downloads/craw/doc_cat/part2_with_cat.jsonl',
        '/path/downloads/craw/doc_cat/part3_with_cat.jsonl',
        '/path/downloads/craw/doc_cat/part4_with_cat.jsonl'
    ]
    id2cat = {}
    cat_len2doc_count = {}  # 不同类目层级，有多少个doc
    cate_2id_dict = [{}, {}, {}, {}, {}, {}, {}, {}, {}]  # 9级类目的字典
    for file in file_list:
        with open(file, "r+", encoding="utf8") as f:
            print('process: {}'.format(file))
            for item in jsonlines.Reader(f):
                craw_cat_list = item['craw_cat_list']
                if len(craw_cat_list) == 0:  # if not have cat, pass
                    continue
                doc_id = item['doc_id']
                if doc_id in id2cat:
                    print('already have!', doc_id,
                          craw_cat_list, id2cat[doc_id])
                id2cat[doc_id] = craw_cat_list
                if len(craw_cat_list) not in cat_len2doc_count:
                    cat_len2doc_count[len(craw_cat_list)] = 0
                cat_len2doc_count[len(craw_cat_list)] += 1
                for idx, cate in enumerate(craw_cat_list):
                    if cate not in cate_2id_dict[idx]:
                        cate_2id_dict[idx][cate] = len(cate_2id_dict[idx])
    print('找到类目的doc 数目:', len(id2cat))
    print('cat_len2doc_count:', cat_len2doc_count)

    for idx, catedict in enumerate(cate_2id_dict):
        if '' not in catedict:  # 添加一个空类别
            catedict[''] = len(catedict)
        dict_path = '/path/downloads/craw/cate{}2id.json'.format(
            idx+1)
        print('cate{} dict len:'.format(idx+1), len(catedict))

        with open(dict_path, 'w', encoding='utf-8') as f_out:
            json.dump(catedict, f_out, ensure_ascii=False, indent=2)

    print('small版本的train+test 训练数据条数：', len(df_task_en))
    product_id_dict = defaultdict(int)
    not_empty_dict = {
        'product_title': 0,
        'product_description': 0,
        'product_bullet_point': 0,
        'product_brand': 0,
        'product_color': 0,
        'product_cate': 0
    }
    len_dict = {
        'product_title': [],
        'product_description': [],
        'product_bullet_point': [],
        'product_brand': [],
        'product_color': [],
        'product_cate': []
    }
    with jsonlines.open("/path/downloads/amazon_smaller_version/amazon_corpus_with_cate.jsonl", 'w') as w:
        for index, row in tqdm(df_task_en.iterrows()):
            product_id = row['product_id'].strip()
            if product_id not in product_id_dict:
                product_id_dict[product_id] = 0
            else:
                continue
            product_title = "" if row["product_title"] is None or row["product_title"].lower(
            ) == 'none' else row["product_title"].strip()

            product_description = "" if row["product_description"] is None or row["product_description"].lower(
            ) == 'none' else row["product_description"].strip()

            product_bullet_point = "" if row["product_bullet_point"] is None or row["product_bullet_point"].lower(
            ) == 'none' else row["product_bullet_point"].strip()

            product_brand = "" if row["product_brand"] is None or row["product_brand"].lower(
            ) == 'none' else row["product_brand"].strip()

            product_color = "" if row["product_color"] is None or row["product_color"].lower(
            ) == 'none' else row["product_color"].strip()

            instance = {
                'product_id': product_id,
                "product_title": product_title,
                "product_description": product_description,
                "product_bullet_point": product_bullet_point,
                "product_brand": product_brand,
                "product_color": product_color,
                'product_cate': id2cat[product_id] if product_id in id2cat else []
            }
            for k in not_empty_dict.keys():
                if len(instance[k]) != 0:
                    not_empty_dict[k] += 1
            for k in len_dict.keys():
                if len(instance[k]) == 0:  # 不统计空的
                    continue
                if type(instance[k]) == list:
                    len_dict[k].append(len(instance[k][0]))
                else:
                    len_dict[k].append(len(instance[k]))
            w.write(instance)
    print("corpus size: ", len(product_id_dict))
    print('not_empty_dict:', not_empty_dict)

    for k in len_dict.keys():
        print('avg for {}: '.format(k), np.mean(len_dict[k]))
        print('min for {}: '.format(k), np.min(len_dict[k]))
        print('max for {}: '.format(k), np.max(len_dict[k]))


def clean_data():
    from clean import clean
    input_file = '/path/downloads/amazon_smaller_version/amazon_corpus_with_cate.jsonl'
    with jsonlines.open("/path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl", 'w') as w:
        with open(input_file, "r+", encoding="utf8") as f:
            print('process: {}'.format(input_file))
            for item in tqdm(jsonlines.Reader(f)):
                instance = {
                    'product_id': item['product_id'],
                    "product_title": clean(item['product_title']),
                    "product_description": clean(item['product_description']),
                    "product_bullet_point": clean(item['product_bullet_point']),
                    "product_brand": clean(item['product_brand']),
                    "product_color": clean(item['product_color']),
                    'product_cate': item['product_cate']
                }
                w.write(instance)


def make_train_data():
    """
    q_rel 中 query的数量： 20888
    """
    f_query_info = open(
        '/path/downloads/amazon_smaller_version/train_query_info.jsonl', 'w', encoding='utf-8')
    f_qrel_info = open(
        '/path/downloads/amazon_smaller_version/train_dev_qrel.txt', 'w', encoding='utf-8')
    all_query_dict = {}  # 全部query的集合
    qrel = {}

    num_dict = {
        'E_list': [],
        'S_list': [],
        'C_list': [],
        'I_list': []
    }

    for index, row in tqdm(df_task_1_train.iterrows()):
        product_id = row["product_id"].strip()

        query_id = row["query_id"]
        query = row["query"].strip()

        if query_id not in all_query_dict:  # 没有出现过的query集
            q_info = {
                'qid': query_id,
                'q_text': query
            }
            all_query_dict[query_id] = q_info  # 存字典
            f_query_info.write(json.dumps(q_info, ensure_ascii=False))
            f_query_info.write("\n")

        if query_id not in qrel:
            qrel[query_id] = {}
            qrel[query_id]['qid'] = query_id
            qrel[query_id]['q_text'] = query
            qrel[query_id]['E_list'] = []
            qrel[query_id]['S_list'] = []
            qrel[query_id]['C_list'] = []
            qrel[query_id]['I_list'] = []

        if product_id not in qrel[query_id]['{}_list'.format(row["esci_label"])]:
            qrel[query_id]['{}_list'.format(
                row["esci_label"])].append(product_id)

    print('q_rel 中 query的数量：', len(qrel))
    for query_id, qrel_detail in qrel.items():
        if len(qrel_detail['E_list']) == 0:  # 没有正样本
            continue
        for k in num_dict.keys():
            num_dict[k].append(len(qrel_detail[k]))
        f_qrel_info.write(json.dumps(
            {
                'q_text': qrel_detail['q_text'],
                'q_id': query_id,
                'E_list': qrel_detail['E_list'],
                'S_list': qrel_detail['S_list'],
                'C_list': qrel_detail['C_list'],
                'I_list': qrel_detail['I_list']
            }, ensure_ascii=False
        ))
        f_qrel_info.write("\n")
    f_query_info.close()
    f_qrel_info.close()
    for k in num_dict.keys():
        print('avg for {}: '.format(k), np.mean(num_dict[k]))
        print('min for {}: '.format(k), np.min(num_dict[k]))
        print('max for {}: '.format(k), np.max(num_dict[k]))


def make_test_data():
    """
    test q_rel 中 query的数量： 8956
    """
    test_qrel = {}
    for index, row in tqdm(df_task_1_test.iterrows()):
        product_id = row["product_id"].strip()

        query_id = row["query_id"]
        query = row["query"].strip()

        if query_id not in test_qrel:
            test_qrel[query_id] = {}
            test_qrel[query_id]['qid'] = query_id
            test_qrel[query_id]['q_text'] = query
            test_qrel[query_id]['E_list'] = []
            test_qrel[query_id]['S_list'] = []
            test_qrel[query_id]['C_list'] = []
            test_qrel[query_id]['I_list'] = []

        if product_id not in test_qrel[query_id]['{}_list'.format(row["esci_label"])]:
            test_qrel[query_id]['{}_list'.format(
                row["esci_label"])].append(product_id)

    f_qrel_info = open(
        '/path/downloads/amazon_smaller_version/test_qrel.txt', 'w', encoding='utf-8')
    print('test q_rel 中 query的数量：', len(test_qrel))
    num_dict = {
        'E_list': [],
        'S_list': [],
        'C_list': [],
        'I_list': []
    }
    for q_id, qrel_detail in test_qrel.items():
        if len(qrel_detail['E_list']) == 0:  # 没有正样本
            continue
        for k in num_dict.keys():
            num_dict[k].append(len(qrel_detail[k]))
        f_qrel_info.write(json.dumps(
            {
                'q_id': q_id,
                'q_text': qrel_detail['q_text'],
                'E_list': qrel_detail['E_list'],
                'S_list': qrel_detail['S_list'],
                'C_list': qrel_detail['C_list'],
                'I_list': qrel_detail['I_list']
            }, ensure_ascii=False
        ))
        f_qrel_info.write("\n")
    for k in num_dict.keys():
        print('avg for {}: '.format(k), np.mean(num_dict[k]))
        print('min for {}: '.format(k), np.min(num_dict[k]))
        print('max for {}: '.format(k), np.max(num_dict[k]))


def split_train_dev_test():
    f = open('/path/downloads/amazon_smaller_version/train_dev_qrel.txt',
             'r', encoding='utf-8')
    ori_lines = f.readlines()
    f.close()
    print('ori整体的query数量：', len(ori_lines))
    lines = []
    for line in ori_lines:
        item = eval(line)
        if len(item['E_list']) == 0:
            continue
        # if len(item['pos_list']) == 0:
        #     continue
        lines.append(item)
    print('过滤掉pos_list=[] or neg_list=[]后，整体的query数量：', len(lines))
    np.random.seed(42)  # 保证每次第一次np.random.permutation得到的结果一致
    shuffled_indices = np.random.permutation(len(lines))  # 生成和原数据等长的无序索引

    dev_indices = shuffled_indices[:3500]
    train_indices = shuffled_indices[3500:]
    print(len(dev_indices), len(train_indices))

    f_out_dev = open(
        '/path/downloads/amazon_smaller_version/dev_qrel.txt', 'w', encoding='utf-8')
    f_out_train = open(
        '/path/downloads/amazon_smaller_version/train_qrel.txt', 'w', encoding='utf-8')

    for idx, item in enumerate(lines):
        if idx in train_indices:
            f_out_train.write(json.dumps(item, ensure_ascii=False))
            f_out_train.write('\n')
        elif idx in dev_indices:
            f_out_dev.write(json.dumps(item, ensure_ascii=False))
            f_out_dev.write('\n')
        else:
            print('wrong idx!', idx)
    f_out_dev.close()
    f_out_train.close()


def accord_qrel_generate_detail(doc_info, need_qrel_file, out_qrel_detail_file, handle_type='train'):
    manager = Manager()
    docid_to_idx = manager.dict(
        {k: v for v, k in enumerate(doc_info['product_id'])})

    f_in = open(need_qrel_file, 'r', encoding='utf-8')
    all_qrels = f_in.readlines()
    f_in.close()

    print(handle_type + ' qrel 数量：', len(all_qrels))

    f_detail_out = open(out_qrel_detail_file, 'w', encoding='utf-8')
    no_find_docid_set = set()

    valid_count = 0
    middle_doc_count = 0
    for line in tqdm(all_qrels):
        item = eval(line)
        item['E_detail_list'] = []
        item['S_detail_list'] = []
        item['C_detail_list'] = []
        item['I_detail_list'] = []

        Level_list = ['E', 'S', 'C', 'I']
        for level in Level_list:
            for docid in item['{}_list'.format(level)]:
                if docid not in docid_to_idx:
                    no_find_docid_set.add(docid)
                else:
                    item['{}_detail_list'.format(level)].append(
                        doc_info[docid_to_idx[docid]])
            assert len(item['{}_list'.format(level)]) == len(
                item['{}_detail_list'.format(level)])

        f_detail_out.write(json.dumps(
            item, ensure_ascii=False
        ))
        f_detail_out.write("\n")
        valid_count += 1

    f_detail_out.close()

    print('<q,d->同时也是<q,d+>的数量：', middle_doc_count)

    print('no find docid count:', len(no_find_docid_set))
    print(handle_type + '过滤完失效的docid 后 qrel 的数量', valid_count)


def generate_deatil():
    corpus_file = '/path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl'
    corpus_dataset = load_dataset('json',  # ‘json'
                                  'default',  # 'default'
                                  data_files={'train': corpus_file})['train']

    train_qrel_file = '/path/downloads/amazon_smaller_version/train_qrel.txt'
    train_qrel_detail_file = '/path/downloads/amazon_smaller_version/train_qrel_detail.txt'
    accord_qrel_generate_detail(
        corpus_dataset, train_qrel_file, train_qrel_detail_file, 'train')

    dev_qrel_file = '/path/downloads/amazon_smaller_version/dev_qrel.txt'
    dev_qrel_detail_file = '/path/downloads/amazon_smaller_version/dev_qrel_detail.txt'
    accord_qrel_generate_detail(
        corpus_dataset, dev_qrel_file, dev_qrel_detail_file, 'dev')

    test_qrel_file = '/path/downloads/amazon_smaller_version/test_qrel.txt'
    test_qrel_detail_file = '/path/downloads/amazon_smaller_version/test_qrel_detail.txt'
    accord_qrel_generate_detail(
        corpus_dataset, test_qrel_file, test_qrel_detail_file, 'test')


def tokenize_for_qrel_detail(tokenizer, input_file, output_file):
    print('process: {}'.format(input_file))
    count = 0

    with open(output_file, 'w') as outFile:
        with open(input_file, "r+", encoding="utf8") as f:
            for item in tqdm(jsonlines.Reader(f)):
                item['q_text_tokenized'] = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(item["q_text"]))
                item['E_tokenize_list'] = []
                item['S_tokenize_list'] = []
                item['C_tokenize_list'] = []
                item['I_tokenize_list'] = []
                level_list = ['E', 'S', 'C', 'I']
                for level in level_list:
                    level_name = '{}_detail_list'.format(level)
                    for i in range(len(item[level_name])):
                        item_dict = {
                            'product_title': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[level_name][i]["product_title"])),
                            'product_description': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[level_name][i]["product_description"])),
                            'product_bullet_point': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[level_name][i]["product_bullet_point"])),
                            'product_brand': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[level_name][i]["product_brand"])),
                            'product_color': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[level_name][i]["product_color"])),
                            'product_cate': [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c)) for c in item[level_name][i]['product_cate']]
                        }
                        item['{}_tokenize_list'.format(
                            level)].append(item_dict)
                    assert len(item[level_name]) == len(
                        item['{}_tokenize_list'.format(level)])
                outFile.write(json.dumps(item, ensure_ascii=False))
                outFile.write("\n")
                count += 1

    print('process over! count={}'.format(count))


def tokenize_for_corpus(tokenizer, input_file, output_file):
    print('process: {}'.format(input_file))

    count = 0
    with open(output_file, 'w') as outFile:
        with open(input_file, "r+", encoding="utf8") as f:
            for item in tqdm(jsonlines.Reader(f)):
                dict_tmp = {}
                dict_tmp["doc_info"] = item
                dict_tmp["doc_info_tokenize"] = {
                    'product_title': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["product_title"])),
                    'product_description': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["product_description"])),
                    'product_bullet_point': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["product_bullet_point"])),
                    'product_brand': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["product_brand"])),
                    'product_color': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["product_color"])),
                    'product_cate': [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c)) for c in item['product_cate']]
                }
                outFile.write(json.dumps(dict_tmp, ensure_ascii=False))
                outFile.write("\n")
                count += 1
    print('process over! count={}'.format(count))


def make_brand_color_dict(brand_num=18, color_num=7):

    def check_num2count(product_dict, save_path=None):
        example_num2brand_num = {}
        for brand, example_num in product_dict.items():
            if example_num not in example_num2brand_num:
                example_num2brand_num[example_num] = 0
            example_num2brand_num[example_num] += 1
        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as f:
                for example_num, brand_num in example_num2brand_num.items():
                    f.write(str(example_num) + '\t' + str(brand_num) + '\n')
        return

    def save_dict(save_path, dict_need):
        import json
        with open(save_path, 'w', encoding='utf-8') as f_out:
            json.dump(dict_need, f_out, ensure_ascii=False, indent=2)

    def filter_low_brand_or_color(product_dict, threshold=0, name='brand'):
        """输入check_brand_and_color 得到的dict，对低频次的key进行过滤，返回新的key"""
        product_dict_more = {}
        len_list = []
        int_id = 0
        for brand in product_dict:
            if product_dict[brand] >= threshold and brand.isdigit() is False:  # 去掉纯数字
                product_dict_more[brand] = int_id
                int_id += 1
                len_list.append(len(brand))
        print('avg for {}: '.format(name), np.mean(len_list))
        print('min for {}: '.format(name), np.min(len_list))
        print('max for {}: '.format(name), np.max(len_list))
        return product_dict_more

    product_brand_dict2 = {}
    product_color_dict2 = {}
    input_file = '/path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl'
    with open(input_file, "r+", encoding="utf8") as f:
        print('process: {}'.format(input_file))
        for item in tqdm(jsonlines.Reader(f)):
            brand = item['product_brand'].lower()
            color = item['product_color'].lower()
            if brand not in product_brand_dict2:
                product_brand_dict2[brand] = 0
            if color not in product_color_dict2:
                product_color_dict2[color] = 0
            product_brand_dict2[brand] += 1
            product_color_dict2[color] += 1
    print('brand:', len(product_brand_dict2))
    print('color:', len(product_color_dict2))
    # check_num2count(product_brand_dict2, '/path/downloads/amazon_ori/ori_example_num2brand_num.csv')
    # check_num2count(product_color_dict2, '/path/downloads/amazon_ori/ori_example_num2brand_num.csv')
    product_brand_dict_14 = filter_low_brand_or_color(
        product_brand_dict2, brand_num, name='brand')
    product_color_dict_6 = filter_low_brand_or_color(
        product_color_dict2, color_num, name='color')
    if '' not in product_brand_dict_14:
        product_brand_dict_14[''] = len(product_brand_dict_14)
    if '' not in product_color_dict_6:
        product_color_dict_6[''] = len(product_color_dict_6)
    print('===过滤后的 train+test 的 brand 和 color 数量')
    print('brand_{}:'.format(brand_num), len(product_brand_dict_14))
    print('color_{}:'.format(color_num), len(product_color_dict_6))
    save_dict('/path/downloads/amazon_ori/product_brand2id_' +
              str(brand_num)+'.json', product_brand_dict_14)
    save_dict('/path/downloads/amazon_ori/product_color2id_' +
              str(color_num)+'.json', product_color_dict_6)


def static_cate():
    file = '/path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl'
    cate_dict = {
        '>=1': 0,
        '>=2': 0,
        '>=3': 0,
        '>=4': 0,
        '>=5': 0,
        '>=6': 0,
        '>=7': 0,
        '>=8': 0,
        '>=9': 0,
        '>=10': 0,
    }
    not_empty_dict = {
        'brand_not_empty': 0,
        'color_not_empty': 0
    }

    all_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_doc_num = 0
    with open(file, 'r', encoding='utf8') as f:
        for item in tqdm(jsonlines.Reader(f)):
            all_doc_num += 1
            cate_len = len(item['product_cate'])
            for num in all_num:
                if cate_len >= num:
                    cate_dict['>={}'.format(num)] += 1
            if item['product_brand'] != "":
                not_empty_dict['brand_not_empty'] += 1
            if item['product_color'] != "":
                not_empty_dict['color_not_empty'] += 1

    for key, value in cate_dict.items():
        print(key, value, all_doc_num, round(value / all_doc_num, 2))
    
    for key, value in not_empty_dict.items():
        print(key, value, all_doc_num, round(value / all_doc_num, 2))
    
    

def load_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dict(save_path, dict_need):
        import json
        with open(save_path, 'w', encoding='utf-8') as f_out:
            json.dump(dict_need, f_out, ensure_ascii=False, indent=2)

def generate_category_word_vocab():
    """
    /path/downloads/craw/cate12id.json 59
    /path/downloads/craw/cate22id.json 455
    /path/downloads/craw/cate32id.json 2303
    /path/downloads/craw/cate42id.json 5277
    /path/downloads/craw/cate52id.json 4606
    /path/downloads/craw/cate62id.json 1436
    /path/downloads/craw/cate72id.json 228
    /path/downloads/craw/cate82id.json 29
    /path/downloads/craw/cate92id.json 5
    """
    import re
    regex = r'[^\w\s]|[\s]'
    
    # cate_dict_list = {}
    # for i in range(1, 10):
    #     file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
    #     cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
    #     print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    # word_to_id = {}
    # for cate_dict in cate_dict_list.values():
    #     for cate in cate_dict.keys():
    #         # 使用 re 模块的 split() 方法来以标点符号分割字符串
    #         words = re.split(regex, cate)
    #         for word in words:
    #             if word.strip():
    #                 if word not in word_to_id:
    #                     word_to_id[word] = len(word_to_id)
    
    # save_dict('/path/downloads/craw/cate_word_vocab.json', word_to_id)

    # brand_dict =load_dict('/path/downloads/amazon_ori/product_brand2id_13.json')

    # word_to_id = {}
    # for brand in tqdm(brand_dict.keys()):
    #     words = re.split(regex, brand)
    #     for word in words:
    #         if word.strip():
    #             if word not in word_to_id:
    #                 word_to_id[word] = len(word_to_id)
    # save_dict('/path/downloads/craw/brand_word_vocab.json', word_to_id)

    # color_dict = load_dict('/path/downloads/amazon_ori/product_color2id_7.json')
    # word_to_id = {}
    # for color in color_dict.keys():
    #     words = re.split(regex, color)
    #     for word in words:
    #         if word.strip():
    #             if word not in word_to_id:
    #                 word_to_id[word] = len(word_to_id)
    # save_dict('/path/downloads/craw/color_word_vocab.json', word_to_id)

    cate_dict_list = {}
    for i in range(1, 5):
        file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
        cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
        print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    word_to_id = {}
    for cate_dict in cate_dict_list.values():
        for cate in cate_dict.keys():
            # 使用 re 模块的 split() 方法来以标点符号分割字符串
            words = re.split(regex, cate)
            for word in words:
                if word.strip():
                    if word not in word_to_id:
                        word_to_id[word] = len(word_to_id)
    
    save_dict('/path/downloads/craw/cate1-4_word_vocab.json', word_to_id)


def generate_whole_aspect_dict():
    """
    /path/downloads/craw/cate12id.json 59
    /path/downloads/craw/cate22id.json 455
    /path/downloads/craw/cate32id.json 2303
    /path/downloads/craw/cate42id.json 5277
    /path/downloads/craw/cate52id.json 4606
    /path/downloads/craw/cate62id.json 1436
    /path/downloads/craw/cate72id.json 228
    /path/downloads/craw/cate82id.json 29
    /path/downloads/craw/cate92id.json 5
    """
    # cate_dict_list = {}
    # for i in range(1, 10):
    #     file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
    #     cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
    #     print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    # word_to_id = {}
    # for cate_dict in cate_dict_list.values():
    #     for cate in cate_dict.keys():
    #         # cate = cate.lower()
    #         if cate and cate not in word_to_id:
    #             word_to_id[cate] = len(word_to_id)
    
    # save_dict('/path/downloads/craw/whole_cate_vocab.json', word_to_id)
    # 12895 -> 12894

    cate_dict_list = {}
    for i in range(1, 5):
        file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
        cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
        print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    word_to_id = {}
    for cate_dict in cate_dict_list.values():
        for cate in cate_dict.keys():
            # cate = cate.lower()
            if cate and cate not in word_to_id:
                word_to_id[cate] = len(word_to_id)
    
    save_dict('/path/downloads/craw/whole_cate1-4_vocab.json', word_to_id)


def generate_category_wordpiece_vocab():
    """
    /path/downloads/craw/cate12id.json 59
    /path/downloads/craw/cate22id.json 455
    /path/downloads/craw/cate32id.json 2303
    /path/downloads/craw/cate42id.json 5277
    /path/downloads/craw/cate52id.json 4606
    /path/downloads/craw/cate62id.json 1436
    /path/downloads/craw/cate72id.json 228
    /path/downloads/craw/cate82id.json 29
    /path/downloads/craw/cate92id.json 5
    """
    tokenizer = BertTokenizer.from_pretrained("/path/downloads/bert-base-uncased")
    import re
    regex = r'[^\w\s]|[\s]'
    
    # cate_dict_list = {}
    # for i in range(1, 10):
    #     file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
    #     cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
    #     print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    # word_to_id = {}
    # for cate_dict in cate_dict_list.values():
    #     for cate in cate_dict.keys():
    #         # 使用 re 模块的 split() 方法来以标点符号分割字符串
    #         words = re.split(regex, cate)
    #         for word in words:
    #             if word.strip():
    #                 for w in tokenizer.tokenize(word):
    #                     if w not in word_to_id:
    #                         word_to_id[w] = len(word_to_id)
    
    # save_dict('/path/downloads/craw/cate_wordpiece_vocab.json', word_to_id)

    # brand_dict =load_dict('/path/downloads/amazon_ori/product_brand2id_13.json')

    # word_to_id = {}
    # for brand in tqdm(brand_dict.keys()):
    #     words = re.split(regex, brand)
    #     for word in words:
    #         if word.strip():
    #             for w in tokenizer.tokenize(word):
    #                 if w not in word_to_id:
    #                     word_to_id[w] = len(word_to_id)
    # save_dict('/path/downloads/craw/brand_wordpiece_vocab.json', word_to_id)

    # color_dict = load_dict('/path/downloads/amazon_ori/product_color2id_7.json')
    # word_to_id = {}
    # for color in color_dict.keys():
    #     words = re.split(regex, color)
    #     for word in words:
    #         if word.strip():
    #             for w in tokenizer.tokenize(word):
    #                 if w not in word_to_id:
    #                     word_to_id[w] = len(word_to_id)
    # save_dict('/path/downloads/craw/color_wordpiece_vocab.json', word_to_id)

    cate_dict_list = {}
    for i in range(1, 5):
        file_dict = '/path/downloads/craw/cate{}2id.json'.format(i)
        cate_dict_list['cate{}2id'.format(i)] = load_dict(file_dict)
        print(file_dict, len(cate_dict_list['cate{}2id'.format(i)]))
    
    word_to_id = {}
    for cate_dict in cate_dict_list.values():
        for cate in cate_dict.keys():
            # 使用 re 模块的 split() 方法来以标点符号分割字符串
            words = re.split(regex, cate)
            for word in words:
                if word.strip():
                    for w in tokenizer.tokenize(word):
                        if w not in word_to_id:
                            word_to_id[w] = len(word_to_id)
    
    save_dict('/path/downloads/craw/cate1-4_wordpiece_vocab.json', word_to_id)

# step1
# makde_corpus_with_cate()
# clean_data()
# static_cate()  
# 统计cate的信息
# make_brand_color_dict(brand_num=13, color_num=7)  # 制作词典
# make_train_data()
# make_test_data()
# generate_category_word_vocab()

# step 2 根据qrel切分训练和测试集
# split_train_dev_test()

# 3.为qrel补充query和doc的信息，方便训练, 过滤失效doc id
# generate_deatil()


# 4. tokenize for step3
# tokenizer = BertTokenizer.from_pretrained(
#     "/path/downloads/bert-base-uncased")
# tokenize_for_qrel_detail(tokenizer, '/path/downloads/amazon_smaller_version/dev_qrel_detail.txt',
#                          '/path/downloads/amazon_smaller_version/tokenize/dev_qrel_detail.txt')
# tokenize_for_qrel_detail(tokenizer, '/path/downloads/amazon_smaller_version/test_qrel_detail.txt',
#                          '/path/downloads/amazon_smaller_version/tokenize/test_qrel_detail.txt')
# tokenize_for_qrel_detail(tokenizer, '/path/downloads/amazon_smaller_version/train_qrel_detail.txt',
#                          '/path/downloads/amazon_smaller_version/tokenize/train_qrel_detail.txt')

# tokenize_for_corpus(tokenizer, '/path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl',
#                     '/path/downloads/amazon_smaller_version/tokenize/amazon_corpus_with_cate_clean.jsonl')
# generate_whole_aspect_dict()
generate_category_wordpiece_vocab()