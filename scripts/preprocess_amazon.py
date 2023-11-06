# -*- coding: utf-8 -*-

# 这个文件是用来将整个 ide 项目部署到 d2 的时候做入口文件使用的
# 使用文档见链接 https://yuque.antfin.com/aii/aistudio/cgmmax

import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict

# 读取原始的数据集
import pandas as pd
df_examples = pd.read_parquet('/path/downloads/amazon_ori/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('/path/downloads/amazon_ori/shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("/path/downloads/amazon_ori/shopping_queries_dataset_sources.csv")

df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)
df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
df_task_large = df_examples_products[df_examples_products["large_version"] == 1]
df_products_en = df_products[df_products["product_locale"] == 'us']

df_task_en_small = df_task_1[df_task_1["product_locale"] == 'us']
df_task_en_large = df_task_large[df_task_large["product_locale"] == 'us']

# preprocess corpus
# product_id_dict = defaultdict(int)
# with jsonlines.open("/path/downloads/amazon_larger_version/amazon_corpus.jsonl", 'w') as w:
#     for index, row in tqdm(df_products_en.iterrows()):
#         product_id = row['product_id'].strip()
#         if product_id not in product_id_dict:
#             product_id_dict[product_id] = 0
#         else:
#             continue
#         product_title = "None" if row["product_title"] is None else row["product_title"].strip()
#         product_title = product_title.replace("\n","")
#         product_description = "None" if row["product_description"] is None else row["product_description"].strip()
#         product_description = product_description.replace("\n","")
#         product_bullet_point = "None" if row["product_bullet_point"] is None else row["product_bullet_point"].strip()
#         product_bullet_point = product_bullet_point.replace("\n","")
#         product_brand = "None" if row["product_brand"] is None else row["product_brand"].strip()
#         product_brand = product_brand.replace("\n","")
#         product_color = "None" if row["product_color"] is None else row["product_color"].strip()
#         product_color = product_color.replace("\n","")
        
#         instance = {
#             'product_id': product_id,
#             "product_title": product_title,
#             "product_description": product_description,
#             "product_bullet_point": product_bullet_point,
#             "product_brand": product_brand,
#             "product_color": product_color
#         }
#         w.write(instance)
# print("corpus size: ", len(product_id_dict))
        
# preprocess query
df_examples_us = df_examples[df_examples["product_locale"] == 'us']
df_examples_us_source = pd.merge(
    df_examples_us,
    df_sources,
    how='left',
    left_on=['query_id'],
    right_on=['query_id']
)

train_query_id_dict = defaultdict(int)
test_query_id_dict = defaultdict(int)

with jsonlines.open("/path/downloads/amazon_larger_version/train_query.jsonl", 'w') as w:
    df_examples_us_source_train = df_examples_us_source[df_examples_us_source['split']=='train']
    for index, row in tqdm(df_examples_us_source_train.iterrows()):
        query_id = row['query_id']
        if query_id not in train_query_id_dict:
            train_query_id_dict[query_id] = 0
        else:
            continue

        query_text = row['query'].strip()
        query_source = row['source'].strip()
        instance = {
            "query_id" : query_id,
            "query_text" : query_text,
            "query_type" : query_source
        }
        w.write(instance)
print("train query size: ", len(train_query_id_dict))
