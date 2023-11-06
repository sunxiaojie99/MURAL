class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}

class TrainPreProcessor_a:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, item):
        tokenizer = self.tokenizer
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
        return item

class QueryPreProcessor_a:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, item):
        tokenizer = self.tokenizer
        item['q_text_tokenized'] = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(item["q_text"]))
        return item


class CorpusPreProcessor_a:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, item):
        tokenizer = self.tokenizer
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
        return dict_tmp