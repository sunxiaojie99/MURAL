import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--qrel_file')
args = parser.parse_args()

with open(args.qrel_file) as f:
    lines = f.readlines()

f_out = open(args.qrel_file.strip('.txt') + '.trec', 'w', encoding='utf-8')

with open(args.qrel_file, "r+", encoding="utf8") as f:
    count = 0
    for item in jsonlines.Reader(f):
        for doc_id in set(item['E_list']):
            f_out.write("{} {} {} {}\n".format(item['q_id'], '0', doc_id, 1))

# python /path/scripts/convert_to_trec_qrel.py --qrel_file /path/downloads/amazon_smaller_version/dev_qrel.txt