cd /path/src
model_type='madr'
prompt_type='three-sep'  # 类型为two-sep
q_pool_type='att'
doc_pool_type='att'

is_concat='no'  # 如果拼接作为doc输入，改为yes
is_concat_query='no'  # 如果拼接作为doc输入，改为yes

epoch=20
bs=64
lr=5e-6

aspect_mlm_prob=0.0  # mask 比例
cat_level=4  # !!!! 使用多少级的cat
# 设置为0 不计算rel loss
finetune_aspect_alpha=0.0  # 辅助loss系数, for mtbert\madr\prompt
aspect_loss_type='softmax-ignore'
dataset_name='amazon'  # 'amazon'

# for madr and mgrain
aspect_num=4
begin_id=0

train_path='/path/downloads/amazon_smaller_version/tokenize/train_qrel_detail.txt'
# === 1. 改动预训练模型名 + 上面的参数
model_path='/path/output/1e-4-o_madr-mul-softmax-a01-fix-app-no-cls/checkpoint-30140'
output_path='/path/output/1e-4-o_madr-mul-softmax-a01-fix-app-no-cls/ck-30140-app-with-cls-alpha'
bert_dir='/path/downloads/bert-base-uncased'
pretrain_model_name_or_path=$model_path
gating_type='first-k:no-cls'

gpu_id=0
# === 2. 改动评估的epoch 和 gpuid 1507
fine_alpha=(0.05 0.1)
for ckpt in ${fine_alpha[@]}
do
finetune_aspect_alpha=${ckpt}  # 辅助loss系数, for mtbert\madr\prompt
# CUDA_VISIBLE_DEVICES=$gpu_id python  -m tevatron.driver.train \
#   --output_dir ${output_path}/${ckpt} \
#   --tensorboard_output_dir ${output_path}/${ckpt}/tboard \
#   --tokenizer_name ${bert_dir} \
#   --model_name_or_path ${model_path} \
#   --train_dir $train_path \
#   --brand2id_path /path/downloads/amazon_ori/product_brand2id_13.json \
#   --color2id_path /path/downloads/amazon_ori/product_color2id_7.json \
#   --cate1_2id_path /path/downloads/craw/cate12id.json \
#   --cate2_2id_path /path/downloads/craw/cate22id.json \
#   --cate3_2id_path /path/downloads/craw/cate32id.json \
#   --cate4_2id_path /path/downloads/craw/cate42id.json \
#   --cate5_2id_path /path/downloads/craw/cate52id.json \
#   --whole_cate_vocab_file /path/downloads/craw/whole_cate1-4_vocab.json \
#   --cate_wordpiece_vocab_file /path/downloads/craw/cate1-4_wordpiece_vocab.json \
#   --brand_word_vocab_file /path/downloads/craw/brand_word_vocab.json \
#   --whole_brand_vocab_file /path/downloads/amazon_ori/product_brand2id_13.json \
#   --brand_wordpiece_vocab_file /path/downloads/craw/brand_wordpiece_vocab.json \
#   --color_word_vocab_file /path/downloads/craw/color_word_vocab.json \
#   --whole_color_vocab_file /path/downloads/amazon_ori/product_color2id_7.json \
#   --color_wordpiece_vocab_file /path/downloads/craw/color_wordpiece_vocab.json \
#   --save_steps 50 \
#   --add_pooler \
#   --per_device_train_batch_size $bs \
#   --train_n_passages 2 \
#   --learning_rate $lr \
#   --q_max_len 32 \
#   --p_max_len 156 \
#   --fp16 \
#   --num_train_epochs $epoch \
#   --logging_steps 500 \
#   --overwrite_output_dir \
#   --model_type $model_type \
#   --q_pool_type $q_pool_type \
#   --doc_pool_type $doc_pool_type \
#   --save_strategy no \
#   --is_concat $is_concat \
#   --is_concat_query $is_concat_query \
#   --aspect_mlm_prob $aspect_mlm_prob \
#   --finetune_aspect_alpha $finetune_aspect_alpha \
#   --cat_level $cat_level \
#   --aspect_num $aspect_num \
#   --prompt_type $prompt_type \
#   --aspect_loss_type $aspect_loss_type \
#   --dataset_name $dataset_name \
#   --begin_id $begin_id \
#   --gating_type $gating_type


#   --untie_encoder \

#   --negatives_x_device \


# ===== dev query
# mkdir ${output_path}/${ckpt}/encode
# CUDA_VISIBLE_DEVICES=$gpu_id python -m tevatron.driver.encode \
#   --output_dir=temp \
#   --model_type $model_type \
#   --tokenizer_name ${bert_dir} \
#   --config_name ${bert_dir} \
#   --model_name_or_path ${output_path}/${ckpt}  \
#   --pretrain_model_name_or_path $pretrain_model_name_or_path/${ckpt} \
#   --per_device_eval_batch_size 156 \
#   --encode_in_path /path/downloads/amazon_smaller_version/tokenize/dev_qrel_detail.txt \
#   --encoded_save_path ${output_path}/${ckpt}/encode/query_eval.pkl \
#   --q_max_len 32 \
#   --encode_is_qry \
#   --fp16 \
#   --q_pool_type $q_pool_type \
#   --doc_pool_type $doc_pool_type \
#   --brand2id_path /path/downloads/amazon_ori/product_brand2id_13.json \
#   --color2id_path /path/downloads/amazon_ori/product_color2id_7.json \
#   --cate1_2id_path /path/downloads/craw/cate12id.json \
#   --cate2_2id_path /path/downloads/craw/cate22id.json \
#   --cate3_2id_path /path/downloads/craw/cate32id.json \
#   --cate4_2id_path /path/downloads/craw/cate42id.json \
#   --cate5_2id_path /path/downloads/craw/cate52id.json \
#   --cate_word_vocab_file /path/downloads/craw/cate1-4_word_vocab.json \
#   --whole_cate_vocab_file /path/downloads/craw/whole_cate1-4_vocab.json \
#   --cate_wordpiece_vocab_file /path/downloads/craw/cate1-4_wordpiece_vocab.json \
#   --brand_word_vocab_file /path/downloads/craw/brand_word_vocab.json \
#   --whole_brand_vocab_file /path/downloads/amazon_ori/product_brand2id_13.json \
#   --brand_wordpiece_vocab_file /path/downloads/craw/brand_wordpiece_vocab.json \
#   --color_word_vocab_file /path/downloads/craw/color_word_vocab.json \
#   --whole_color_vocab_file /path/downloads/amazon_ori/product_color2id_7.json \
#   --color_wordpiece_vocab_file /path/downloads/craw/color_wordpiece_vocab.json \
#   --is_concat $is_concat \
#   --is_concat_query $is_concat_query \
#   --cat_level $cat_level \
#   --aspect_num $aspect_num \
#   --prompt_type $prompt_type \
#   --dataset_name $dataset_name \
#   --begin_id $begin_id \
#   --gating_type $gating_type


# ====== encode doc
CUDA_VISIBLE_DEVICES=$gpu_id python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_type $model_type \
  --model_name_or_path ${output_path}/${ckpt}  \
  --tokenizer_name ${bert_dir} \
  --config_name ${bert_dir} \
  --pretrain_model_name_or_path $pretrain_model_name_or_path/${ckpt} \
  --per_device_eval_batch_size 1280 \
  --p_max_len 156 \
  --fp16 \
  --encode_in_path /path/downloads/amazon_smaller_version/amazon_corpus_with_cate_clean.jsonl \
  --encoded_save_path ${output_path}/${ckpt}/encode/corpus_emb_all.pkl \
  --q_pool_type $q_pool_type \
  --doc_pool_type $doc_pool_type \
  --brand2id_path /path/downloads/amazon_ori/product_brand2id_13.json \
  --color2id_path /path/downloads/amazon_ori/product_color2id_7.json \
  --cate1_2id_path /path/downloads/craw/cate12id.json \
  --cate2_2id_path /path/downloads/craw/cate22id.json \
  --cate3_2id_path /path/downloads/craw/cate32id.json \
  --cate4_2id_path /path/downloads/craw/cate42id.json \
  --cate5_2id_path /path/downloads/craw/cate52id.json \
  --cate_word_vocab_file /path/downloads/craw/cate1-4_word_vocab.json \
  --whole_cate_vocab_file /path/downloads/craw/whole_cate1-4_vocab.json \
  --cate_wordpiece_vocab_file /path/downloads/craw/cate1-4_wordpiece_vocab.json \
  --brand_word_vocab_file /path/downloads/craw/brand_word_vocab.json \
  --whole_brand_vocab_file /path/downloads/amazon_ori/product_brand2id_13.json \
  --brand_wordpiece_vocab_file /path/downloads/craw/brand_wordpiece_vocab.json \
  --color_word_vocab_file /path/downloads/craw/color_word_vocab.json \
  --whole_color_vocab_file /path/downloads/amazon_ori/product_color2id_7.json \
  --color_wordpiece_vocab_file /path/downloads/craw/color_wordpiece_vocab.json \
  --is_concat $is_concat \
  --is_concat_query $is_concat_query \
  --cat_level $cat_level \
  --aspect_num $aspect_num \
  --prompt_type $prompt_type \
  --dataset_name $dataset_name \
  --begin_id $begin_id \
  --gating_type $gating_type \


# CUDA_VISIBLE_DEVICES=$gpu_id python -m tevatron.driver.encode \
#   --output_dir=temp \
#   --model_type $model_type \
#   --tokenizer_name ${bert_dir} \
#   --config_name ${bert_dir} \
#   --model_name_or_path ${output_path}/${ckpt}  \
#   --pretrain_model_name_or_path $pretrain_model_name_or_path \
#   --per_device_eval_batch_size 156 \
#   --encode_in_path /path/downloads/amazon_smaller_version/tokenize/test_qrel_detail.txt \
#   --encoded_save_path ${output_path}/${ckpt}/encode/test_query_eval.pkl \
#   --q_max_len 32 \
#   --encode_is_qry \
#   --fp16 \
#   --q_pool_type $q_pool_type \
#   --doc_pool_type $doc_pool_type \
#   --brand2id_path /path/downloads/amazon_ori/product_brand2id_13.json \
#   --color2id_path /path/downloads/amazon_ori/product_color2id_7.json \
#   --cate1_2id_path /path/downloads/craw/cate12id.json \
#   --cate2_2id_path /path/downloads/craw/cate22id.json \
#   --cate3_2id_path /path/downloads/craw/cate32id.json \
#   --cate4_2id_path /path/downloads/craw/cate42id.json \
#   --cate5_2id_path /path/downloads/craw/cate52id.json \
#   --cate_word_vocab_file /path/downloads/craw/cate1-4_word_vocab.json \
#   --whole_cate_vocab_file /path/downloads/craw/whole_cate1-4_vocab.json \
#   --cate_wordpiece_vocab_file /path/downloads/craw/cate1-4_wordpiece_vocab.json \
#   --brand_word_vocab_file /path/downloads/craw/brand_word_vocab.json \
#   --whole_brand_vocab_file /path/downloads/amazon_ori/product_brand2id_13.json \
#   --brand_wordpiece_vocab_file /path/downloads/craw/brand_wordpiece_vocab.json \
#   --color_word_vocab_file /path/downloads/craw/color_word_vocab.json \
#   --whole_color_vocab_file /path/downloads/amazon_ori/product_color2id_7.json \
#   --color_wordpiece_vocab_file /path/downloads/craw/color_wordpiece_vocab.json \
#   --is_concat $is_concat \
#   --is_concat_query $is_concat_query \
#   --cat_level $cat_level \
#   --aspect_num $aspect_num \
#   --prompt_type $prompt_type \
#   --dataset_name $dataset_name \
#   --begin_id $begin_id \
#   --gating_type $gating_type \


# # ====dev search
# python -m tevatron.faiss_retriever \
# --query_reps ${output_path}/${ckpt}/encode/query_eval.pkl \
# --passage_reps ${output_path}/${ckpt}/encode/corpus_emb_all*.pkl \
# --depth 1000 --batch_size -100 --save_text  \
# --save_ranking_to ${output_path}/${ckpt}/encode/all_rank_eval.txt

# python -m tevatron.faiss_retriever \
# --query_reps ${output_path}/${ckpt}/encode/test_query_eval.pkl \
# --passage_reps ${output_path}/${ckpt}/encode/corpus_emb_all*.pkl \
# --depth 1000 --batch_size -100 --save_text  \
# --save_ranking_to ${output_path}/${ckpt}/encode/test_all_rank_eval.txt

# echo ${output_path}/${ckpt}
# echo '=====dev'
# python /path/scripts/convert_to_trec_result.py --score_file ${output_path}/${ckpt}/encode/all_rank_eval.txt
# ./trec_eval -m recall /path/downloads/amazon_smaller_version/dev_qrel.trec \
#         ${output_path}/${ckpt}/encode/all_rank_eval.trec
# ./trec_eval -m recall.50 /path/downloads/amazon_smaller_version/dev_qrel.trec \
#         ${output_path}/${ckpt}/encode/all_rank_eval.trec
# ./trec_eval -m map /path/downloads/amazon_smaller_version/dev_qrel.trec \
#         ${output_path}/${ckpt}/encode/all_rank_eval.trec
# ./trec_eval -m recip_rank /path/downloads/amazon_smaller_version/dev_qrel.trec \
#         ${output_path}/${ckpt}/encode/all_rank_eval.trec
# ./trec_eval -m ndcg_cut /path/downloads/amazon_smaller_version/dev_qrel_mul.trec \
#         ${output_path}/${ckpt}/encode/all_rank_eval.trec
# ./trec_eval -m ndcg_cut.50 /path/downloads/amazon_smaller_version/dev_qrel_mul.trec \
        # ${output_path}/${ckpt}/encode/all_rank_eval.trec

# echo '=====test'
# python /path/scripts/convert_to_trec_result.py --score_file ${output_path}/${ckpt}/encode/test_all_rank_eval.txt
# ./trec_eval -m recall /path/downloads/amazon_smaller_version/test_qrel_detail.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m recall.50 /path/downloads/amazon_smaller_version/test_qrel_detail.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m map /path/downloads/amazon_smaller_version/test_qrel_detail.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m recip_rank /path/downloads/amazon_smaller_version/test_qrel_detail.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m ndcg_cut /path/downloads/amazon_smaller_version/test_qrel_detail_mul.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m ndcg_cut.50 /path/downloads/amazon_smaller_version/test_qrel_detail_mul.trec \
#         ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
# ./trec_eval -m ndcg_cut.10,50,100 /path/downloads/amazon_smaller_version/test_qrel_detail_mul_0001011.trec ${output_path}/${ckpt}/encode/test_all_rank_eval.trec
done