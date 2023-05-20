#!/bin/bash
set -e

# preprocess -------------------------------
python tasks/chatbot/preprocess.py --model_name_or_path=facebook/bart-base

# =============================== train ====================
python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=218 --eval_batch_size=24 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop --experiment_name=chatbot_onehop-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1

python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=218 --eval_batch_size=24 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_no_agg --experiment_name=chatbot_onehop_no_agg-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1

python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=218 --eval_batch_size=24 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_no_child --experiment_name=chatbot_onehop_no_child-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1

python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=218 --eval_batch_size=24 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_eh --experiment_name=chatbot_onehop_eh-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1

python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=256 --eval_batch_size=24 --model_name_or_path=facebook/bart-base \
 --output_dir=output/chatbot --model_name mybart --experiment_name=mybart-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1

# =============================== test ====================
python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop --experiment_name=chatbot_onehop-cadge

python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_no_agg --experiment_name=chatbot_onehop_no_agg-cadge

python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_no_child --experiment_name=chatbot_onehop_no_child-cadge

python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop_eh --experiment_name=chatbot_onehop_eh-cadge

python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=facebook/bart-base \
 --output_dir=output/chatbot --model_name mybart --experiment_name=mybart-cadge