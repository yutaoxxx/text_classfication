python3 run_classifier.py \
        --task_name=thucnews \
        --do_train=true \
        --do_eval=true \
        --do_test=false \
        --data_dir=/yutao/industry_class/BERT/data \
        --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
        --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
        --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt \
        --max_seq_length=200 \
        --train_batch_size=20 \
        --learning_rate=2e-5 \
        --num_train_epochs=20.0 \
        --output_dir=./output/thucnews/

