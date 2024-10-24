실행 방법 :  python train.py --model_name_or_path openai-community/openai-gpt --per_device_train_batch_size 8 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --output_dir tmp/test-clm --save_total_limit 1 --logging_steps 100 --evaluation_strategy "steps" --eval_steps 100

