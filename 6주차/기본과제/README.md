이번 과제에서는 GPT fine-tuning을 할 때 validation data를 두어 validation loss도 같이 측정하는 코드를 구현하면 됩니다. 

자세한 요구사항들은 다음과 같습니다:

- [ ]  Validation data 준비
- [ ]  학습 시 validation loss 계산
    - Trainer를 정의할 때 validation data를 추가하고 validation data에 대한 evaluation을 진행하도록 수정합니다. 이전 주차들의 코드를 참고하시면 쉽게 구현할 수 있습니다.
    - 실제로 학습 후, `train/loss`와 `eval/loss` 에 해당하는 wandb log를 공유해주시면 됩니다.

사용한 데이터셋 wikitext 중에서 wikitext-2-raw-v1을 아래 처럼 사용했습니다.

실행 방법 :  python train.py --model_name_or_path openai-community/openai-gpt --per_device_train_batch_size 8 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --output_dir tmp/test-clm --save_total_limit 1 --logging_steps 100 --evaluation_strategy "steps" --eval_steps 100

eval/loss 링크 : https://api.wandb.ai/links/kimyonghyeondata-soongsil-university/9iwb55yk

train/loss 링크 : https://api.wandb.ai/links/kimyonghyeondata-soongsil-university/0ejv1sfz

