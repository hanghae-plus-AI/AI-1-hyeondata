import torch
import wandb
import logging
import json
import random
from typing import Optional
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
from transformers import (
   AutoConfig,
   AutoModelForCausalLM, 
   AutoTokenizer,
   HfArgumentParser,
   TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM

# 학습에 필요한 인자들을 정의하는 클래스
@dataclass
class Arguments:
   model_name_or_path: Optional[str] = field(default="openai-community/openai-gpt")  # 사용할 사전학습 모델 경로
   torch_dtype: Optional[str] = field(default=None)  # 모델의 데이터 타입(precision) 설정
   corpus_path: Optional[str] = field(default="corpus.json")  # 학습 데이터 경로
   block_size: int = field(default=1024)  # 입력 시퀀스 최대 길이
   train_val_split: float = field(default=0.8)  # 학습/검증 데이터 분할 비율

# 학습 관련 설정을 정의하는 클래스
@dataclass
class CustomTrainingArguments(TrainingArguments):
   output_dir: str = field(default="tmp/clm-instruction-tuning")  # 모델과 체크포인트가 저장될 경로
   per_device_train_batch_size: int = field(default=8)  # GPU당 배치 크기
   evaluation_strategy: str = field(default="steps")  # 평가 전략 ("steps": 일정 스텝마다 평가)
   eval_steps: int = field(default=100)  # 몇 스텝마다 평가할지 설정
   logging_steps: int = field(default=100)  # 몇 스텝마다 로그를 남길지 설정
   save_total_limit: int = field(default=1)  # 저장할 체크포인트 수 제한

# 인자 파싱
parser = HfArgumentParser((Arguments, CustomTrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# wandb 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-instruction-tuning'

# 데이터셋 로드 함수
def load_corpus(file_path: str, train_ratio: float = 0.8):
   """
   JSON 파일에서 데이터를 로드하고 학습/검증 데이터셋으로 분할
   """
   with open(file_path, 'r', encoding='utf-8') as f:
       corpus = json.load(f)
   random.shuffle(corpus)  # 데이터 섞기
   split_idx = int(len(corpus) * train_ratio)
   return DatasetDict({
       'train': Dataset.from_list(corpus[:split_idx]),  # 학습 데이터
       'validation': Dataset.from_list(corpus[split_idx:])  # 검증 데이터
   })

# 데이터셋 로드
raw_datasets = load_corpus(args.corpus_path, args.train_val_split)

# 모델과 토크나이저 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
   args.model_name_or_path,
   config=config,
   torch_dtype=args.torch_dtype
)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
   if tokenizer.eos_token is not None:
       tokenizer.pad_token = tokenizer.eos_token  # EOS 토큰을 패딩 토큰으로 사용
       tokenizer.pad_token_id = tokenizer.eos_token_id
   else:
       tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 새로운 패딩 토큰 추가
       model.resize_token_embeddings(len(tokenizer))  # 토크나이저 크기에 맞게 임베딩 조정

# 프롬프트 포맷팅 함수 정의
def formatting_prompts_func(example):
   """
   instruction과 output을 지정된 형식으로 포맷팅
   """
   return [f"### Question: {instruction}\n### Answer: {output}" 
           for instruction, output in zip(example['instruction'], example['output'])]

# 응답 템플릿과 데이터 콜레이터 설정
response_template = " ### Answer:"  # 응답 시작 부분을 나타내는 템플릿
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 트레이너 초기화
trainer = SFTTrainer(
   model=model,
   args=SFTConfig(
       num_train_epochs=50,
       output_dir=training_args.output_dir,
       per_device_train_batch_size=training_args.per_device_train_batch_size,
       logging_steps=training_args.logging_steps,
       evaluation_strategy=training_args.evaluation_strategy,
       eval_steps=training_args.eval_steps,
       save_total_limit=training_args.save_total_limit,
       max_seq_length=args.block_size,
       packing=False
   ),
   train_dataset=raw_datasets["train"],
   eval_dataset=raw_datasets["validation"],
   tokenizer=tokenizer,
   data_collator=collator,
   formatting_func=formatting_prompts_func
)

# 학습 및 평가 실행
# train_result = trainer.train()
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_args.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_args.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.  
    checkpoint = last_checkpoint
    
train_result = trainer.train(resume_from_checkpoint=checkpoint)

#eval데이터셋 사용
final_eval_results = trainer.evaluate()

# 모델과 메트릭 저장
trainer.save_model()  # 학습된 모델 저장
trainer.log_metrics("train", train_result.metrics)  # 학습 메트릭 저장
trainer.log_metrics("eval", final_eval_results)  # 평가 메트릭 저장

wandb.finish()  # wandb 종료