import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import wandb
import psutil
import os
from datetime import datetime

def get_memory_usage():
    """현재 GPU/CPU 메모리 사용량을 반환합니다."""
    gpu_memory = round(torch.cuda.max_memory_allocated(0)/1024**3, 1)
    cpu_memory = round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 1)
    return gpu_memory, cpu_memory

def formatting_prompts_func(examples):
    """데이터셋 포맷팅 함수"""
    texts = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output}"
        texts.append(text)
    return texts

def train_with_lora(lora_r, experiment_name):
    """주어진 rank로 LoRA 학습을 수행합니다."""
    
    # wandb 초기화
    run = wandb.init(
        project="lora-rank-comparison",
        name=f"rank_{lora_r}_{experiment_name}",
        reinit=True
    )
    
    # 모델과 토크나이저 로드
    model_name = "facebook/opt-350m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # 데이터셋 로드
    dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=f"./results/rank_{lora_r}",
        max_steps=1000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
    )
    
    # 트레이너 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    # 초기 메모리 사용량 기록
    gpu_memory, cpu_memory = get_memory_usage()
    print('init Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
    wandb.log({
        "initial_gpu_memory_gb": gpu_memory,
        "initial_cpu_memory_gb": cpu_memory
    })
    
    # 학습 시작
    start_time = datetime.now()
    trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    
    # 최종 메모리 사용량 기록
    gpu_memory, cpu_memory = get_memory_usage()
    print('End Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
    wandb.log({
        "final_gpu_memory_gb": gpu_memory,
        "final_cpu_memory_gb": cpu_memory,
        "training_time_seconds": training_time
    })
    
    wandb.finish()

def main():
    """세 가지 다른 rank로 실험을 실행합니다."""
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for rank in [8, 128, 256]:
        print(f"\nStarting experiment with rank {rank}")
        train_with_lora(rank, experiment_name)
        torch.cuda.empty_cache()  # GPU 메모리 정리

if __name__ == "__main__":
    main()