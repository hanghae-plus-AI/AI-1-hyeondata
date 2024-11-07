## 목표

---

이번 실습에서는 LoRA에서 rank를 변화시켰을 때, 성능 및 메모리 사용량 차이를 살펴볼 것입니다. 기존의 LoRA 실습 코드를 그대로 사용하되, 다음 부분들을 report 하시면 됩니다:

- [ ] `lora_r`를 `[8, 128, 256]`로 변화시켜가며 학습
    
    - Deepspeed 없이 순수 LoRA만을 가지고 기존과 같은 LLM(`facebook/opt-350m`)과 dataset(`lucasmccabe-lmi/CodeAlpaca-20k`)를 활용합니다.
        
    - Rank를 8, 128, 256로 바꿔가며 학습을 진행해봅니다.
        
    - SFTTrainer는 다음과 같이 변경합니다:
        
        ```python
        trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            args=SFTConfig(output_dir="/tmp/clm-instruction-tuning", **max_seq_length=128**),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
        )
        trainer.train()
        ```
        
- [ ] Rank에 따른 loss, 학습 속도, 그리고 메모리 점유율 공유
    
    - Loss는 wandb를 활용하여 다음과 같은 log를 공유합니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/83c75a39-3aba-4ba4-a792-7aefe4b07895/b3073297-aa25-459d-bee9-0ed6843b447b/image.png)
        
    - 학습 속도 또한 wandb의 `Runtime` 항목을 공유합니다.
        
    - 메모리 점유율은 다음 코드를 적절히 추가하여 print한 후, 공유합니다.
        
        ```python
        print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
        ```
        
- [ ] LoRA의 장단점 분석
    
    - 위에서 공유한 loss, 학습 속도, 메모리 점유율 각각을 보고 LoRA의 장단점을 분석하시면 됩니다.


---


![[스크린샷 2024-11-08 오전 1.51.03.png]] 
## Rank 8

### Run summary:

| final_cpu_memory_gb      | 1.9                |
| ------------------------ | ------------------ |
| final_gpu_memory_gb      | 8.8                |
| initial_cpu_memory_gb    | 1.6                |
| initial_gpu_memory_gb    | 1.2                |
| total_flos               | 6358023989231616.0 |
| train/epoch              | 0.79904            |
| train/global_step        | 1000               |
| train/grad_norm          | 4.55822            |
| train/learning_rate      | 0                  |
| train/loss               | 1.6452             |
| train_loss               | 1.6836             |
| train_runtime            | 436.4671           |
| train_samples_per_second | 36.658             |
| train_steps_per_second   | 2.291              |
| training_time_seconds    | 436.82808          |
메모리 점유율 초기 : init Max Alloc: 1.2 GB
메모리 점유율 마지막 : End Max Alloc: 8.8 GB
## Rank 128
### Run summary:

| final_cpu_memory_gb      | 3.1                |
| ------------------------ | ------------------ |
| final_gpu_memory_gb      | 9.1                |
| initial_cpu_memory_gb    | 3.1                |
| initial_gpu_memory_gb    | 8.8                |
| total_flos               | 6604624968155136.0 |
| train/epoch              | 0.79904            |
| train/global_step        | 1000               |
| train/grad_norm          | 1.13409            |
| train/learning_rate      | 0                  |
| train/loss               | 1.632              |
| train_loss               | 1.67582            |
| train_runtime            | 456.9278           |
| train_samples_per_second | 35.016             |
| train_steps_per_second   | 2.189              |
| training_time_seconds    | 457.27191          |
메모리 점유율 초기 : init Max Alloc: 8.8 GB
메모리 점유율 마지막 : End Max Alloc: 9.1 GB
## Rank 256
### Run summary:

| final_cpu_memory_gb      | 3.3                |
| ------------------------ | ------------------ |
| final_gpu_memory_gb      | 9.3                |
| initial_cpu_memory_gb    | 3.2                |
| initial_gpu_memory_gb    | 9.1                |
| total_flos               | 6867666012340224.0 |
| train/epoch              | 0.79904            |
| train/global_step        | 1000               |
| train/grad_norm          | 0.80191            |
| train/learning_rate      | 0                  |
| train/loss               | 1.6305             |
| train_loss               | 1.67473            |
| train_runtime            | 470.6815           |
| train_samples_per_second | 33.993             |
| train_steps_per_second   | 2.125              |
| training_time_seconds    | 471.04438          |
메모리 점유율 초기 : init Max Alloc: 9.1 GB
메모리 점유율 마지막 : End Max Alloc: 9.3 GB

train/loss 확인 링크 : https://wandb.ai/kimyonghyeondata-soongsil-university/lora-rank-comparison/reports/train-loss-24-11-08-01-57-04---VmlldzoxMDA3Mjc0Mw?accessToken=tn3pt7jisfq65o4intv5lhxb2gz2ae3ty7etaf9c84o3qjx3ib8vb21x1ym9lpj1


## LoRA의 장단점 분석
- loss는 미세하지만 128과256은 미세하게 차이나지만 256이 제일 좋았고, rank를 올리더라도 8과는 상대적으로 차이가 많이 났지만 128과 256의 차이는 미세해 지는 것 처럼 적절한 rank 정해야 할 것 같다
- 학습 속도는 Rank가 올라갈수록 시간이 더 걸리는 것으로 보입니다
- 메모리 점유율은 코드에 init 메모리와 end 메모리를 구하게 코드를 수정해서 추가 했는데 "torch.cuda.empty_cache()" 코드가 제대로 동작하지 않아서 gpu 메모리가 잘 정리가 안된건지 모르지만 눈으로 보이는 차이로는 8 rank는 급격히 올라가서 알 수 없지만 128과 256을 비교 했을 때 신기하게 256이 메모리를 덜 소모하는 것을 보이고 있으며, 실험에 실패한 듯 하며, 추후 원인을 찾을 예정입니다.
## 결론
LoRA는 지금 실험 결과로는 메모리가 확실히 잘 최적화 되어 있어서 rank를 변경해서 사용하는데 많은 부담이 없을 것 같고, wandb그래프를 보면 확실하게 loss가 떨어지는 것 같으며, 원하는 loss와 학습시간을 고려만 한다면 적절한 rank를 넣고 LoRA를 사용한다면 fine tunning을 부담이 덜 되지 않을까 합니다! 