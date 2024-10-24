이번 과제에서는 LLM instruction-tuning을 HuggingFace hub의 data를 활용하는 것이 아닌 자체 제작한 data를 활용하여 학습하는 것이 목표입니다.

자세한 요구사항들은 다음과 같습니다:

- [ ]  Instruction-data 준비
    - 먼저 text corpus를 `corpus.json`의 이름으로 준비합니다.
    - Corpus의 형식은 제한이 없고, 100개 이상의 sample들로 구성되어 있으면 됩니다.
- [ ]  Train 및 validation data 준비
    - 먼저 `corpus.json`를 불러옵니다.
    - 그 다음 8:2 비율로 나눠, train과 validation data를 나눕니다.
    - 그 다음 기존의 data 전처리 코드를 적절히 수정하여 불러온 train과 validation data를 전처리합니다.
- [ ]  학습 결과 공유
    - 직접 만든 data로 GPT를 fine-tune한 후 저장된 `train/loss`과 `valid/loss`에 대한 wandb log를 공유해주시면 됩니다.


챗봇 컨셉은 아래와 같습니다

- 산업군 : AI (데이터 분석과 전 처리)
- 사용자 : 머신러닝 개발자 및 데이터 분석가
- 문제 상황 : 데이터를 전처리 또는 데이터를 분석하는 초보자 또는 인사이트를 찾아야 하는 사람이지만 인사이트를 찾기 위해서 어떻게 해야할지 막막한 사람이 어떤 방법들과 어떤 기법들이 있는지 초반에 물어보고 같이 분석해 나간다
- 목적 : 점점 나오는 통계 기법들과 논문들을 계속 학습 시켜서 새로 나오는 기법들과 기술들도 질문을 통해서 분석과 전 처리에 특화된 LLM을 만들고자 합니다.



실행 방법 :  python train.py (이전에 기본과제와 다르게 매개변수들을 default에 다 넣어 주었습니다.)

110개의 데이터를 임의로 만들어서 학습을 시켰고 8:2로 88개의 train데이터와 22개의 validation데이터 셋으로 학습을 했으며 아래와 같은 결과가 나왔습니다.

eval/loss 링크 : https://api.wandb.ai/links/kimyonghyeondata-soongsil-university/h7rjgmzy

train/loss 링크 : https://api.wandb.ai/links/kimyonghyeondata-soongsil-university/omzh71sm

