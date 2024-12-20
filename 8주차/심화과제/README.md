- [ ] LLM 서비스 경량화 진행
    - 다음 두 가지 방향 중 하나로 LLM 서비스를 경량화하시면 됩니다:
        1. **학습 속도 개선:** 만약 직접 학습을 통해 LLM 서비스를 개발하시는 분들은 학습 속도를 개선하는 방향으로 경량화를 진행하셔도 좋습니다. 이런저런 기법들을 적용했을 때, 성능은 보존하면서 속도가 개선되는 방향으로 학습 과정을 경량화하시면 됩니다.
        2. **큰 LLM 활용:** 같은 dataset으로 학습됐지만 더 큰 open LLM을 활용할 수 있도록 경량화하시면 됩니다. 예를 들어, `gemma-2b-it`이라는 모델로 LLM 서비스를 개발했지만 경량화를 진행하여 `gemma-9b-it`로 LLM 서비스를 구현하게끔 할 수 있습니다.
    - 이외에도 경량화를 했을 때 어떠한 방향으로든 확실한 장점을 가지는 경우면 상관없습니다.
    - 아무튼 어떤 경량화를 진행했는지 코드 등을 포함하여 잘 작성해주셔야 합니다.
- [ ] 경량화를 했을 때 얻을 수 있었던 이점 공유
    - 경량화를 했을 때 어떠한 이점을 가지는지 분석합니다. 다음과 같은 예시들이 있습니다:
        - 성능은 비슷한데 응답 속도가 빨라졌습니다.
        - GPU 메모리 때문에 큰 LLM을 사용하지 못하다가, LLM의 크기를 경량화를 통해 늘렸을 때 속도는 느리지만 더 정확한 응답을 하게 되었습니다.
    - 성능 같은 경우에는 정량적인 방법(e.g., benchmark 성능) 또는 정성적인 방법(e.g., 같은 instruction을 주었을 때 나오는 답변 분석)으로 분석해주시면 됩니다.

---

# 위의 과제 중에서 2.큰 LLM을 활용해서 테스트 해보고자 했습니다! 

gemma-2b-it를 이용한 결과와 Quantization을 이용해서 gemma-9b-it 모델을 경량화 해서 실행해 본 결과는 아래와 같습니다.
### 같은 question을 주었을 때 결과 입니다.

### 1. gemma-2b-it로 지문 인식률 높이는 방법을 물어보았을 경우
	1.  지문인식률 높이는 방법 : """ 지문 인식률을 높이려면 다음과 같은 방법을 사용하세요.
* 손가락을 구부리거나 손끝을 이용해 지문을 인식하면 지문 인식률이 떨어질 수 있습니다. 
자세한 내용은 지문 인식을 참고하세요. """

### 2. 제품에서 열이 나는 경우와 조치 방법: """ 제품에서 열이 나는 경우, 열을 식히기 위해 제품의 기능 및 성능을 제한하거나 전원이 꺼질 수 있습니다.
**참고:** 
* 제품에서 발열이 심하거나 오래 지속될 경우 제품 사용을 잠시 중단하세요. 
* 열이 나는 경우 이렇게 해 보세요. """

### 3. 개인 정보 및 데이터 보호 방법 : """ 제공된 문맥에서 개인 정보 및 데이터 보호 방법은 다음과 같습니다.
개인정보 처리방침 확인: 개인 정보 처리 방침을 확인할 수 있습니다.
데이터 삭제: 제품에 있는 데이터가 지워졌습니다. """

### 4. 내 기기 찾기 서비스 설정하기 : """ 내 기기 찾기 서비스 설정하기는 Google 계정을 사용하여 설정할 수 있습니다.
자세한 내용은 "내 디바이스 찾기 서비스 설정하기"를 참고하세요. """

### 5. 무선 충전하는 법 : """무선 충전기는 삼성전자의 정품 무선 충전기를 사용해야 합니다. 고속 무선 충전 기능을 이용하여 배터리를 더욱 빠르게 충전할 수 있습니다. 무선 충전을 지원하는 기기만 충전할 수 있습니다.""

## 2. gemma-9b-it로 지문 인식률 높이는 방법을 물어보았을 경우

### 1.  지문인식률 높이는 방법 :  """ 지문 인식률을 높이려면 주로 사용하는 손의 지문을 등록하고, 지문 인식 영역에 손가락이 넓게 닿도록 화면을 눌러주세요. 또한, 손가락을 구부리거나 손끝을 이용해 지문을 인식하지 않도록 주의해야 합니다. """

### 2. 제품에서 열이 나는 경우와 조치 방법: """ 제품에서 열이 나는 경우, 다음과 같은 조치를 취해주세요.
제품에서 발열이 심하거나 오래 지속될 경우 제품 사용을 잠시 중단하세요."""

### 3. 개인 정보 및 데이터 보호 방법 : """ 제품에 저장된 연락처, 사진 등의 개인 정보 및 데이터를 보호하기 위해 다음 사항을 꼭 지키세요 라고 명시되어 있습니다.
구체적인 방법은 제공되지 않았지만,  "설정 화면에서 개인정보 보호를 선택하세요" 와 "설정 화면에서 생체 인식 및 보안을 선택하세요" 라는 문구가 있으므로, 설정 화면에서 개인정보 보호 및 생체 인식 설정을 활용하는 것이 중요하다는 것을 알 수 있습니다. """

### 4. 내 기기 찾기 서비스 설정하기 : """ Google에서 제공하는 내 기기 찾기 서비스를 사용하면 제품의 위치를 확인하거나 원격으로 제어할 수 있습니다. 컴퓨터 또는 다른 모바일 기기에서 내 디바이스 찾기 사이트(findmymobile.samsung.com)를 이용할 수 있습니다. """

### 5. 무선 충전하는 방법 : """ 무선 충전하려면 무선 충전을 지원하는 기기를 무선 충전기에 올려놓으면 됩니다. 충전이 끝나면 무선 충전기를 분리하세요. 고속 무선 충전 기능을 이용하면 더욱 빠르게 배터리를 충전할 수 있습니다. 삼성전자의 정품 무선 충전기를 사용하는 것이 좋습니다. """

## 결론
### 4bit 양자화를 통해서 경량화한 모델을 사용했을 경우  
- 대답 속도가 생각보다 엄청 빠르다 
- 일반 2b모델 보다 메모리도 덜 소모한다
- 위에 처럼 2b 모델 보다 더 정확한 응답을 하는 것을 볼수 있다.

