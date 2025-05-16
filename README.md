# viseme-lipsync

한국어 음성을 기반으로 자연스러운 립싱크 애니메이션을 생성하는 규칙 기반 Viseme 모델입니다. Phoneme에서 Viseme으로의 매핑과 Amplitude 기반 ADSR 곡선을 통해 실제 입 모양에 가까운 표현을 목표로 합니다.

## 🧠 개요

- PPG(Phoneme Posteriorgram)를 활용하여 음성에서 Phoneme 시퀀스 추출
- 한국어 발음 구조를 반영한 규칙 기반 Viseme 매핑 수행
- ADSR(Attack, Decay, Sustain, Release) 곡선과 Amplitude 정보를 이용하여 시간에 따라 부드러운 애니메이션 생성
- Blendshape weight를 직접 출력하여 별도의 후처리 없이 사용 가능

## 🧩 모델 구조
![{206432B4-C673-411A-A05E-09B9D1848F98}](https://github.com/user-attachments/assets/ff3a403a-b1d6-4787-abad-c6c5e19b85f7)

1. **Input Stage**
   - 음성 입력 → PPG → Phoneme 시퀀스

2. **Animation Stage**
   - 규칙 기반 Viseme 매핑:
     - 무음(SIL): Neutral
     - 단모음/이중모음 → Viseme으로 매핑
     - 자음 → 인접한 모음의 입 모양에 종속되며 일부는 예외 처리
   - 특수 처리:
     - 양순음(PBM1/PBM2), 치경음, 치찰음, 입술이 강조된 viseme은 타이밍 보정

3. **Output Stage**
   - Amplitude 값을 비선형 함수로 조정하여 viseme weight에 곱함
   - ADSR 곡선 적용으로 자연스러운 입 모양 전이 구현

## 🎯 결과

- 최대 5000 프레임의 오디오 처리 가능
- 60초 오디오 기준 약 5.5초 내에 처리 완료
- PPG 정확도에 따라 애니메이션 품질 영향 있음

## 📌 참고 문헌

- JALI: An Animator-Centric Viseme Model for Expressive Lip Synchronization  
- http://journal.cg-korea.org/archive/view_article?pid=jkcgs-26-3-49  
- https://www.dgp.toronto.edu/~elf/JALISIG16.pdf
