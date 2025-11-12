## EEG Foundation Challenge

본 레포는 **EEG Foundation Challenge** 세팅에서  
1️⃣ **Challenge 1:** 반응시간 회귀를 위한 DIR 기반 멀티-전문가 회귀  
2️⃣ **Challenge 2:** externalizing 예측을 위한 다중과제 학습  
두 가지 파이프라인을 제공합니다.

- 공식 페이지: [https://eeg2025.github.io/](https://eeg2025.github.io/)

---

### 🧭 Repository 구조

| 파일 | 설명 |
|------|------|
| `challenge_1.py` | DIR 기반 모델 학습 스크립트 (cross-task → cross-subject 회귀) |
| `challenge_2.py` | CascadedEEG 다중과제 프리트레인 스크립트 (factors → externalizing) |
| `dir.py` | DIR 유틸 및 손실 함수 (그룹 분할, 소프트 라벨, ordinal contrastive 등) |
| `model.py` | `CascadedEEGModel` 정의 (EEGNeX 인코더, 게이트 융합 옵션 포함) |

---
### 📄 Reference

[1] Aristimunha, Bruno, et al. "EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding." arXiv preprint arXiv:2506.19141 (2025). 

[2] Shirazi, Seyed Yahya, et al. "HBN-EEG: The FAIR implementation of the Healthy Brain Network (HBN) electroencephalography dataset." bioRxiv (2024): 2024-10. 

[3] Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018. 

[4] Liu, Yahao, et al. "Balanced Sharpness-Aware Minimization for Imbalanced Regression." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025. 

[5] Pu, Ruizhi, et al. "Leveraging group classification with descending soft labeling for deep imbalanced regression." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 19. 2025.

---
### ⚖️ 라이선스 및 고지

본 레포는 연구용 예제 코드입니다.

EEGNeX 및 Braindecode의 저작권은 각 저자에게 있습니다.

데이터셋 사용 시 원 저작권 및 사용 약관을 따라야 합니다.
