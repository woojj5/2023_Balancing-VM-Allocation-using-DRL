# Balancing-VM-Allocation-using-DRL

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SQLite3](https://img.shields.io/badge/sqlite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

# RL_VM_Allocation

강화학습(Deep Reinforcement Learning, DRL) 기반의 VM(가상머신) 할당 및 밸런싱 실험 프로젝트입니다.
데이터센터 환경에서 VM을 효율적으로 할당하여 자원 활용률을 높이고, 서버 부하를 균형 있게 분산시키는 것을 목표로 합니다.

---

## 프로젝트 소개

![image (11)](https://github.com/user-attachments/assets/8ceb091a-1d2e-4d56-b419-544075b486e8)

- 다양한 DRL 알고리즘(Q-learning, DQN, Actor-Critic 등)을 활용해 VM 할당 정책을 학습합니다.
- VM의 자원 요구량, 서버 상태 등 환경 정보를 바탕으로 에이전트가 최적의 할당 결정을 내립니다.
- 실험 결과는 자원 활용률, 부하 밸런싱, 에너지 효율성 등의 지표로 분석합니다.

--

## 개발 환경

- **Python 3.7+**
- 주요 라이브러리: `numpy`, `pandas`, `matplotlib`, `torch` 또는 `tensorflow` (알고리즘별 상이)
- 버전 및 이슈관리: GitHub
- 협업 툴: GitHub Issues, GitHub Projects, Notion

---

## 폴더 및 파일 구조

```
RL_VM_Allocation/
├── Allocation_AC.py           # Actor-Critic 기반 에이전트 및 학습
├── Allocation_Q.py            # Q-learning/DQN 기반 에이전트 및 학습
├── VM_allocation.py           # VM 할당 환경 및 에이전트-환경 상호작용(메인 루프)
├── VmResult.py                # 실험 결과 기록 및 시각화
├── Vmallocation_Q_GPU.py      # GPU 환경에서 Q-learning 실험용 스크립트
└── README.md                  # 프로젝트 설명 및 사용법
```


---

## 주요 파일 설명

- **Allocation_AC.py**
Actor-Critic 방식의 DRL 에이전트 클래스 및 학습 루프 구현
- **Allocation_Q.py**
Q-learning 또는 DQN 방식의 DRL 에이전트 클래스 및 학습 루프 구현
- **VM_allocation.py**
VM 할당 환경(Environment) 정의 및 에이전트와의 상호작용 관리
- **VmResult.py**
실험 결과(자원 사용률, 서버 부하, 누적 보상 등) 저장 및 시각화
- **Vmallocation_Q_GPU.py**
GPU 환경에서 Q-learning 기반 실험을 실행하는 스크립트

---

## 역할 분담

| 이름 | 주요 역할 |
| :-- | :-- |
| 안현준 | Actor-Critic 알고리즘 구현 |
| 최윤수 | 논문 작성, 발표 준비|
| 전진우 | 프로젝트 기획, 설계|


---

## 작업 관리

- GitHub Projects와 Issues를 사용하여 진행 상황을 공유합니다.
- 주간 회의를 통해 작업 순서와 방향성을 논의하고, Notion에 회의 내용을 기록합니다.

---

## 사용 예시

```bash
# Actor-Critic 에이전트로 VM 할당 학습
python Allocation_AC.py

# Q-learning 에이전트로 VM 할당 학습
python Allocation_Q.py
```


---

## 신경 쓴 부분

- 기존 클라우드 VM 할당 문제는 ML/딥러닝 기반 접근이 많았으나,
**자동화된 의사결정과 시스템 최적화**를 위해 RL(강화학습) 기반 방식을 제안합니다.
- VM 할당은 클라우드 컴퓨팅에서 가장 중요한 이슈 중 하나입니다.
- 본 프로젝트에서는 다음과 같은 RL 알고리즘을 적용하였습니다:
    - **DQN 방식**: 각 상태에서 액션을 선택하고, 누적 보상이 최대가 되도록 학습합니다.
      
    - ![image (12)](https://github.com/user-attachments/assets/29e5f633-cda4-42c9-a2e8-70dbf17c9a00)
      
    - **Actor-Critic 방식**: 각 유저가 상황에 따라 행동을 실행하면 Critic 네트워크가 피드백을 제공하여 정책을 업데이트합니다. 이 과정을 통해 시간에 따른 최적의 VM 할당 방식을 찾고, CPU 등 자원 할당량을 효율적으로 조정합니다.
 
    - ![image (13)](https://github.com/user-attachments/assets/1391074d-c489-4667-b55d-dd61a1ea2822)

- 결과: 시간에 따른 최적의 할당 방식을 찾고 이에 따라 CPU 할당되는 양을 알려준다.
- ![VM 할당 구조 예시](image22.gif)
