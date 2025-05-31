# Balancing-VM-Allocation-using-DRL

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

![image (12)](https://github.com/user-attachments/assets/29e5f633-cda4-42c9-a2e8-70dbf17c9a00)

![image (13)](https://github.com/user-attachments/assets/1391074d-c489-4667-b55d-dd61a1ea2822)

최적화를 위해서 사용된 RL 알고리즘들은 다음과 같습니다.

DQN방식의 경우, 각 상황에 따라 action을 하고 이를 누적으로 하여 얻는 최댓값을 찾는다.

Actor-Critic 방식의 경우, 각 유저들이 상황에 따라 행동을 실행하고 이를 최적의 값들을 얻는 critic network에게 피드백을 받아서 값을 얻는다.

![image (14)](https://github.com/user-attachments/assets/f3ab877b-9539-434f-b08f-0ca5d3b876c3)

시간에 따른 최적의 할당 방식을 찾고 이에 따라 CPU 할당되는 양을 알려준다.
