# Balancing VM Allocation using DRL

![image (9)](https://github.com/user-attachments/assets/64ace5b3-38a2-48b6-8e1f-d3ef9a48b6fb)

기존의 클라우드에서 데이터를 할당하는 작업을 진행하는 경우, ML이나 DR이 자주 사용되었지만 

자동적인 의사결정을 만들고 시스템을 최적화하기 위해서 RL을 적용한 방식을 제안하려고 한다.

![image (10)](https://github.com/user-attachments/assets/d167b11b-7805-44f3-8689-cc9cc2a6cce7)

![image (11)](https://github.com/user-attachments/assets/8ceb091a-1d2e-4d56-b419-544075b486e8)

특히 virtual machine 할당의 경우는 클라우드 컴퓨팅의 할당 측면에서 가장 중요한 이슈이다.

![image (12)](https://github.com/user-attachments/assets/29e5f633-cda4-42c9-a2e8-70dbf17c9a00)

![image (13)](https://github.com/user-attachments/assets/1391074d-c489-4667-b55d-dd61a1ea2822)

최적화를 위해서 사용된 RL 알고리즘들은 다음과 같습니다.

DQN방식의 경우, 각 상황에 따라 action을 하고 이를 누적으로 하여 얻는 최댓값을 찾는다.

Actor-Critic 방식의 경우, 각 유저들이 상황에 따라 행동을 실행하고 이를 최적의 값들을 얻는 critic network에게 피드백을 받아서 값을 얻는다.

![image (14)](https://github.com/user-attachments/assets/f3ab877b-9539-434f-b08f-0ca5d3b876c3)

시간에 따른 최적의 할당 방식을 찾고 이에 따라 CPU 할당되는 양을 알려준다.
