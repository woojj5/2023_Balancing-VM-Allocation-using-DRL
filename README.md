# Balancing VM Allocation using DRL

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/dcbb579e-ac32-481f-92f6-8375b15ae1f8/image.png)

기존의 클라우드에서 데이터를 할당하는 작업을 진행하는 경우, ML이나 DR이 자주 사용되었지만 

자동적인 의사결정을 만들고 시스템을 최적화하기 위해서 RL을 적용한 방식을 제안하려고 한다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/0f5d3acb-5350-41e0-918c-32625d292a4a/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/c2c63ee9-ce3f-48ea-8af4-444e5e189819/image.png)

특히 virtual machine 할당의 경우는 클라우드 컴퓨팅의 할당 측면에서 가장 중요한 이슈이다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/078acb1a-22c4-41ab-baef-44f9f8803c80/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/c972f5a6-fecb-4f7e-a543-3e7e716b0962/image.png)

최적화를 위해서 사용된 RL 알고리즘들은 다음과 같습니다.

DQN방식의 경우, 각 상황에 따라 action을 하고 이를 누적으로 하여 얻는 최댓값을 찾는다.

Actor-Critic 방식의 경우, 각 유저들이 상황에 따라 행동을 실행하고 이를 최적의 값들을 얻는 critic network에게 피드백을 받아서 값을 얻는다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9560afd5-fa0d-49ca-bd64-e5db5ccacbc9/81641f54-4398-4126-90c1-6bd406d3e6e4/image.png)

시간에 따른 최적의 할당 방식을 찾고 이에 따라 CPU 할당되는 양을 알려준다.
