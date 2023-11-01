"""
Torch.autogrid
신경망 학습을 지원하는 PyTorch의 자동 미분 엔진!

신경망은 입력 데이터에 대해 실행되는 중첩된 함수들의 모음.
이 함수들은 텐서로 저장되는(가중치와 편향으로 구성된) 매개변수들로 정의됨

신경망 학습의 두 단계
1. 순전파(Forward Propagation)
신경망은 정답을 맞추기 위해 최선의 추측을 함
-> 추측을 하기 위해서 입력 데이터를 각 함수들에서 실행!

2. 역전파(Backward Propagation)
신경망은 추측한 값에서 발생한 오류에 비례하여 매개변수들을 적절히 조절
출력으로부터 역방향으로 이동하면서 오류에 대한 함수들의 매개변수들의 미분값(변화값 gradient) 수집
-> 경사하강법(gradient descent)을 사용하여 매개변수 최적화!
"""

# 사용법
## 이 튜토리얼은 텐서를 이동하더라도 GPU에서 작동하지 않고 CPU에서 작동
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
# 입력 데이터를 모델의 각 층에 통과시켜 예측값을 생성 -> 순전파
prediction = model(data)

# 모델의 예측값과 그에 해당하는 정답을 사용하여 오차를 계산
# 다음 단계는 신경망을 통해 이 에러를 역전파하는 것
# 오차 텐서에 .backward()를 호출하면 역전파 시작!
# 그 다음 Autogrid가 매개변수의 .grad 속성에 모델의 각 매개변수에 대한 변화도를 계산하고 저장
loss = (prediction - labels).sum()
loss.backward()

# 옵티마이저 부르기
# 0.1의 학습률과 0.9의 모멘텀을 갖는 SGD
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# .step()을 호출하여 경사하강법을 시작
# 옵티마이저는 .grad에 저장된 변화도에 따라 각 매개변수를 조정함
optim.step()

# Autogrid에서 미분
# requires_grad=True 는 모든 연산들을 추적해야 된다고 알려줌
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# 새로운 텐서를 만듦!
Q = 3 * a ** 3 - b ** 2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)

# 연산 그래프(Computational Graph)
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

# 입력 텐서 중 하나라도 requires_grad = True 를 갖는 경우 연산 결과도 변화도를 갖게 됨!
a = x + y
print(f"Does 'a' require gradients? : {a.requires_grad}")
b = x + z
print(f"Does 'b' require gradients? : {b.requires_grad}")

from torch import nn, optim

model = resnet18(weight=ResNet18_Weights.DEFAULT)

# 신경망의 모든 매개변수를 고정함
for param in model.parameters():
    param.requires_grad = False

# 10개의 정답을 갖는 새로운 데이터세트로 모델을 미세조정하는 상황을 가정
# resnet의 classifier는 linear layer인 model.fc
model.fc = nn.Linear(512, 10)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
