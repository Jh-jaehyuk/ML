"""
신경망을 학습할 때 가장 자주 사용되는 알고리즘은 역전파!!
이 알고리즘에서 매개변수는 주어진 매개변수에 대한
손실 함수의 변화도(gradient)에 따라 조정된다.
torch.autograd는 이러한 변화도를 계산하기 위한
자동 미분 엔진 -> 모든 계산 그래프에 대한 변화도의 자동 계산
"""

import torch

x = torch.ones(5)
y = torch.zeros(3)
# requires_grad는 나중에 w.requires_gred_(True)로 설정도 가능하다
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


# Tensor, Function과 연산그래프(Computational graph)
# w, b는 최적화 해야 하는 매개변수
print(f'Gradient function for z = {z.grad_fn}')
print(f'Gradient function for loss = {loss.grad_fn}')


# 변화도(Gradient) 계산하기
loss.backward()
print(w.grad)
print(b.grad)

# 변화도 추척 멈추기
"""
변화도 추척을 멈춰야 하는 이유는
1. 신경망의 일부 매개변수를 고정된 매개변수로 표시함
2. 변화도를 추적하지 않는 텐서의 연산 효율이 더 좋다
-> 순전파 단계만 수행할 때 연산 속도가 향상됨!
"""
# 첫번째 방법
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w) + b
    print(z.requires_grad)
# 두번째 방법
z_det = z.detach()
print(z_det.requires_grad)

