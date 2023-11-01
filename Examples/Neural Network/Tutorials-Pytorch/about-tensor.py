import torch
import numpy as np

"""
텐서 초기화하기
"""

# 1. 데이터로부터 직접 생성하기
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. NumPy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. 다른 텐서로부터 생성하기
x_ones = torch.ones_like(x_data)
# print(f'Ones Tensor: \n {x_ones} \n')
x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f'Random Tensor: \n {x_rand} \n')

# 4. 무작위 또는 상수 값을 사용하기
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f'Random Tensor: \n {rand_tensor} \n')
# print(f'Ones Tensor: \n {ones_tensor} \n')
# print(f'Zeros Tensor: \n {zeros_tensor} \n')

"""
텐서의 속성(Attribute)
텐서의 모양, 자료형 및 어느 장치에 저장되는지를 나타냄
"""

tensor = torch.rand(3, 4)
# print(f'Shape of tensor: {tensor.shape}')
# print(f'Datatype of tensor: {tensor.dtype}')
# print(f'Device tensor is stored on: {tensor.device}')

"""
텐서 연산(Operation)
전치, 인덱싱, 슬라이싱, 수학 계산, 선형 대수, 임의 샘플링 등
각 연산들은 GPU에서 실행 가능
"""

# GPU가 존재하면 텐서를 이동
if torch.backends.mps.is_available():
    tensor = tensor.to('mps')
    # print(f'Device tensor is stored on: {tensor.device}')

# NumPy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
# print(tensor)

# 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# 텐서 곱하기
# 요소별 곱(element-wise product)
# print(f'tensor.mul(tensor) \n {tensor.mul(tensor)} \n')
# print(f'tensor * tensor \n {tensor * tensor} \n')
# 두 텐서 간의 행렬 곱(matrix multiplication)
# print(f'tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n')
# print(f'tensor @ tensor.T \n "{tensor @ tensor.T} \n')

# 바꿔치기(in-place) 연산
## 사용을 권장하지 않음!
# print(tensor, "\n")
tensor.add_(5)
# print(tensor)

"""
NumPy 변환(Bridge)
"""

# 텐서 -> 배열
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

## 텐서의 변경사항이 배열에도 반영됨
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

# 배열 -> 텐서
n = np.ones(5)
t = torch.from_numpy(n)

## 배열의 변경사항이 텐서에도 반영됨
np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')