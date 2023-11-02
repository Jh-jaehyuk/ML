"""
데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종처리가 된
형태로 제공되지 않음!!
변형(transform)을 해서 데이터를 조작하고 학습에 적합하도록 하자
torchvision의 데이터셋은 변형 로직은 갖는 호출 가능한 객체를
받는 매개변수 2개(특징을 변경하기 위한 transform과 정답을 변경하기 위한 target_transform)를 갖는다

FashionMNIST의 특징은 PIL Image 형식, 정답은 정수
학습을 하려면 정규화된 텐서 형태의 특징과
원-핫(one-hot)으로 부호화(encode)된 텐서 형태의 정답 필요
이러한 변형을 하기 위해 ToTensor와 Lambda 사용
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="../Tutorials-Pytorch",
    train=True,
    download=True,
    # ToTensor()는 PIL Image나 NumPy ndarray를 FloatTensor로 변환하고
    # 이미지 픽셀의 크기 값을 [0., 1.] 범위로 비례하여 조정함
    transform=ToTensor(),
    # Lambda 변형은 사용자 정의 람다 함수를 적용
    # 여기에서는 정수를 원-핫으로 부호화된 텐서로 바꾸는 함수를 정의함
    # 먼저 데이터셋 정답 개수 크기의 zero tensor를 만들고
    # scatter_를 호출하여 주어진 정답에 해당하는 인덱스에 value=1을 할당
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

print(ds)