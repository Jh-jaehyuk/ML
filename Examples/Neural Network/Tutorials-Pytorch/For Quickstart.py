"""
빠른 시작!
기계 학습의 일반적인 작업들을 위해
"""

# 필요한 라이브러리 불러오기
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 데이터 작업하기
training_data = datasets.FashionMNIST(
    root='MNIST',
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root='MNIST',
    train=False,
    download=True,
    transform=ToTensor(),
)

print(training_data)
print(test_data)