import torch
import torch.nn as nn # 신경망 포함
import torch.optim as optim # 최적화 알고리즘
import torch.nn.init as init # 텐서에 초기값 부여

import torchvision.datasets as datasets # 데이터 세트 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴

import numpy as np
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 100
learning_rate = 0.0002
num_epoch = 10

# MNIST 데이터 불러오기
mnist_train = datasets.MNIST(root="../MNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../MNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# 데이터로더 정의하기
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# CNN 모델 정의하기
class CNN(nn.Module):
    def __init__(self):
        # super함수는 CNN 클래스의 부모 클래스인 nn.Module을 초기화
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            # [100, 1, 28, 28] -> [100, 16, 24, 24]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),

            # [100, 16, 24, 24] -> [100, 32, 20, 20]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),

            # [100, 32, 20, 20] -> [100, 32, 10, 10]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [100, 32, 10, 10] -> [100, 64, 6, 6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),

            # [100, 64, 6, 6] -> [100, 64, 3, 3]
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            # [100, 64 * 3 * 3] -> [100, 100]
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            # [100, 100] -> [100, 10]
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # self.layer에 정의한 함수 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100, 나머지]로 변환
        out = out.view(batch_size, -1)
        # self.fc_layer에 정의한 연산 수행
        out = self.fc_layer(out)

        return out


# gpu가 사용가능하면 gpu로 보내기
device = torch.device("mps:0" if torch.backends.mps.is_available() else 'cpu')
model = CNN().to(device)

# 손실함수와 최적화함수 정의하기
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 모델 만들기
loss_arr = []

for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad()

        output = model.forward(x)

        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())

# 테스트 모델 만들기
correct = 0
total = 0

# evaluate model
model.eval()

with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)

        # torch.max 함수는 (최댓값, index)를 반환
        _, output_index = torch.max(output, 1)

        # 전체 개수 += 라벨의 개수
        total += label.size(0)

        # 도출한 모델의 index와 라벨이 일치한다면 correct에 개수 추가
        correct += (output_index == y).sum().float()
    print(f"Accuracy of Test Data: {100 * correct / total}")
