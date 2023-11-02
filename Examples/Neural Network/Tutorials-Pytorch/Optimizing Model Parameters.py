import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root='../Tutorials-Pytorch',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='../Tutorials-Pytorch',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# 하이퍼파라미터는 모델 최적화 과정을 제어할 수 있는
# 조절 가능한 매개변수
# 서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율에 영향
# 학습률 - 각 배치/에폭에서 모델의 매개변수를 조절하는 비율
# 값이 작을수록 학습 속도가 느려지고
# 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있음
learning_rate = 1e-3
# 배치 크기 - 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
batch_size = 64
# 에폭 수 - 데이터셋을 반복하는 횟수
epochs = 10

# 최적화 단계
"""
하나의 에폭은 다음 두 부분으로 구성
1. 학습 단계 : 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴
2. 검증/테스트 단계 : 모델의 성능이 개선되고 있는지 확인하기 위해 테스트 데이터셋을 반복함
"""

# 학습 단계에서 일어나는 개념
# 손실 함수는 획득한 결과와 실제 값 사이의 틀린 정도를 측정
# 학습 중 이 값을 최소화하려고 함
loss_fn = nn.CrossEntropyLoss()
# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델의 매개변수를 조정하는 과정
# 최적화 알고리즘은 이 과정이 수행되는 방식을 정의함
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측과 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item() ,(batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f'Epoch {t + 1}\n----------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")
