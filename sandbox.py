import torch.nn as nn
import torch

data = torch.tensor([   # (B=학생, C=과목, L=모의고사 차수)
    [[0.2, 0.12, 8],
    [0.1, 0.72, 0.3],
    [0.3, 0.52, 0.1],
    ],
    [[0.22, 0.11, 0.58],
    [0.21, 0.02, 0.73],
    [0.11, 0.22, 0.2],
    ],
])


layernorm_layer = nn.LayerNorm(3)
# 1. 배치 정규화 선언 (채널 수 2개 입력)
# 아직 코드 실행해서 텐서가 할당되기 전이라 수동으로 특징채널수 입력해줘야함
batchnorm_layer = nn.BatchNorm1d(2)

# 2. 실행
output = layernorm_layer(data)

print("=== Batch Norm 결과 ===")
print(output)

print("=== 검증결과 ===")
# x = torch.tensor([0.2, 0.12, 0.8])
mean = torch.mean(data, dim=-1, keepdim=True)
print(mean)
std = torch.std(data, dim=-1, unbiased=False, keepdim=True)
print(std)

