import torch

# -------------------------------
# 参数设置
# -------------------------------
input_channels  = 1
output_channels = 2
kernel_size     = (2, 2)
stride          = 1
padding         = 0

# -------------------------------
# 正确构造输入：Shape [1, 1, 3, 3] -> (batch, channel, height, width)
# -------------------------------
input_data = torch.tensor([
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
])  # shape: [1, 1, 3, 3]

# ↑ 解释:
# 第一层 [] 是 batch (N=1)
# 第二层 [] 是 channel (C=1)
# 后面是 3x3 图像

print("Input Image (3x3):")
print(input_data[0, 0].numpy())  # 只显示图像部分
print(f"Input shape: {input_data.shape}")  # 应该是 [1, 1, 3, 3]

# -------------------------------
# 定义卷积核权重
# -------------------------------
weight_data = torch.tensor([
    [[ [1.0, -1.0],
       [0.0,  2.0] ]],  # output channel 0
    [[ [1.0,  0.0],
       [1.0,  0.0] ]]   # output channel 1
])  # shape: [2, 1, 2, 2]

# 创建卷积层
conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=(2, 2),
    stride=1,
    padding=0,
    bias=False
)

# 手动加载权重
with torch.no_grad():
    conv.weight.copy_(weight_data)

print("\nFilter Weights:")
for i in range(output_channels):
    print(f"Filter {i}:")
    print(weight_data[i, 0].numpy())

# -------------------------------
# 前向传播（现在能正确运行）
# -------------------------------
output = conv(input_data)
output_np = output.squeeze(0).detach().numpy()  # 去掉 batch 维度 -> [2, 2, 2]

print("\nOutput Feature Maps:")
for i in range(output_channels):
    print(f"Channel {i} (2x2):")
    print(output_np[i])
