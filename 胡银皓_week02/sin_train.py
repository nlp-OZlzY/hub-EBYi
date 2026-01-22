import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X_numpy = np.random.uniform(-2*np.pi, 2*np.pi, (500, 1))
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(500, 1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1), nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2), nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3), nn.ReLU(),
            nn.Linear(hidden_size3, output_size)
        )
    def forward(self, x):
        return self.network(x)

# 创建模型
model = MLP(input_size=1, hidden_size1=64, hidden_size2=32, hidden_size3=16, output_size=1)

# 训练
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1500):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 300 == 0:
        print(f'Epoch [{epoch+1}/1500], Loss: {loss.item():.6f}')

# 绘制结果
x_test = np.linspace(-2*np.pi, 2*np.pi, 300).reshape(-1, 1)
x_test_tensor = torch.from_numpy(x_test).float()

with torch.no_grad():
    model.eval()
    y_pred_test = model(x_test_tensor)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, alpha=0.5, s=20, label='原始数据点', color='blue')
plt.plot(x_test, y_pred_test.numpy(), 'r-', linewidth=2, label='MLP拟合曲线')
plt.plot(x_test, np.sin(x_test), 'g--', linewidth=2, label='真实 sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('使用MLP类拟合sin函数')
plt.legend()
plt.grid(True)
plt.show()