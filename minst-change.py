import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x



def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])             # to_tensor = Compose(ToTensor())
    #data_set = MNIST("", train=is_train, transform=to_tensor, download=True)
    data_set = MNIST(root="D:/machine learning/minst-learning", train=is_train, transform=to_tensor,download=False)   #注：这里后续MNIST等路径会自己添加，写到上一级即可
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total



train_data = get_data_loader(is_train=True)
test_data = get_data_loader(is_train=False)
net = Net()
print("initial accuracy:", evaluate(test_data, net))
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


for epoch in range(3):
    for (x, y) in train_data:
        net.zero_grad()  #清空梯度
        # x.shape = torch.Size([15, 1, 28, 28])

        output = net.forward(x.view(-1, 28 * 28)) #得到输出，重塑为 batch_size 个铺开的28*28的向量，

        loss = torch.nn.functional.nll_loss(output, y) #计算损失：使用 NLL 损失函数来评估模型的预测与真实标签之间的差距。

        loss.backward()  #反向传播：计算损失相对于模型参数的梯度，以更新模型。

        optimizer.step()     #更新参数：通过优化器根据计算得到的梯度来更新模型的参数，从而使模型逐步提高性能。

    print("epoch", epoch, "accuracy:", evaluate(test_data, net))

model_path = r"D:\machine learning\minst-learning\model\minstM-1.pth"

torch.save(net.state_dict(), model_path)