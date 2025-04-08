import torch
import torchvision
import torchvision.transforms as transforms
from ablative_experiments.vgg.dataset import SundownSyndrome
import torch.nn as nn
from ablative_experiments.vgg.model import VGG
import torch.optim as optim
import torchvision.models as models




# 数据预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小到48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

btz = 2
index = 4

# 加载训练集
trainset = SundownSyndrome(data_root='/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2', phase='train', index=index)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=btz,
                                          shuffle=True, num_workers=1)

# 加载测试集
testset = SundownSyndrome(data_root='/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2', phase='test', index=index)
testloader = torch.utils.data.DataLoader(testset, batch_size=btz,
                                         shuffle=False, num_workers=1)

device = 'cuda'

# model = light_resnet18().to(device)  # 使用定义的模型，并将模型移动到GPU
# 加载预训练的VGG16模型
model = VGG(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.005)  # adam优化器

epoches = 70

for epoch in range(epoches):  # 循环遍历数据集多次

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data[0].to(device), data[1].to(device)
        # 参数梯度清零
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # 每一个epoch打印一次
    print(f'Epoch {epoch+1}, Accuracy on training set: {100 * correct / total}%, mean loss is: {running_loss / i}')

torch.save(model.state_dict(), 'trained_vggmodel_val' + str(index) + '.pth')
print('Finished Training')