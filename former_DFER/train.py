import torch
import torch.nn as nn
import torch.optim as optim

from former_DFER.models.ST_Former import GenerateModel
from video_dataset import VideoDataset

btz = 2
index = 4

# 加载训练集
train_dataset = VideoDataset(
        video_folders=["/media/mengting/Expansion/CMVS_projects/EmotionAI/SPFEED_dataset/SPFEED_dataset/verified_lable/pose_cropped",
                       "/media/mengting/Expansion/CMVS_projects/EmotionAI/SPFEED_dataset/SPFEED_dataset/verified_lable/spon_cropped"],
        image_size=256,
        sign='binary'
    )
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=btz,
                                          shuffle=True, num_workers=4)

# # 加载测试集
# testset = SundownSyndrome(data_root='/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2', phase='test', index=index)
# testloader = torch.utils.data.DataLoader(testset, batch_size=btz,
#                                          shuffle=False, num_workers=1)

device = 'cuda'

# model = light_resnet18().to(device)  # 使用定义的模型，并将模型移动到GPU
model = GenerateModel(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.005)  # adam优化器

epoches = 70

for epoch in range(epoches):  # 循环遍历数据集多次

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader):
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


torch.save(model.state_dict(), 'trained_former_DFERmodel_val' + str(index) + '.pth')

if __name__ == '__main__':

    print('Finished Training')