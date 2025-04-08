import torch
import torchvision
import torchvision.transforms as transforms
from dataset import SundownSyndrome, SundownSyndromeAllAnnotation
import torch.nn as nn
from model import light_resnet18
import torch.optim as optim


# 如果非要实现的话，那就一个epoch一测吧

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小到48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

btz = 2
index = 0


def add_to_list(pre, lab, pred_total, lab_total):
    pp = pre.detach().cpu().numpy().tolist()
    pred_total = pred_total + pp
    ll = lab.detach().cpu().numpy().tolist()
    lab_total = lab_total + ll
    return pred_total, lab_total




# 加载训练集
trainset = SundownSyndrome(data_root='/media/mengting/Expansion/CMVS_projects/SundownSyndrome/SS_dataset/SS_video_clips_with_labels2',
                                        phase='train', index=index)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=btz,
                                          shuffle=True, num_workers=1)

# 加载测试集
testset = SundownSyndrome(data_root='/media/mengting/Expansion/CMVS_projects/SundownSyndrome/SS_dataset/SS_video_clips_with_labels2',
                                       phase='test', index=index)
testloader = torch.utils.data.DataLoader(testset, batch_size=btz, drop_last=False,
                                         shuffle=False, num_workers=1)

device = 'cuda'

model = light_resnet18().to(device)  # 使用定义的模型，并将模型移动到GPU

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.005)  # adam优化器

epoches = 150

def val(cur_model):
    pred = []
    label = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(testloader, 0):
            # 获取输入数据
            inputs, labels = data[0].to(device), data[1].to(device)

            # 前向 + 反向 + 优化
            outputs = cur_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred, label = add_to_list(predicted, labels, pred, label)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct, total, pred, label


best_acc = 0
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
    print(
        f'Epoch {epoch + 1}, Accuracy on training set: {100 * correct / total}%, mean loss is: {running_loss / i}')
    epoch_correct, epoch_total, epoch_pred, epoch_label = val(model)
    print('test accuracy is ', (epoch_correct / epoch_total))
    if (epoch_correct / epoch_total) > best_acc:
        best_acc = epoch_correct / epoch_total
        torch.save(model.state_dict(), 'trained_model_val_all_annotation_' + str(index) + '.pth')



if __name__ == '__main__':
    # torch.save(model.state_dict(), 'trained_model_val_all_annotation_' + str(index) + '.pth')
    print('Finished Training')