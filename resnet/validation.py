import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from dataset import SundownSyndrome, SundownSyndromeAllAnnotation
from model import light_resnet18

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小到48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

btz = 2
index = 1

# 加载测试集
testset = SundownSyndrome(data_root='/media/mengting/Expansion/CMVS_projects/SundownSyndrome/SS_dataset/SS_video_clips_with_labels2',
                                       phase='test', index=index)
testloader = torch.utils.data.DataLoader(testset, batch_size=btz, drop_last=False,
                                         shuffle=False, num_workers=1)

device = 'cuda'

model = light_resnet18().to(device)  # 使用定义的模型，并将模型移动到GPU


dict_s = torch.load('./trained_model_val_' + str(index) + '.pth')
model.load_state_dict(dict_s)

pred = []
label = []

def add_to_list(pre, lab, pred_total, lab_total):
    pp = pre.detach().cpu().numpy().tolist()
    pred_total = pred_total + pp
    ll = lab.detach().cpu().numpy().tolist()
    lab_total = lab_total + ll
    return pred_total, lab_total

with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        # 获取输入数据
        inputs, labels = data[0].to(device), data[1].to(device)

        # 前向 + 反向 + 优化
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        pred, label = add_to_list(predicted, labels, pred, label)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'The test accuracy is {100 * correct / total}')
# 预测的准确率结果为80%
print(pred)
print(label)


if __name__ == '__main__':
    macro_F1 = f1_score(label, pred, average='macro')
    macro_F1 = macro_F1 * 100
    print(macro_F1)


# 75.47 43.01
# 45.28 31.17
# 69.62 59.83
# 79.27 53.58
# 68.97 47.01
# 67.72 56.92
