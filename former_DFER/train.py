import os

import torch
import torch.nn as nn
import torch.optim as optim

from former_DFER.models.ST_Former import GenerateModel
from video_dataset import VideoDataset, obtain_subjects
import argparse
import logging



# btz = 2
# index = 1   # index is for fold, from 1 to 10
# cls_sign = 'binary' # choose binary or multi


def main(args):
    # 设置日志系统（只需设置一次）
    logging.basicConfig(
        filename=f'output_train_{args.index}.log',  # 保存的日志文件名
        level=logging.INFO,  # 日志等级（还可以用 DEBUG, WARNING, ERROR 等）
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )

    train_subjects, test_subjects = obtain_subjects(args.index)
    # 加载训练集
    train_dataset = VideoDataset(
            video_folders=["/media/mengting/Expansion/CMVS_projects/EmotionAI/SPFEED_dataset/SPFEED_dataset/added/aligned/pose_cropped",
                           "/media/mengting/Expansion/CMVS_projects/EmotionAI/SPFEED_dataset/SPFEED_dataset/added/aligned/spon_cropped"
                           ],
            image_size=224,
            sign=args.sign,
            sublst=train_subjects,
        )

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.btz,
                                              shuffle=True, num_workers=4)

    device = 'cuda'

    if args.sign == 'binary':
        num_classes = 2
    elif args.sign == 'multi':
        num_classes = 12
    else:
        raise ValueError('sign must be either "binary" or "multi"')
    model = GenerateModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # adam优化器

    epoches = args.epochs

    # save model every 10 epochs
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
        logging.info(f'Epoch {epoch+1}, Accuracy on training set: {100 * correct / total}%, mean loss is: {running_loss / i}')

        model_save_path = './saved_models/fold_'+ str(args.index)
        os.makedirs(model_save_path, exist_ok=True)

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'trained_former_DFERmodel_val_' + str(epoch) + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--btz", type=int, default=2)
    parser.add_argument("--index", type=int, default=1) # which fold
    parser.add_argument("--sign", type=str, default="multi") # multi or binary
    parser.add_argument("--epochs", type=int, default=270)  # total epochs
    args = parser.parse_args()
    main(args)