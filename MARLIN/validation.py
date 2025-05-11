import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from video_dataset import VideoDataset, obtain_subjects
from model import Finetune_Marlin as GenerateModel
import os
import argparse
import logging


def main(args):
    logging.basicConfig(
        filename=f'output_val_{args.index}.log',  # 保存的日志文件名
        level=logging.INFO,  # 日志等级（还可以用 DEBUG, WARNING, ERROR 等）
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )

    train_subjects, test_subjects = obtain_subjects(args.index)
    test_dataset = VideoDataset(
        video_folders=["/home/hq/Documents/data/SPFEED dataset/pose_cropped",
                        "/home/hq/Documents/data/SPFEED dataset/spon_cropped"
                        ],
        image_size=224,
        sign=args.sign,
        sublst=test_subjects,
    )

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.btz,
                                             shuffle=True, num_workers=4)

    device = 'cuda'
    if args.sign == 'binary':
        num_classes = 2
    elif args.sign == 'multi':
        num_classes = 12
    else:
        raise ValueError('sign must be either "binary" or "multi"')

    model = GenerateModel(n_classes=num_classes).to(device)  # 使用定义的模型，并将模型移动到GPU

    model_save_path = './saved_models/fold_' + str(args.index)
    epoch = args.epoch

    dict_s = torch.load(os.path.join(model_save_path, 'trained_former_DFERmodel_val_' + str(epoch) + '.pth'))
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
            inputs = inputs.permute(0, 2, 1, 3, 4)
            
            # 前向 + 反向 + 优化
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred, label = add_to_list(predicted, labels, pred, label)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logging.info(f'The test accuracy is {100 * correct / total}')

    logging.info(f"pred: {pred}")
    logging.info(f"label: {label}")

    macro_F1 = f1_score(label, pred, average='macro')
    macro_F1 = macro_F1*100
    logging.info(f"macro F1 score: {macro_F1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--btz", type=int, default=4)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--sign", type=str, default="binary")
    parser.add_argument("--epoch", type=int, default=90)
    args = parser.parse_args()
    main(args)