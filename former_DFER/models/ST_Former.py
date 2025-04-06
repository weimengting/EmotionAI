import torch
from torch import nn
from former_DFER.models.S_Former import spatial_transformer
from former_DFER.models.T_Former import temporal_transformer
import matplotlib.pyplot as plt
import numpy as np

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


class GenerateModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, self.num_classes)
        self._load_state()

    def _load_state(self):
        new_state_dict = {}
        state_dict = torch.load('/home/mengting/projects/emotionAI/former_DFER/models/pretrained_model_set_1.pth')
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # 移除前缀
                new_key = k[7:]  # 删除前七个字符 'module.'
            else:
                new_key = k
            if "fc." not in new_key:
                new_state_dict[new_key] = v
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 256, 256))
    model = GenerateModel(num_classes=6)
    out = model(img)
    print(out.shape)
    # new_state_dict = {}
    # state_dict = torch.load('model_set_1.pth')
    # for k, v in state_dict["state_dict"].items():
    #     if k.startswith('module.'):
    #         # 移除前缀
    #         new_key = k[7:]  # 删除前七个字符 'module.'
    #     else:
    #         new_key = k
    #     if "fc." not in new_key:
    #         new_state_dict[new_key] = v
    # model.load_state_dict(new_state_dict, strict=False)
    # print(state_dict.keys())
    # model.load_state_dict(state_dict["state_dict"])
    #model.load_state_dict(state_dict)
