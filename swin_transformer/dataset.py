# dataloader，每个视频对应16张图像，resize的尺寸是48*48，二分类问题
# 400个样本，不需要

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import shutil

'''
没有图像的目录
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/3/2_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/3/3_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/35/1_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/35/2_0
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/35/5_0
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/36/1_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/39/1_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/39/2_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/5/1_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/52/9_1
/media/mengting/data2/SS_dataset/SS_video_clips_with_label/53/1_1
'''

nums = [14, 14, 14, 13, 13]

# 定义transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小到48x48
    transforms.ToTensor(),  # 将图像转换成Tensor
])
# 少于16张的图像补齐，多余的采样

def clear_imgs():
    data_root = '/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2'
    subjects = sorted(os.listdir(data_root))
    for subject in subjects:
        cur_subject_path = os.path.join(data_root, subject)
        videos = os.listdir(cur_subject_path)
        for video in videos:
            cur_video_path = os.path.join(cur_subject_path, video)
            if os.path.isdir(cur_video_path):
                imgs = os.listdir(cur_video_path)
                if len(imgs) < 16:
                    print(cur_video_path)
                    # shutil.rmtree(cur_video_path)

class SundownSyndrome(Dataset):
    """一个简单的自定义数据集类"""

    def __init__(self, data_root, phase, index):
        """
        初始化数据集
        :param data: 数据列表，每个元素是一个样本（例如，图像和标签）
        :param transform: 一个可选的转换函数或组合，用于对样本进行预处理
        """
        self.data_root = data_root
        self.phase = phase
        self.index = index # 表示哪一折用来测试
        self.data = self._get_video_list()

    def _get_video_list(self):
        data = []
        subjects = sorted(os.listdir(self.data_root))
        start_num = sum(nums[:self.index])
        end_num = start_num + nums[self.index]
        if self.phase == 'train':
            subjects = subjects[0:start_num] + subjects[end_num:len(subjects)]
        else:
            subjects = subjects[start_num:end_num]
        for subject in subjects:
            cur_subject_path = os.path.join(self.data_root, subject)
            videos = os.listdir(cur_subject_path)
            for video in videos:
                cur_video_path = os.path.join(cur_subject_path, video)
                if os.path.isdir(cur_video_path):
                    data.append(cur_video_path)
        return data

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据给定的索引idx加载并返回一个样本
        :param idx: 样本的索引
        """
        sample = self.data[idx]
        img_tmp_list = os.listdir(sample)
        imgs = [im for im in img_tmp_list if im.endswith('jpg')]
        imgs = sorted(imgs)
        # 选择16帧的图像
        selected = []
        gap = len(imgs) / 16
        cur = 0
        selected.append(imgs[0])
        for i in range(15):
            cur = cur + gap
            selected.append(imgs[int(cur)])
        selected = [os.path.join(sample, img) for img in selected]
        images = [Image.open(img) for img in selected]
        images = [transform(img) for img in images]
        images = torch.stack(images, dim=0)

        label = int(sample[-1])


        return images, label

# 五折交叉验证， 14, 14, 14, 13, 13
if __name__ == '__main__':
    ss = SundownSyndrome(data_root='/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2', phase='test', index=0)

    # print(ss._get_video_list())
    # # print(ss.data)
    # # ss.__getitem__(8)
    # clear_imgs()