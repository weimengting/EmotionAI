import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Tuple

#
class VideoDataset(Dataset):
    def __init__(
            self,
            video_folders: List[str],
            image_size: int = 256,
            sample_frames: int = 16,
            sample_rate: int = 8,
            sign: str = 'binary',
    ):
        super().__init__()
        self.video_folders = video_folders
        self.image_size = image_size
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.sign = sign
        self.binary_class_dict = {'pose': 0, 'spon': 1}
        self.multi_class_dict = {'0_2': 0, '0_3': 1, '0_4': 2, '0_5': 3, '0_6': 4, '0_7': 5,
                                 '1_2': 6, '1_3': 7, '1_4': 8, '1_5': 9, '1_6': 10, '1_7': 11}

        self.data_lst, self.path_lst = self.generate_data_lst()

        self.pixel_transform = self.setup_transform()

    def generate_data_lst(self):
        path_lst = []
        data_lst = {}
        for folder in self.video_folders:
            video_folder = Path(folder)
            prefix = str(video_folder).split('/')[-1].split('_')[0]

            for video_dir in sorted(video_folder.iterdir()):
                res = self.is_valid(video_dir, prefix)
                if res[0]:
                    data_lst[video_dir] = res[1] # add labels here
                    path_lst.append(video_dir)

        return data_lst, path_lst
    # need to modify
    def is_valid(self, video_dir, prefix):
        if self.sign == 'binary':
            return True, prefix

        elements = str(video_dir).split('/')[-1].split('_')
        label = elements[-1]
        binary_label = self.binary_class_dict[prefix]
        return True, str(binary_label) + '_' + label


    def setup_transform(self):
        pixel_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        return pixel_transform

    def set_clip_idx(self, video_length):
        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int
        ).tolist()
        return clip_idxes


    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            ret_lst = []
            for img in images:
                if isinstance(img, list):
                    transformed_sub_images = [transform(sub_img) for sub_img in img]
                    sub_ret_tensor = torch.cat(transformed_sub_images, dim=0)  # (c*n, h, w)
                    ret_lst.append(sub_ret_tensor)
                else:
                    transformed_images = transform(img)
                    ret_lst.append(transformed_images)  # (c*1, h, w)
            ret_tensor = torch.stack(ret_lst, dim=0)  # (f, c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        video_dir = self.path_lst[idx]

        # tgt frames indexes
        video_length = len(list(video_dir.iterdir()))
        clip_idxes = self.set_clip_idx(video_length)  # sample a video clip

        img_path_lst = sorted([img.name for img in video_dir.glob("*.jpg")])
        tgt_vidpil_lst = []

        # tgt frames
        # guid frames: [[frame0: n_type x pil], [frame1: n x pil], [frame2: n x pil], ...]
        for c_idx in clip_idxes:
            tgt_img_path = video_dir / img_path_lst[c_idx]
            tgt_img_pil = Image.open(tgt_img_path)
            tgt_vidpil_lst.append(tgt_img_pil)

        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.pixel_transform, state)
        label = self.data_lst[video_dir]
        # print(tgt_vid.shape) [24, 3, 512, 512]
        # print(tgt_guid_vid.shape) [24, 9, 512, 512]
        if self.sign == 'binary':
            label = self.binary_class_dict[label]
        else:
            label = self.multi_class_dict[label]
        return tgt_vid, label  # [16, 3, 256, 256]


if __name__ == '__main__':
    train_dataset = VideoDataset(
        video_folders=["/home/mengting/Desktop/EmotionAI/data/pose_cropped",
                       "/home/mengting/Desktop/EmotionAI/data/spon_cropped"
                       ],
        image_size=256,
        sign='binary'
    )
    print(len(train_dataset))
    for i in range(train_dataset.__len__()):
        tgt, label = train_dataset.__getitem__(i)
        print(label)

    # label_path = '/media/mengting/Expansion/CMVS_projects/EmotionAI/SPFEED_dataset/SPFEED_dataset/verified_lable/SPFEED_Beha_Anno_spon_validcut.xlsx'
    # df = pd.read_excel(label_path)
    # category_counts = df['Matching Values'].value_counts()
    # print(category_counts)