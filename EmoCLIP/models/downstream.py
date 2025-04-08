from typing import Mapping, Any

import torch
from torch import nn
from EmoCLIP.models.video_clip import VClip

device = "cuda" if torch.cuda.is_available() else "cpu"

class DownstreamTask(nn.Module):
    def __init__(self, clip_model, d_model: int = 512, n_classes: int = 7):
        super().__init__()
        self.clip_model = clip_model
        self.backbone = self.clip_model.backbone.visual
        self.temporal = self.clip_model.temporal
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_classes)
        )

        self._load_pretrained()

    def _load_pretrained(self):
        state_dict = torch.load('./downstream.pth')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        self.clip_model.load_state_dict(new_state_dict)


    def encode_video(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        v = self.backbone(x).reshape(B, T, -1)
        v = self.temporal(v)
        v = v[:, 0]
        return v

    def forward(self, x):
        v = self.encode_video(x)
        out = self.mlp(v)
        return out


if __name__ == '__main__':
    # 这里必须224, 224 的图像尺寸才行
    x = torch.randn((1, 16, 3, 224, 224)).to(device)
    model = VClip(num_layers=2)
    downstream = DownstreamTask(clip_model=model, n_classes=2).to(device=device, dtype=torch.float32)

    y = downstream(x)
    print(y.shape)