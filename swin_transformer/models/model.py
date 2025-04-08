import torch
import torch.nn as nn
from collections import OrderedDict
from swin_transformer.models.video_swin_transformer import SwinTransformer3D





class SwinTrans(nn.Module):
    def __init__(self, num_classes):
        super(SwinTrans, self).__init__()
        self.num_classes = num_classes
        self.model = SwinTransformer3D(embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       patch_size=(2, 4, 4),
                                       window_size=(16, 7, 7),
                                       drop_path_rate=0.4,
                                       patch_norm=True)
        self._load_checkpoints()
        self.linear = nn.Linear(1024, self.num_classes)

    def _load_checkpoints(self):
        checkpoint = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth')

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        b, l, c, h, w = x.shape
        x = x.view(b, c, l, h, w)
        x = self.model(x)  # [b * l, 2]
        x = torch.mean(torch.mean(x, dim=-1), dim=-1)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    dummy_x = torch.rand(2, 16, 3, 224, 224)
    st = SwinTrans(num_classes=2)
    y = st(dummy_x)
    print(y.shape)