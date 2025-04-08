from marlin_pytorch import Marlin
import torch
import torch.nn as nn


class Finetune_Marlin(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.n_classes = n_classes
        self.marlin = Marlin.from_online("marlin_vit_small_ytf")
        self.classifier = nn.Linear(384, self.n_classes)


    def forward(self, x):
        x = self.marlin.extract_features(x, keep_seq=False)
        x = self.classifier(x)
        return x




if __name__ == '__main__':
    x = torch.rand(1, 3, 16, 224, 224)
    model = Finetune_Marlin(n_classes=2)
    y = model(x)
    print(y.shape)