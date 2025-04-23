# EmotionAI

#### Visit the link of OneDrive shared to you
1. Put the downstream.pth under the path ./EmoCLIP/models
2. Put the marlin_vit_small_ytf.encoder.pt under the path ./MARLIN/.marlin
3. Put the swin_base_patch244_window1677_sthv2.pth under the path ./swin_transformer/models/checkpoints
4. Put the vivit-b-16x2-kinetics400 under the path ./ViViT
5. The dataloader and the training scripts is placed in the former_DFER, so for other models to run, it can be done by copying those scripts.


## 🤪 Getting Started

#### Start with Former-DFER
1. Go inside ./former_DFER
```python
python train.py 
```
You can change the arguments by setting the classification manner(binary or 12 class) and other training settings.

2. For other models, basically you only need to copy the scripts in former-DFER to other baseline scripts,
change the model loaded, and then run again.
3. During training, the model will be saved every 10 epochs in the same directory as the baseline sacripts. The .log
file provides accuracy scores on the training dataset of every epoch. 
4. For validation, load the saved model and record the performance.
5. Note, for other baselines, names of the saved models are still need to change.