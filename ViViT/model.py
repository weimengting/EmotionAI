import av
import numpy as np
import torch
import torch.nn as nn

from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download
# (32, 360, 640, 3)
np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
container = av.open(file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)

video = read_video_pyav(container=container, indices=indices)

class VIVIT(nn.Module):
    def __init__(self, num_classes=2):
        super(VIVIT, self).__init__()

        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")  # (video_length, )
        self.model.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(**x)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400") # (video_length, )


inputs = image_processor(list(video), return_tensors="pt")
print(inputs['pixel_values'].shape)



# x = torch.rand((1, 3, 16, 256, 256))
# y = model(x)

if __name__ == '__main__':
    model = VIVIT(num_classes=2)
    print(model)
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
    print(logits.shape)
    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    # print(y.shape)