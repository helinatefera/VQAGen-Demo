import os
import pickle
import random
import torch
import gc
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TimesformerModel
import torch.nn as nn
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.proj(x).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1)

device = "cpu"
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
ts_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device).eval()
D = ts_model.config.hidden_size
attn = SimpleAttention(D).to(device)

MAX_FRAMES, CHUNK = 16, 4

def process_single_video(frames_folder):
    # 1) Gather and sort all frame file paths
    frame_files = sorted(
        os.path.join(frames_folder, fname)
        for fname in os.listdir(frames_folder)
        if fname.lower().endswith((".jpg", ".png"))
    )
    if not frame_files:
        return None, None

    # 2) Sample up to MAX_FRAMES evenly
    indices = np.linspace(0, len(frame_files) - 1, min(MAX_FRAMES, len(frame_files)), dtype=int)
    sampled = [frame_files[i] for i in indices]

    cls_list = []
    frame_feats = []

    # 3) Process in chunks
    for i in range(0, len(sampled), CHUNK):
        batch_paths = sampled[i : i + CHUNK]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inp = processor(images=imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            out = ts_model(**inp).last_hidden_state  # shape: [1, 1 + T*P, D]

        cls_list.append(out[:, 0, :])  # [1, D]
        B, TP1, _ = out.shape
        T = len(imgs)
        tkn = out[:, 1:, :].reshape(B, T, -1, D).mean(2)  # [1, T, D]
        frame_feats.append(tkn)  # list of tensors [1, T, D]

        # del imgs, inp, out, tkn
        # gc.collect()
        # if device == "cuda":
        #     torch.cuda.empty_cache()

    # 4) Aggregate CLS tokens
    cls_vec = torch.cat(cls_list, dim=0).mean(0, keepdim=True)  # [1, D]

    # 5) Concatenate temporal features and apply attention
    frames_cat = torch.cat(frame_feats, dim=1)  # [1, T_total, D]
    temporal = attn(frames_cat)  # [D]

    # 6) Build feature dict and save
    vid = os.path.basename(os.path.normpath(frames_folder))
    features = {
        vid: {
            "cls": cls_vec.cpu().numpy(),         # shape (1, D)
            "temporal": temporal.detach().cpu().numpy()  # shape (D,)
        }
    }
    # with open(out_path, "wb") as f:
    #     pickle.dump(features, f)

    # 7) Return numpy arrays
    return features[vid]["cls"], features[vid]["temporal"]

if __name__ == "__main__":
    frames_folder = sys.argv[1]
    out_path = sys.argv[2]
    cls_feat, temp_feat = process_single_video(frames_folder, out_path)
    if cls_feat is not None:
        print("CLS shape:", cls_feat.shape, "TEMP shape:", temp_feat.shape)
    else:
        print("No frames found in", frames_folder)
