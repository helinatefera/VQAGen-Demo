import os
import sys
import pickle
import torch
import torch.nn as nn
import clip
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np
import random


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
fastrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
fastrcnn_model.eval()
det_transform = transforms.Compose([transforms.ToTensor()])

model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
model_mul = clip_model  # reuse CLIP model for text embedding
tokenizer = None        # not needed here, we’ll call CLIP’s tokenizer via `clip.tokenize`

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.q = nn.Linear(feature_dim, hidden_dim)
        self.k = nn.Linear(feature_dim, hidden_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        Q = self.q(x)                                   # (T, D_hidden)
        K = self.k(x)                                   # (T, D_hidden)
        V = self.v(x)                                   # (T, D_feat)
        scores = torch.matmul(Q, K.transpose(0, 1)) * self.scale  # (T, T)
        attn = torch.softmax(scores, dim=1)             # (T, T)
        out = torch.matmul(attn, V)                     # (T, D_feat)
        return out.mean(dim=0)                          # (D_feat,)

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert output_dim % input_dim == 0
        self.num_queries = output_dim // input_dim
        self.queries = nn.Parameter(torch.randn(self.num_queries, input_dim))
        self.input_dim = input_dim

    def forward(self, x):
        # x: (K, input_dim)
        scores = torch.matmul(x, self.queries.t()) / (self.input_dim ** 0.5)  # (K, num_queries)
        weights = torch.softmax(scores, dim=0)                                # (K, num_queries)
        context = torch.matmul(weights.t(), x)                                # (num_queries, input_dim)
        return context.view(-1)                                               # (num_queries * input_dim,)

def process_single_video(question: str, frames_folder: str):
    """
    Given an Amharic question and a folder of frames (JPEG/PNG),
    run Faster-RCNN on each frame, match object crops to question tokens via CLIP,
    then apply TemporalAttention and AttentionAggregator to produce a 1024-dim feature.
    Returns a NumPy array of shape (1024,).
    """
    # 1) Tokenize question into CLIP text embeddings
    words = [w.strip() for w in question.split() if w.strip()]
    if not words:
        raise ValueError("Question must contain at least one token.")

    text_tokens = clip.tokenize(words).to(device)   # (num_words, 77), CLIP default
    with torch.no_grad():
        word_feats = model_mul.encode_text(text_tokens)  # (num_words, 512)
        word_feats /= word_feats.norm(dim=-1, keepdim=True)

    # 2) Collect (frame_id, word, feature) tuples
    vdata = []
    for idx, fname in enumerate(sorted(os.listdir(frames_folder))):
        img_path = os.path.join(frames_folder, fname)
        if not img_path.lower().endswith((".jpg", ".png")):
            continue

        img = Image.open(img_path).convert("RGB")
        inp = det_transform(img).to(device)           # (3, H, W)
        with torch.no_grad():
            pred = fastrcnn_model([inp])[0]

        boxes = pred["boxes"].cpu().numpy()           # (num_boxes, 4)
        scores = pred["scores"].cpu().numpy()         # (num_boxes,)
        valid_idxs = np.where(scores >= 0.5)[0]       # filter by confidence
            
        for i_box in valid_idxs:
            x1, y1, x2, y2 = map(int, boxes[i_box])
            crop = img.crop((x1, y1, x2, y2))
            clip_inp = preprocess(crop).unsqueeze(0).to(device)  # (1, 3, 224, 224)
            with torch.no_grad():
                feat = model_mul.encode_image(clip_inp)         # (1, 512)
                feat /= feat.norm(dim=-1, keepdim=True)         # normalize
                sim = (100.0 * feat @ word_feats.T).softmax(dim=-1)  # (1, num_words)
                best = sim.argmax(dim=-1).item()                # best word index

            vdata.append({
                "frame_id": idx,
                "word": words[best],
                "clip_feature": feat.cpu().numpy().reshape(-1)  # (512,)
            })

    if not vdata:
        raise RuntimeError(f"No valid object detections for any word in '{question}'.")

    feature_dim = vdata[0]["clip_feature"].shape[0]  # 512
    temporal_model = TemporalAttention(feature_dim, hidden_dim=128)
    aggregator = AttentionAggregator(input_dim=feature_dim, output_dim=1024)

    attended_per_word = {}
    unique_words = { e["word"] for e in vdata }
    for w in unique_words:
        seq = []
        for frame_idx in range(64):
            feats = [e["clip_feature"] 
                     for e in vdata 
                     if (e["frame_id"] == frame_idx and e["word"] == w)]
            if feats:
                tensor = torch.tensor(feats[0], dtype=torch.float32).to(device)
            else:
                tensor = torch.zeros(feature_dim, dtype=torch.float32).to(device)
            seq.append(tensor)
        x = torch.stack(seq, dim=0)  # (64, 512)
        with torch.no_grad():
            attended_per_word[w] = temporal_model(x)  # (512,)

    # 4) Aggregate all word-specific vectors into final video_rep
    object_vectors = list(attended_per_word.values())  # list of (512,) tensors
    x_all = torch.stack(object_vectors, dim=0)          # (K, 512)
    with torch.no_grad():
        video_rep = aggregator(x_all).cpu().numpy()     # (1024,)

    return video_rep

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fasRCNN.py <question> <frames_folder>")
        sys.exit(1)

    question = sys.argv[1]
    frames_folder = sys.argv[2]
    video_rep = process_single_video(question, frames_folder)
    print("video_rep shape:", video_rep.shape)  # (1024,)
