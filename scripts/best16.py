# best16.py
import os
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
import clip

device = "cpu"
model, processor = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, question, output_base_folder="best16", num_selected=16):
    """
    Sample all frames, pick ‘num_selected’ representative ones guided by ‘question’,
    save them under output_base_folder/<video_id>/, and return that folder path.
    """
    # 1. load video and sample all frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # 2. compute frame features
    def compute_frame_features(frames):
        feats = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_in = processor(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                v = model.encode_image(img_in)
            feats.append(v.cpu().numpy().squeeze())
        return np.array(feats)

    # 3. compute question embedding
    def compute_caption_embedding(caption):
        txt_in = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            t = model.encode_text(txt_in)
        return t.cpu().numpy().squeeze()

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # 4. if fewer frames than requested, just save all
    if len(frames) < num_selected:
        selected = frames
    else:
        ff = compute_frame_features(frames)
        qe = compute_caption_embedding(question)
        # cluster into num_selected groups
        kmeans = KMeans(n_clusters=num_selected, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ff)
        selected = []
        for i in range(num_selected):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                continue
            sims = [cosine_similarity(ff[j], qe) for j in idxs]
            best_idx = idxs[np.argmax(sims)]
            selected.append(frames[best_idx])

    # 5. save picked frames and return folder path
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    out_folder = os.path.join(output_base_folder, vid_id)
    os.makedirs(out_folder, exist_ok=True)
    for i, frame in enumerate(selected):
        cv2.imwrite(os.path.join(out_folder, f"frame_{i:03d}.jpg"), frame)

    return out_folder
