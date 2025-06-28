import random
import os
import pickle
import torch
import torch.nn as nn
import clip
from PIL import Image
import torchvision
from torchvision import transforms
import transformers
import gradio as gr
from transformers import BertModel, BertTokenizer
import numpy as np
from scripts.best16 import extract_frames
from scripts.timeSformer import process_single_video as extract_timesformer_feats
from scripts.fasRCNN import process_single_video as extract_obj_feats  # same for both langs


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

# set_seed(42)


device = "cuda"

# Visual extractor setup (shared)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
fastrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
fastrcnn_model.eval()
det_transform = transforms.Compose([transforms.ToTensor()])

# Amharic BCMA + tokenizer
class BidirectionalCrossModalAttention(nn.Module):
    def __init__(self, bert_model_name, num_answers, hidden_dim=512, dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.visual_proj = nn.ModuleDict({
            'cls': nn.Sequential(nn.Linear(768, 768), nn.BatchNorm1d(768)),
            'temp': nn.Sequential(nn.Linear(768, 768), nn.BatchNorm1d(768)),
            'obj': nn.Sequential(nn.Linear(1024, 768), nn.BatchNorm1d(768))
        })
        self.text2vis_attn = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout_rate)
        self.vis2text_attn = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_answers)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, cls_feat, temp_feat, obj_feat):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state.transpose(0, 1)          # (seq_len, batch, 768)
        visual_feats = torch.stack([
            self.visual_proj['cls'](cls_feat),
            self.visual_proj['temp'](temp_feat),
            self.visual_proj['obj'](obj_feat)
        ], dim=1).transpose(0, 1)                                        # (3, batch, 768)

        t2v_out, _ = self.text2vis_attn(query=text_feat, key=visual_feats, value=visual_feats)
        t2v_vec = t2v_out.mean(dim=0)                                    # (batch, 768)
        v2t_out, _ = self.vis2text_attn(query=visual_feats, key=text_feat, value=text_feat)
        v2t_vec = v2t_out.mean(dim=0)                                    # (batch, 768)

        fused = torch.cat([t2v_vec, v2t_vec], dim=1)                     # (batch, 1536)
        return self.classifier(fused)                                    # (batch, num_answers)

# Load Amharic
state_am = torch.load("model/am_BCMA.pt", map_location=device)
num_answers_am = state_am["classifier.3.weight"].shape[0]
model_am = BidirectionalCrossModalAttention(
    bert_model_name="Davlan/bert-base-multilingual-cased-finetuned-amharic",
    num_answers=num_answers_am,
    hidden_dim=512,
    dropout_rate=0.3,
    num_heads=8
)
model_am.load_state_dict(state_am, strict=False)
model_am.eval()
tokenizer_am = BertTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-finetuned-amharic")
idx2ans_am = {}
with open("answer_set/am_answer_set.txt", "r", encoding="utf-8") as f:
    for line in f:
        i, a = line.strip().split(": ")
        idx2ans_am[int(i)] = a[1:-1]

# Load English
state_en = torch.load("model/en_BCMA.pt", map_location=device)
num_answers_en = state_en["classifier.3.weight"].shape[0]
model_en = BidirectionalCrossModalAttention(
    bert_model_name="bert-base-uncased",
    num_answers=num_answers_en,
    hidden_dim=512,
    dropout_rate=0.3,
    num_heads=8
)
model_en.load_state_dict(state_en, strict=False)
model_en.eval()
tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
idx2ans_en = {}
with open("answer_set/en_answer_set.txt", "r", encoding="utf-8") as f:
    for line in f:
        i, a = line.strip().split(": ")
        idx2ans_en[int(i)] = a[1:-1]

def load_caption_dict(path):
    caption_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                video_id, caption = line.split(",", 1)
                caption_dict[video_id.strip()] = caption.strip()
            except ValueError:
                continue
    return caption_dict

caption_dict_en = load_caption_dict("annotation/en_annotation.txt")
caption_dict_am = load_caption_dict("annotation/am_annotation.txt")

@torch.inference_mode()
def answer_question(lang, video_file, question, current_video_path, max_q_len=32):
    answer = ""
    video_output = video_file or current_video_path
    video_name = os.path.splitext(os.path.basename(video_output))[0]
    if lang == "Amharic":
        caption_dict = caption_dict_am
    else:
        caption_dict = caption_dict_en

    question = caption_dict.get(video_name, question)
    print(f"Video: {video_name}, Question: {question}")
    if video_file and question:
        try:
            # 1) select best-16 frames
            frames_folder = extract_frames(video_file, question)

            # 2) timesformer features
            cls_np, temp_np = extract_timesformer_feats(frames_folder)
            cls_feat = torch.tensor(cls_np, dtype=torch.float).view(1, 768)
            temp_feat = torch.tensor(temp_np, dtype=torch.float).view(1, 768)

            # 3) object features (same script, regardless of lang)
            obj_np = extract_obj_feats(question, frames_folder)
            obj_feat = torch.tensor(obj_np, dtype=torch.float).view(1, 1024)

            # 4) choose tokenizer & model based on lang
            if lang == "Amharic":
                toks = tokenizer_am(question, padding="max_length", truncation=True, max_length=max_q_len, return_tensors="pt")
                model_sel = model_am
                idx2ans = idx2ans_am
            else:
                toks = tokenizer_en(question, padding="max_length", truncation=True, max_length=max_q_len, return_tensors="pt")
                model_sel = model_en
                idx2ans = idx2ans_en

            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]

            with torch.no_grad():
                logits = model_sel(input_ids, attention_mask, cls_feat, temp_feat, obj_feat)
                print(f"Logits shape: {logits.shape}, Values: {logits}")
                pred_idx = logits.argmax(dim=1).item()
            answer = idx2ans.get(pred_idx, "Error: Answer not found.")
        except Exception as e:
            answer = f"Error: {e}"

    new_current = video_file or current_video_path
    return answer, video_output, new_current

css = """
#video_component {
    max-width: 800px;
    width: 100%;
    height: 450px;
    aspect-ratio: 16/9;
    margin-left: auto;
    margin-right: auto;
    display: block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Video Question Answering ")
    gr.Markdown("Upload a video (preferably MP4 with H.264 codec for best compatibility), pick English or Amharic, then ask a question. Only the text embedder changes; all visual features reuse the same scripts.")

    current_video_path = gr.State(value=None)

    with gr.Row():
        lang_choice = gr.Radio(["English", "Amharic"], label="Select Language", value="Amharic")
        video_input = gr.Video(label="Upload Video", interactive=True, elem_id="video_component")
        question_input = gr.Textbox(label="Question", placeholder="Enter your question here...")
    submit_button = gr.Button("Submit")

    with gr.Row():
        video_output = gr.Video(label="Video", elem_id="video_component", autoplay=True, loop=True)
        answer_output = gr.Textbox(label="Answer", lines=2)

    submit_button.click(
        fn=answer_question,
        inputs=[lang_choice, video_input, question_input, current_video_path],
        outputs=[answer_output, video_output, current_video_path]
    )

demo.launch()