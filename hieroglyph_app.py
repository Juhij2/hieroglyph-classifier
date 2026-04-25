import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import json
import numpy as np
import cv2

st.set_page_config(page_title="Hieroglyph Classifier", page_icon="𓂀", layout="wide")

@st.cache_resource
def load_model():
    with open('label_mapping.json') as f:
        idx_to_label = json.load(f)
    num_classes = len(idx_to_label)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    model.load_state_dict(torch.load('hieroglyph_model.pth', map_location='cpu'))
    model.eval()
    return model, idx_to_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SIGN_INFO = {
    'A1': 'Seated man', 'A2': 'Man with hand to mouth', 'A17': 'Boy with hand to mouth',
    'B1': 'Seated woman', 'D1': 'Head in profile', 'D2': 'Face', 'D19': 'Nose and eye',
    'D21': 'Mouth', 'E1': 'Bull', 'E17': 'Jackal', 'E34': 'Hare',
    'G1': 'Egyptian vulture', 'G5': 'Falcon', 'G17': 'Owl', 'G43': 'Quail chick',
    'I9': 'Horned viper', 'N35': 'Water ripple', 'O1': 'House', 'R8': 'Flag on pole',
    'S29': 'Folded cloth', 'T28': 'Butcher block', 'V4': 'Lasso', 'W24': 'Bowl',
    'X1': 'Bread loaf', 'Z1': 'Single stroke', 'Z2': 'Multiple strokes',
}

def classify_crop(crop_bgr, model, idx_to_label):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top5_probs, top5_idx = probs.topk(5)
    top_label = idx_to_label[str(top5_idx[0][0].item())]
    top_conf = top5_probs[0][0].item()
    top5 = [(idx_to_label[str(top5_idx[0][i].item())], top5_probs[0][i].item()) for i in range(5)]
    return top_label, top_conf, top5

def detect_signs(img_bgr):
    scale = 6
    img_up = cv2.resize(img_bgr, (img_bgr.shape[1]*scale, img_bgr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if 500 < area < 20000 and bw > 20 and bh > 20:
            # Scale back to original coordinates
            boxes.append((x//scale, y//scale, bw//scale, bh//scale))
    return boxes

st.title("𓂀 Hieroglyph Sign Classifier")
st.markdown("---")

model, idx_to_label = load_model()

mode = st.radio("Mode", ["Single Sign", "Papyrus Column (auto-detect signs)"], horizontal=True)
st.markdown("---")

if mode == "Single Sign":
    uploaded_file = st.file_uploader("Upload a single hieroglyph image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(img, caption="Uploaded hieroglyph", use_container_width=True)
        with col2:
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)
                top5_probs, top5_idx = probs.topk(5)
            top_label = idx_to_label[str(top5_idx[0][0].item())]
            top_prob = top5_probs[0][0].item()
            st.markdown(f"### Predicted Sign: `{top_label}`")
            if top_label in SIGN_INFO:
                st.markdown(f"**Meaning:** {SIGN_INFO[top_label]}")
            st.markdown(f"**Confidence:** {top_prob*100:.1f}%")
            if top_prob > 0.8:
                st.success("High confidence prediction")
            elif top_prob > 0.5:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence — image may be unclear")
            st.markdown("#### Top 5 Predictions")
            for prob, idx in zip(top5_probs[0], top5_idx[0]):
                label = idx_to_label[str(idx.item())]
                pct = prob.item() * 100
                st.progress(int(pct), text=f"`{label}` — {pct:.1f}%")

else:
    uploaded_file = st.file_uploader("Upload a papyrus column image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting signs..."):
            boxes = detect_signs(img_bgr)

        st.markdown(f"**{len(boxes)} signs detected**")

        if boxes:
            # Draw boxes on image
            img_draw = img_bgr.copy()
            results = []
            for (x, y, bw, bh) in boxes:
                pad = 3
                crop = img_bgr[max(0,y-pad):y+bh+pad, max(0,x-pad):x+bw+pad]
                if crop.size == 0:
                    continue
                label, conf, top5 = classify_crop(crop, model, idx_to_label)
                results.append((x, y, bw, bh, label, conf, top5))
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)
                cv2.rectangle(img_draw, (x, y), (x+bw, y+bh), color, 1)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), caption="Detected signs", use_container_width=True)

            with col2:
                st.markdown("#### Identified Signs")
                # Sort top to bottom
                results_sorted = sorted(results, key=lambda r: r[1])
                for i, (x, y, bw, bh, label, conf, top5) in enumerate(results_sorted):
                    conf_color = "🟢" if conf > 0.5 else "🟡" if conf > 0.3 else "🔴"
                    meaning = SIGN_INFO.get(label, "")
                    st.markdown(f"{conf_color} **{label}** {f'— {meaning}' if meaning else ''} `{conf*100:.0f}%`")

                high = [r for r in results if r[5] > 0.5]
                st.markdown("---")
                st.markdown(f"**Summary:** {len(results)} signs detected, {len(high)} high confidence (>50%)")

st.markdown("---")
st.markdown("**Model:** ResNet-50 | **Accuracy:** 91.8% on clean images | **Classes:** 170 Gardiner signs | **Detection:** OpenCV contour analysis")
