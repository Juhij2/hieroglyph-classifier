"""Score the published classifier on the dataset's own held-out test page.

The training images (Dataset/train) come from nine photographs of Piankoff's
"The Pyramid of Unas"; Dataset/test is picture 7, which is never used in
training. Labels are read from the file names (e.g. 070003_M17.png -> M17).

    git clone https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs
    python evaluate_heldout_page.py --test-dir Egyptian_hieroglyphs/Dataset/test
"""
import argparse, collections, json, os
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ap = argparse.ArgumentParser()
ap.add_argument('--test-dir', required=True)
ap.add_argument('--model', default='hieroglyph_model.pth')
ap.add_argument('--labels', default='label_mapping.json')
args = ap.parse_args()

idx_to_label = json.load(open(args.labels))
classes = set(idx_to_label.values())
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(idx_to_label)))
model.load_state_dict(torch.load(args.model, map_location='cpu'))
model.eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

files = sorted(f for f in os.listdir(args.test_dir) if f.endswith('.png'))
outside, scored, top1, top5 = 0, 0, 0, 0
with torch.no_grad():
    for f in files:
        label = f.rsplit('_', 1)[1].rsplit('.', 1)[0]
        if label not in classes:
            outside += 1
            continue
        x = tf(Image.open(os.path.join(args.test_dir, f)).convert('RGB')).unsqueeze(0)
        pred = [idx_to_label[str(i)] for i in model(x)[0].topk(5).indices.tolist()]
        scored += 1; top1 += pred[0] == label; top5 += label in pred
print(f'test images: {len(files)}; labels outside the 170 classes: {outside}; scored: {scored}')
print(f'top-1 accuracy: {top1}/{scored} = {100 * top1 / scored:.1f}%')
print(f'top-5 accuracy: {100 * top5 / scored:.1f}%')
