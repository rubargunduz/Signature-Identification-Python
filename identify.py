# identify.py
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import euclidean
import cv2

def preprocess_image(path, target_size=300):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    h, w = img.shape
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (target_size, target_size))
    return Image.fromarray(img)

def get_embedding_model():
    import torchvision.models as models
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # remove classification head
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    return model, transform, device

def extract_embedding(img_pil, model, transform, device):
    with torch.no_grad():
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        emb = model(img_tensor).squeeze().cpu().numpy()
    return emb

def identify_signature(test_img_path, database_path="signature_features.pkl", threshold=10):
    with open(database_path, "rb") as f:
        db = pickle.load(f)

    model, transform, device = get_embedding_model()
    test_img = preprocess_image(test_img_path)
    test_emb = extract_embedding(test_img, model, transform, device)

    best_match = None
    best_score = float("inf")
    for person, emb_list in db.items():
        for emb in emb_list:
            score = euclidean(test_emb, emb)
            if score < best_score:
                best_score = score
                best_match = person

    if best_score <= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score

# Optional visualization for debugging
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_img_path = "assets/6.png"

    who, score = identify_signature(test_img_path)
    print(f"Identified as: {who}, Distance: {score:.4f}")

    pre_img = preprocess_image(test_img_path)
    original = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed")
    plt.imshow(pre_img, cmap='gray')
    plt.tight_layout()
    plt.show()
