# feature.py
import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from identify import preprocess_image, get_embedding_model, extract_embedding

def build_signature_database(folder_path, model, transform, device):
    db = {}
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        features = []
        for file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file)
            try:
                img = preprocess_image(img_path)  # returns a 300x300 grayscale PIL image
                emb = extract_embedding(img, model, transform, device)
                features.append(emb)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        if features:
            db[person_name] = features
    return db

if __name__ == "__main__":
    folder = "signature_db"
    model, transform, device = get_embedding_model()
    db = build_signature_database(folder, model, transform, device)

    with open("signature_features.pkl", "wb") as f:
        pickle.dump(db, f)

    print("Database created with people:", list(db.keys()))
