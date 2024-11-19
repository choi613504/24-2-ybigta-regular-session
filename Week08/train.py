import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import CustomDataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze all CLIP model layers
# 일단은 classifier만 학습
for param in model.parameters():
    param.requires_grad = False


# Initialize custom classifier model
classifier_model = CustomCLIPClassifier(model).to(device)
optimizer = torch.optim.Adam(classifier_model.projection_head.parameters(), lr=1e-4)

# Contrastive Loss 정의
def contrastive_loss(features, temperature=0.5):
    features = features / features.norm(dim=1, keepdim=True)  # Normalize
    similarity_matrix = torch.mm(features, features.T)  # Cosine similarity
    labels = torch.arange(features.size(0)).to(device)
    logits = similarity_matrix / temperature
    return nn.CrossEntropyLoss()(logits, labels)

# Augment 함수 정의
def augment(image):
    # image가 Tensor라면 PIL 이미지로 변환
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)

    # 항상 augmentations를 정의
    augmentations = transforms.Compose([
        transforms.Resize((300, 300)), 
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
    ])
    augmented_image = augmentations(image)
    
    # Debugging: Ensure the shape is [3, 224, 224]
    if augmented_image.shape[0] != 3:
        raise ValueError(f"Augmented image has invalid channels: {augmented_image.shape[0]}")
    
    return augmented_image


dataset = load_from_disk("/root/project/Representational-Learning/dataset/train")
custom_dataset = CustomDataset(dataset, preprocess, augment)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Training loop
classifier_model.train()
for epoch in range(5):  # Train for 5 epochs, adjust as needed
    total_loss = 0
    for images, _ in tqdm(dataloader):
        images = torch.stack([augment(image) for image in images], dim=0) # positive pair 생성
        images = images.to(device)

        optimizer.zero_grad()
        features = classifier_model(images)
        loss = contrastive_loss(features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
model_save_path = "/root/project/Representational-Learning/saved_model.pth"
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
