import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model  ## feature extraction from image
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),  # 512 -> 256차원 변환
            nn.ReLU(),
            nn.Linear(256, 128)  # 최종 128차원
        )
    def forward(self, images):
        print(f"Input to CLIP encode_image shape: {images.shape}")
        print(f"Input to CLIP encode_image type: {type(images)}")
        with torch.no_grad(): # clip 모델 파라미터 업데이트 freeze. 
            features = self.clip_model.encode_image(images) # CLIP 모델이 이미지로부터 추출한 512차원의 feature vector
        return self.projection_head(features.float())