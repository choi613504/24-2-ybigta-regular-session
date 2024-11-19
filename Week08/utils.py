import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image



device = "cuda" if torch.cuda.is_available() else "cpu"
# Preprocess data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, augment):
        self.dataset = dataset
        self.preprocess = preprocess
        self.augment = augment # 데이터 증강 파이프라인 추가

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'] #라벨링 정보는 필요 없어서 삭제
        image = Image.fromarray(np.array(image))  # Adjust if not PIL Image
        # 원본 데이터 전처리
        original = self.preprocess(image)
        if original.shape[0] != 3:  # 채널 수가 3이 아니면 오류 출력
            raise ValueError(f"Invalid channel size for original: {original.shape}")

        # 증강된 데이터 생성
        augmented = self.augment(image)
        if augmented.shape[0] != 3:  # 채널 수가 3이 아니면 오류 출력
            raise ValueError(f"Invalid channel size for augmented: {augmented.shape}")

        return original, augmented  # Positive Pair 반환
    




# ## confidence 및 accuracy 시각화  --> contrastive learning에서는 확률과 클래스 예측이 없으므로 confidence나 accuracy 측정할 필요 X
# def plot_confidence_and_accuracy(probs, labels, n_bins=100):
#     bin_boundaries = np.linspace(0, 1, n_bins + 1)
#     bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
#     bin_acc = []
#     bin_conf = []

#     for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
#         # Determine whether each probability falls in the current bin
#         max_probs = probs.max(axis=1)  # Get the maximum probability for each sample
#         in_bin = (max_probs >= bin_lower) & (max_probs < bin_upper)

#         if np.sum(in_bin) > 0:
#             # Compute average confidence and accuracy for the current bin
#             avg_confidence_in_bin = np.mean(max_probs[in_bin])
#             accuracy_in_bin = np.mean(labels[in_bin] == np.argmax(probs[in_bin], axis=1))
#             bin_conf.append(avg_confidence_in_bin)
#             bin_acc.append(accuracy_in_bin)
#         else:
#             bin_conf.append(0)
#             bin_acc.append(0)

#     plt.figure(figsize=(12, 5))
    
#     # Plot confidence
#     plt.subplot(1, 2, 1)
#     plt.bar(bin_centers, bin_conf, width=0.05, alpha=0.7, label="Confidence")
#     plt.xlabel("Confidence")
#     plt.ylabel("Average Confidence")
#     plt.title("Model Confidence Distribution")
#     plt.legend()

#     # Plot accuracy
#     plt.subplot(1, 2, 2)
#     plt.bar(bin_centers, bin_acc, width=0.05, alpha=0.7, label="Accuracy")
#     plt.xlabel("Confidence")
#     plt.ylabel("Accuracy")
#     plt.title("Model Accuracy per Confidence Bin")
#     plt.legend()

#     plt.show()
#     plt.savefig('confidence_accuracy.png')
#     plt.close()
def augment(image):
        from torchvision.transforms.functional import to_pil_image
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)  # 텐서를 PIL 이미지로 변환

        augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2, num_output_channels=3),
            transforms.ToTensor()
        ])
        return augmentations(image)
## 임베딩 시각화
def visualize_embeddings_with_tsne(model, dataloader):
    model.eval()
    embeddings = []
    # labels_list = []  # contrastive learning에는 클래스 정보 없으므로 제거하고 임베딩 자체를 t-SNE로 분석해야 함
    
    # with torch.no_grad():
    #     for images, labels in dataloader:
    #         images = images.to(device)
    #         features = model.clip_model.encode_image(images).cpu().numpy()
    #         embeddings.append(features)
    #         labels_list.extend(labels.numpy())

    with torch.no_grad():
        for original, augmented in dataloader:  # Positive Pair 입력
            original, augmented = original.to(device), augmented.to(device)
            original_features = model(original).cpu().numpy()
            augmented_features = model(augmented).cpu().numpy()
            embeddings.append(original_features)
            embeddings.append(augmented_features)

    embeddings = np.concatenate(embeddings)
    # labels_list = np.array(labels_list)

    # Standardize embeddings before applying t-SNE
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot t-SNE results
    plt.figure(figsize=(8, 8))
    # scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    # plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Contrastive Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    plt.savefig('contrastive_tsne_embeddings.png')
    plt.close()

## ece score 계산 
def compute_ece(probs, labels, n_bins=100):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Ensure the indexing matches the array dimensions
            in_bin_indices = np.where(in_bin)[0]  # Get the indices where in_bin is True
            
            # Calculate accuracy and confidence only for valid indices
            if len(in_bin_indices) > 0:
                accuracy_in_bin = np.mean(labels[in_bin_indices] == probs[in_bin_indices].argmax(axis=1))
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def augment(image):
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)

    augmentations = transforms.Compose([
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
    return augmentations(image)