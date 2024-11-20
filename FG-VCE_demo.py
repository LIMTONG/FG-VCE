import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, resize
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def apply_blur(image, radius=5):
    """
    使用Pillow库对图像进行模糊处理。
    :param image: 输入的PIL图像
    :param radius: 模糊半径，默认值为5
    :return: 模糊后的PIL图像
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load('./res50_cub200.pth'))
model = model.to(device)
model.eval()

features_before = None
def hook_before(module, input, output):
    global features_before
    features_before = output.clone().detach().to(device)


features_after = None
def hook_after(module, input, output):
    global features_after
    features_after = output.clone().detach().to(device)


model.layer4[2].register_forward_hook(hook_before)





def interpolate_to_224(activation_map):

    return F.interpolate(activation_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)


def pgd_attack(model, images, labels, eps=0.1, alpha=0.05, iters=100):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device).long()
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        if outputs.max(1)[1].item() != labels.item():
            break

    return images

def feature_replacement_attack(all_feature_units, attack_feature_map, img_tensor, true_class, max_iterations=100):
    attack_feature_map = attack_feature_map.clone().detach()
    best_attack_feature_map = attack_feature_map.clone().detach()
    rows, cols = attack_feature_map.size(2), attack_feature_map.size(3)
    iteration_count = 0
    mu_row, mu_col = rows / 2, cols / 2
    initial_sigma = min(rows, cols) / 6
    sigma_increment = initial_sigma / 5

    while iteration_count < max_iterations:
        current_sigma = initial_sigma + sigma_increment * iteration_count
        gaussian_indices = []
        for _ in range(49):
            row = int(np.clip(np.random.normal(mu_row, current_sigma), 0, rows - 1))
            col = int(np.clip(np.random.normal(mu_col, current_sigma), 0, cols - 1))
            gaussian_indices.append((row, col))

        for row, col in gaussian_indices:
            max_prob_increase = -float('inf')
            best_replacement_unit = None
            current_unit = attack_feature_map[:, :, row, col].view(1, -1)
            candidate_units = all_feature_units
            similarities = F.cosine_similarity(current_unit, candidate_units)
            top_50_candidates = torch.topk(similarities, 50, largest=True).indices


            for idx in top_50_candidates:
                candidate_unit = candidate_units[idx].view(1, 2048)
                temp_feature_map = attack_feature_map.clone().detach()
                temp_feature_map[:, :, row, col] = candidate_unit
                output = model.fc(temp_feature_map.mean(dim=(2, 3)))
                true_class_prob = F.softmax(output, dim=1)[0, true_class].item()

                if true_class_prob > max_prob_increase:
                    max_prob_increase = true_class_prob
                    best_replacement_unit = candidate_unit

            if best_replacement_unit is not None:
                best_attack_feature_map[:, :, row, col] = best_replacement_unit
                attack_feature_map = best_attack_feature_map.clone().detach()


            if torch.isnan(attack_feature_map).any() or torch.isinf(attack_feature_map).any():
                print(f"NaN or Inf detected in attack_feature_map at step ({row}, {col})!")
                attack_feature_map = torch.clamp(attack_feature_map, min=-1e6, max=1e6)


            output = model.fc(attack_feature_map.mean(dim=(2, 3)))
            predicted_class = output.argmax(dim=1).item()
            true_class_prob = F.softmax(output, dim=1)[0, true_class].item()
            print(f"Step ({row}, {col}): True Class Probability = {true_class_prob:.4f}")

            if predicted_class == true_class:
                print(f"Attack Successful! Correct Class {true_class} achieved.")
                return attack_feature_map

        iteration_count += 1

    print(f"Reached maximum iterations ({max_iterations}) without successful attack.")
    return attack_feature_map



def extract_all_feature_units_from_category(model, img_dir):
    global all_feature_units  # 确保 all_feature_units 是全局变量
    all_feature_units = []

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # 执行前向传播，触发钩子函数
            _ = model(img_tensor)

            if features_before is not None and len(features_before.shape) == 4:
                # 确保 features_before 是 4 维张量 (batch_size, channels, height, width)
                # 获取特征图的宽和高
                height, width = features_before.size(2), features_before.size(3)

                # 提取所有特征单元，保留在 GPU 上
                for i in range(height):
                    for j in range(width):
                        feature_unit = features_before[:, :, i, j].view(1, -1)  # 保持在 GPU 上
                        all_feature_units.append(feature_unit)
            else:
                print("Error: Unexpected shape for features_before")

    # 将所有特征单元拼接为一个张量，保持在 GPU 上
    if all_feature_units:
        all_feature_units = torch.cat(all_feature_units, dim=0)
def show_cam_on_image(img, mask, title):
    img_resized = resize(img, (224, 224))
    result = overlay_mask(img_resized, to_pil_image(mask, mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compute_shapley_values(features, model, img_tensor, target_class, sigma=0.8):
    device = features.device
    _, c, h, w = features.size()
    shapley_matrix = torch.zeros(h, w).to(device)


    output = model(img_tensor)
    true_class_prob_before = F.softmax(output, dim=1)[0, target_class].item()

    X = torch.arange(0, w, device=device).view(1, -1).repeat(h, 1)
    Y = torch.arange(0, h, device=device).view(-1, 1).repeat(1, w)
    G = torch.zeros(h, w, h, w).to(device)  # Shape: (h, w, h, w)

    for i in range(h):
        for j in range(w):
            G[i, j] = torch.exp(-((X - j) ** 2 + (Y - i) ** 2) / (2 * sigma ** 2))
    G_tilde = 1 - G
    for i in range(h):
        for j in range(w):
            masked_features = features * G_tilde[i, j]
            output = model.fc(masked_features.mean(dim=(2, 3)))
            true_class_prob_after = F.softmax(output, dim=1)[0, target_class].item()
            shapley_matrix[i, j] = true_class_prob_before - true_class_prob_after
    shapley_matrix_relu = F.relu(shapley_matrix)
    shapley_matrix_normalized = normalize_tensor(shapley_matrix_relu)

    return shapley_matrix_normalized

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def predict_and_visualize_combined(model, category_dir, cam_extractor):
    incorrect_images = []
    correct_images = []

    true_class = int(os.path.basename(category_dir).split('.')[0]) - 1


    for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        output = model(img_tensor)
        predicted_class = output.argmax(dim=1).item()

        if predicted_class != true_class and not incorrect_images:
            incorrect_images.append((img, img_tensor, predicted_class))
        elif predicted_class == true_class and not correct_images:
            correct_images.append((img, img_tensor, true_class))

        if incorrect_images and correct_images:
            break
    def normalize_attacked_feature_map(feature_map):
        min_val = feature_map.min()
        max_val = feature_map.max()

        if max_val - min_val > 1e-6:  # 防止极小差值
            normalized_feature_map = (feature_map - min_val) / (max_val - min_val)
        else:
            normalized_feature_map = feature_map.clone()  # 如果 max_val == min_val，则不做任何操作

        return normalized_feature_map
    def visualize_samples_with_metrics(image_data, title_prefix, is_incorrect=True):
        if image_data is not None:
            img, img_tensor_original, target_class = image_data
            img_tensor = img_tensor_original.clone().detach()

            print(f"{title_prefix}: True Class = {true_class}, Predicted Class = {target_class}")


            _ = model(img_tensor)
            if features_before is None:
                print("Error: features_before is None.")
                return
            saved_feature_map = features_before.clone().detach()

            # 1. GradCAM for original prediction (Before attack)
            activation_map_true_before = cam_extractor(true_class, model(img_tensor))
            combined_map_before = activation_map_true_before[0]
            if title_prefix == "Correct Prediction":
                show_cam_on_image(img, combined_map_before.cpu(), f"{title_prefix} (GradCAM Before Attack)")

            # ------------------------ 1. PGD attack ------------------------
            if is_incorrect:
                activation_map_error_before = cam_extractor(target_class, model(img_tensor))
                combined_map_error_before = activation_map_error_before[0]
                show_cam_on_image(img, combined_map_error_before.cpu(), f"{title_prefix} (GradCAM Before Any Attack)")
                #pgd attack
                labels = torch.tensor([true_class]).to(device)
                attacked_images_pgd = pgd_attack(model, img_tensor.clone(), labels)
                adv_activation_map_pgd = cam_extractor(true_class, model(attacked_images_pgd))
                adv_activation_map_tensor_pgd = adv_activation_map_pgd[0]
                show_cam_on_image(img, adv_activation_map_tensor_pgd.cpu(), f"{title_prefix} (GradCAM After PGD Attack)")
                # diff after pgd attack
                difference_map_pgd = adv_activation_map_tensor_pgd - combined_map_error_before
                difference_map_pgdqh = combined_map_error_before - adv_activation_map_tensor_pgd
                difference_map_pgd_relu = F.relu(difference_map_pgd)
                difference_map_pgd_reluqh = F.relu(difference_map_pgdqh)
                difference_map_normalized_pgd = normalize_tensor(difference_map_pgd_relu)
                difference_map_normalized_pgdqh = normalize_tensor(difference_map_pgd_reluqh)

                #
                show_cam_on_image(img, difference_map_normalized_pgd.cpu(),
                                  f"{title_prefix} (Difference Map After PGD Attack)")
                show_cam_on_image(img, difference_map_normalized_pgdqh.cpu(),
                                  f"{title_prefix} (Difference Map After PGD qh)")
                # ------------------------ 2. reset img_tensor ------------------------
                img_tensor = img_tensor_original.clone().detach()

                # ------------------------ 3. Feature unit rep ------------------------

                attacked_feature_map_fr = feature_replacement_attack(all_feature_units, saved_feature_map, img_tensor,
                                                                     true_class)

                # GradCAM for Feature unit rep
                adv_activation_map_fr = cam_extractor(true_class, model(img_tensor))
                adv_activation_map_tensor_fr = adv_activation_map_fr[0]
                attacked_feature_map_normalized = normalize_attacked_feature_map(attacked_feature_map_fr)
                shapley_map_after = compute_shapley_values(attacked_feature_map_normalized, model, img_tensor,
                                                           true_class)
                combined_map_after = adv_activation_map_fr[0] * shapley_map_after
                combined_map_normalized_after = normalize_tensor(combined_map_after)
                show_cam_on_image(img, combined_map_normalized_after.cpu(),
                                  f"{title_prefix} (shap After Feature Replacement Attack)")

                # diff after attack

                shapley_map_before = compute_shapley_values(features_before, model, img_tensor, true_class)
                combined_map_error_before = shapley_map_before * activation_map_error_before[0]
                combined_map_normalized_before = normalize_tensor(combined_map_error_before)

                difference_map_fr = adv_activation_map_tensor_fr - combined_map_normalized_before
                difference_map_frqh = combined_map_normalized_before - adv_activation_map_tensor_fr

                difference_map_fr_relu = F.relu(difference_map_fr)
                difference_map_fr_reluqh = F.relu(difference_map_frqh)
                difference_map_normalized_fr = normalize_tensor(difference_map_fr_relu)
                difference_map_normalized_frqh = normalize_tensor(difference_map_fr_reluqh)


                show_cam_on_image(img, difference_map_normalized_fr.cpu(),
                                  f"{title_prefix} (Difference shapMap After Feature Replacement Attack)")
                show_cam_on_image(img, difference_map_normalized_frqh.cpu(),
                                  f"{title_prefix} (Difference shapMap After Feature Replacement qh)")
            else:
                print(f"No {'incorrect' if is_incorrect else 'correct'} predictions found for visualization.")

    if correct_images:
        visualize_samples_with_metrics(correct_images[0], "Correct Prediction", is_incorrect=False)


    if incorrect_images:
        visualize_samples_with_metrics(incorrect_images[0], "Incorrect Prediction", is_incorrect=True)


def extract_all_feature_units_from_all_categories(model, total_folder_path):
    for category_folder in os.listdir(total_folder_path):
        category_dir = os.path.join(total_folder_path, category_folder)
        if os.path.isdir(category_dir):
            extract_all_feature_units_from_category(model, category_dir)


def process_single_category(model, category_dir, cam_extractor):
    predict_and_visualize_combined(model, category_dir, cam_extractor)


cam_extractor = GradCAM(model, target_layer='layer4.2')


total_folder = 'cub200/images'
extract_all_feature_units_from_all_categories(model, total_folder)

category_folder = 'cub200/images/073.Blue_Jay'
process_single_category(model, category_folder, cam_extractor)


del cam_extractor