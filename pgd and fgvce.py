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
import random
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import interpolate

import numpy as np
import re

def create_class_index_mapping(total_folder_path):
    class_index_mapping = {}
    folder_names = sorted(os.listdir(total_folder_path), key=lambda x: int(re.search(r'\d+', x).group()))
    for idx, folder_name in enumerate(folder_names):
        if os.path.isdir(os.path.join(total_folder_path, folder_name)):
            class_index_mapping[folder_name] = idx
    return class_index_mapping

# Function to extract class indices
def get_class_index(folder_name, class_index_mapping):
    if folder_name in class_index_mapping:
        return class_index_mapping[folder_name]
    else:
        raise ValueError(f"Folder name '{folder_name}' not found in class index mapping")


# Use GPU for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def apply_blur(image, radius=5):
    """
    Apply blur to an image using Pillow.
    :param image: Input PIL image
    :param radius: Blur radius, default is 5
    :return: Blurred PIL image
    """
    return image.filter(ImageFilter.GaussianBlur(radius))
# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 120)
model.load_state_dict(torch.load('./dogresnet50_model.pth'))
model = model.to(device)
model.eval()
# Extract feature map before attack
features_before = None  # Global variable
def hook_before(module, input, output):
    global features_before
    features_before = output.clone().detach().to(device)

# Extract feature map after attack
features_after = None  # Global variable
def hook_after(module, input, output):
    global features_after
    features_after = output.clone().detach().to(device)

# Register hook to the last convolution layer of ResNet
model.layer4[2].register_forward_hook(hook_before)  # Register hook to the last block in layer4




# Interpolation helper
def interpolate_to_224(activation_map):
    # activation_map is originally a 1x7x7 tensor; keep channel dimension and interpolate to 1x224x224
    return F.interpolate(activation_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)



def pgd_attack(model, images, labels, eps=0.1, alpha=0.05, iters=500):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device).long()  # Ensure labels is a 1D long tensor
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

        # Check whether the class has changed
        if outputs.max(1)[1].item() != labels.item():
            break

    return images

# def feature_replacement_attack(all_feature_units, attack_feature_map, img_tensor, true_class, max_iterations=500):
#     # Use clone to ensure data independence
#     attack_feature_map = attack_feature_map.clone().detach()
#     best_attack_feature_map = attack_feature_map.clone().detach()
#     rows, cols = attack_feature_map.size(2), attack_feature_map.size(3)
#
#     # Track replacement count to prevent infinite loops
#     iteration_count = 0
#
#     while iteration_count < max_iterations:
#         for row in range(rows):
#             for col in range(cols):
#                 max_prob_increase = -float('inf')
#                 best_replacement_unit = None
#
#                 # Clone attack_feature_map for each iteration
#                 current_unit = attack_feature_map[:, :, row, col].view(1, -1).cpu().numpy()
#                 similarities = cosine_similarity(current_unit, all_feature_units)
#                 top_50_candidates = np.argsort(similarities[0])[-50:]
#
#                 for idx in top_50_candidates:
#                     candidate_unit = torch.tensor(all_feature_units[idx]).view(1, 2048).to(device)
#
#                     # Temp feature map to ensure no overwriting
#                     temp_feature_map = attack_feature_map.clone().detach()
#                     temp_feature_map[:, :, row, col] = candidate_unit.view(1, 2048)
#
#                     # Compute class probability after replacement
#                     output = model.fc(temp_feature_map.mean(dim=(2, 3)))
#                     true_class_prob = F.softmax(output, dim=1)[0, true_class].item()
#
#                     if true_class_prob > max_prob_increase:
#                         max_prob_increase = true_class_prob
#                         best_replacement_unit = candidate_unit
#
#                 if best_replacement_unit is not None:
#                     best_attack_feature_map[:, :, row, col] = best_replacement_unit.view(1, 2048)
#                     attack_feature_map = best_attack_feature_map.clone().detach()
#
#                 # Check generated feature map to avoid invalid values
#                 if torch.isnan(attack_feature_map).any() or torch.isinf(attack_feature_map).any():
#                     print(f"NaN or Inf detected in attack_feature_map at step ({row}, {col})!")
#                     attack_feature_map = torch.clamp(attack_feature_map, min=-1e6, max=1e6)  # Clamp value range
#
#                 # Check whether the current feature map has been successfully converted
#                 output = model.fc(attack_feature_map.mean(dim=(2, 3)))
#                 predicted_class = output.argmax(dim=1).item()
#                 true_class_prob = F.softmax(output, dim=1)[0, true_class].item()
#                 print(f"Step {row * cols + col + 1}: True Class Probability = {true_class_prob:.4f}")
#
#                 if predicted_class == true_class:
#                     print(f"Attack Successful! Correct Class {true_class} achieved.")
#                     return attack_feature_map  # Class conversion succeeded; return feature map
#
#         iteration_count += 1  # Increment iteration count after one full round
#
#     print(f"Reached maximum iterations ({max_iterations}) without successful attack.")
#     return attack_feature_map  # Reached max iterations; return current feature map

def feature_replacement_attack(all_feature_units, attack_feature_map, img_tensor, true_class, max_iterations=500):
    # Use clone to ensure data independence
    attack_feature_map = attack_feature_map.clone().detach()
    best_attack_feature_map = attack_feature_map.clone().detach()
    rows, cols = attack_feature_map.size(2), attack_feature_map.size(3)

    # Track replacement count to prevent infinite loops
    iteration_count = 0

    # Initialize Gaussian distribution parameters
    mu_row, mu_col = rows / 2, cols / 2  # Set mean to center position
    initial_sigma = min(rows, cols) / 6  # Small initial std to focus on center region
    sigma_increment = initial_sigma / 5  # Increase std each round to gradually expand region

    while iteration_count < max_iterations:
        # Current attack std increases with iteration count
        current_sigma = initial_sigma + sigma_increment * iteration_count

        # Generate attack positions following current Gaussian distribution
        gaussian_indices = []
        for _ in range(49):  # Select 49 positions per attack round
            row = int(np.clip(np.random.normal(mu_row, current_sigma), 0, rows - 1))
            col = int(np.clip(np.random.normal(mu_col, current_sigma), 0, cols - 1))
            gaussian_indices.append((row, col))

        for row, col in gaussian_indices:
            max_prob_increase = -float('inf')
            best_replacement_unit = None

            # Clone attack_feature_map for each iteration
            # current_unit = attack_feature_map[:, :, row, col].view(1, -1).cpu().numpy()
            # similarities = cosine_similarity(current_unit, all_feature_units)
            # top_50_candidates = np.argsort(similarities[0])[-50:]
            # Clone attack_feature_map for each iteration
            current_unit = attack_feature_map[:, :, row, col].view(1, -1)  # Keep on GPU
            candidate_units = all_feature_units  # all_feature_units is already a GPU tensor
            similarities = F.cosine_similarity(current_unit, candidate_units)

            # Use torch.topk instead of np.argsort to keep operations on GPU
            top_50_candidates = torch.topk(similarities, 50, largest=True).indices


            for idx in top_50_candidates:
                # candidate_unit = torch.tensor(all_feature_units[idx]).view(1, 2048).to(device)
                candidate_unit = candidate_units[idx].view(1, 2048)
                # Temp feature map to ensure no overwriting
                temp_feature_map = attack_feature_map.clone().detach()
                # temp_feature_map[:, :, row, col] = candidate_unit.view(1, 2048)
                temp_feature_map[:, :, row, col] = candidate_unit

                # Compute class probability after replacement
                output = model.fc(temp_feature_map.mean(dim=(2, 3)))
                true_class_prob = F.softmax(output, dim=1)[0, true_class].item()

                if true_class_prob > max_prob_increase:
                    max_prob_increase = true_class_prob
                    best_replacement_unit = candidate_unit

            if best_replacement_unit is not None:
                # best_attack_feature_map[:, :, row, col] = best_replacement_unit.view(1, 2048)
                best_attack_feature_map[:, :, row, col] = best_replacement_unit
                attack_feature_map = best_attack_feature_map.clone().detach()

            # Check generated feature map to avoid invalid values
            if torch.isnan(attack_feature_map).any() or torch.isinf(attack_feature_map).any():
                print(f"NaN or Inf detected in attack_feature_map at step ({row}, {col})!")
                attack_feature_map = torch.clamp(attack_feature_map, min=-1e6, max=1e6)  # Clamp value range

            # Check whether the current feature map has been successfully converted
            output = model.fc(attack_feature_map.mean(dim=(2, 3)))
            predicted_class = output.argmax(dim=1).item()
            true_class_prob = F.softmax(output, dim=1)[0, true_class].item()
            print(f"Step ({row}, {col}): True Class Probability = {true_class_prob:.4f}")

            if predicted_class == true_class:
                print(f"Attack Successful! Correct Class {true_class} achieved.")
                return attack_feature_map  # Class conversion succeeded; return feature map

        iteration_count += 1  # Increment iteration count after one full round

    print(f"Reached maximum iterations ({max_iterations}) without successful attack.")
    return attack_feature_map  # Reached max iterations; return current feature map


# Compute Mean Activation Width

# all_feature_units = []  # Store feature units from all categories

# Extract all feature map units
# def extract_all_feature_units_from_category(model, img_dir):
#     global all_feature_units  # Ensure all_feature_units is a global variable
#     for img_name in os.listdir(img_dir):
#         img_path = os.path.join(img_dir, img_name)
#
#         if os.path.isfile(img_path):
#             img = Image.open(img_path).convert('RGB')
#             img_tensor = transform(img).unsqueeze(0).to(device)
#
#             # Run forward pass to trigger hook
#             _ = model(img_tensor)
#
#             if features_before is not None and len(features_before.shape) == 4:
#                 # Ensure features_before is a 4D tensor (batch_size, channels, height, width)
#                 for i in range(features_before.size(2)):
#                     for j in range(features_before.size(3)):
#                         feature_unit = features_before[:, :, i, j].view(2048).detach().cpu().numpy()
#                         all_feature_units.append(feature_unit)
#             else:
#                 print("Error: Unexpected shape for features_before")


# Function to display activation map

def extract_all_feature_units_from_category(model, img_dir):
    global all_feature_units  # Ensure all_feature_units is a global variable
    all_feature_units = []

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Run forward pass to trigger hook
            _ = model(img_tensor)

            if features_before is not None and len(features_before.shape) == 4:
                # Ensure features_before is a 4D tensor (batch_size, channels, height, width)
                # Get feature map height and width
                height, width = features_before.size(2), features_before.size(3)

                # Extract all feature units and keep them on GPU
                for i in range(height):
                    for j in range(width):
                        feature_unit = features_before[:, :, i, j].view(1, -1)  # Keep on GPU
                        all_feature_units.append(feature_unit)
            else:
                print("Error: Unexpected shape for features_before")

    # Concatenate all feature units into a single tensor on GPU
    if all_feature_units:
        all_feature_units = torch.cat(all_feature_units, dim=0)
def show_cam_on_image(img, mask, title):
    img_resized = resize(img, (224, 224))
    result = overlay_mask(img_resized, to_pil_image(mask, mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Other code remains unchanged

# Compute Shapley values
def compute_shapley_values(features, model, img_tensor, target_class, sigma=0.8):
    _, c, h, w = features.size()
    shapley_matrix = torch.zeros(h, w).to(device)

    output = model(img_tensor)
    true_class_prob_before = F.softmax(output, dim=1)[0, target_class].item()

    # Generate coordinate grids for Gaussian neighborhood masking.
    X = torch.arange(0, w, device=device).view(1, -1).repeat(h, 1)
    Y = torch.arange(0, h, device=device).view(-1, 1).repeat(1, w)

    # Build Gaussian kernels centered at each spatial position.
    G = torch.zeros(h, w, h, w).to(device)
    for i in range(h):
        for j in range(w):
            G[i, j] = torch.exp(-((X - j) ** 2 + (Y - i) ** 2) / (2 * sigma ** 2))

    # Inverted Gaussian masks suppress the neighborhood around each point.
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

# Normalization function
def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
def deletion_test(model, img_tensor, gradcam_map, target_class, step=0.03):
    # Bilinearly interpolate gradcam_map to 1x224x224
    gradcam_map = interpolate(gradcam_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()

    # Only keep regions where activation is greater than 0.5
    valid_indices = torch.nonzero(gradcam_map > 0.5, as_tuple=False).cpu().numpy()

    total_pixels = len(valid_indices)
    pixels_per_step = max(1, int(total_pixels * step))

    # Ensure the original img_tensor is not modified
    current_img = img_tensor.clone()

    softmax_scores = []
    for i in range(0, total_pixels, pixels_per_step):
        step_indices = valid_indices[i:i + pixels_per_step]
        for idx in step_indices:
            h, w = idx[-2], idx[-1]  # Get spatial location h, w
            current_img[0, :, h, w] = 0  # Set all channels at spatial location (h, w) to 0

        # Make predictions with the modified image
            x = model(current_img)  # Forward pass through the model
            output = x  # Get the output from the model  # Fully connected layer for classification
            softmax_scores.append(F.softmax(output, dim=1)[0, target_class].item())

    # Calculate the average score
    if softmax_scores:
        average_score = sum(softmax_scores) / len(softmax_scores)
    else:
        average_score = 0

    return {"deletion_avg_score": average_score}

def insertion_test(model, img_tensor, gradcam_map, target_class, step=0.03):
    # Bilinearly interpolate gradcam_map to 1x224x224
    gradcam_map = interpolate(gradcam_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()

    # Only keep regions where activation is greater than 0.5
    valid_indices = torch.nonzero(gradcam_map > 0.5, as_tuple=False).cpu().numpy()

    total_pixels = len(valid_indices)
    pixels_per_step = max(1, int(total_pixels * step))  # Ensure at least one pixel is inserted per step

    # Start with a blurred version of the image (using a zero tensor to simulate blur)
    blurred_img = torch.zeros_like(img_tensor)
    current_img = blurred_img.clone()

    softmax_scores = []
    for i in range(0, total_pixels, pixels_per_step):
        step_indices = valid_indices[i:i + pixels_per_step]
        for idx in step_indices:
            h, w = idx[-2], idx[-1]  # Get spatial location h, w
            current_img[0, :, h, w] = img_tensor[0, :, h, w]  # Restore all channels at spatial location (h, w)

        # Make predictions with the modified image
            x = model(current_img)  # Forward pass through the model
            output = x  # Get the output from the model  # Fully connected layer for classification
            softmax_scores.append(F.softmax(output, dim=1)[0, target_class].item())


    # Calculate the average score
    if softmax_scores:
        average_score = sum(softmax_scores) / len(softmax_scores)
    else:
        average_score = 0

    return {"insertion_avg_score": average_score}

def predict_and_visualize_combined(model, category_dir, cam_extractor, class_index_mapping):
    incorrect_images = []
    correct_images = []

    category_name = os.path.basename(category_dir)
    true_class = get_class_index(category_name, class_index_mapping)

    # Traverse images and collect incorrect/correct predictions
    for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        output = model(img_tensor)
        predicted_class = output.argmax(dim=1).item()

        if predicted_class != true_class and not incorrect_images:  # Keep only the first incorrect prediction
            incorrect_images.append((img, img_tensor, predicted_class))
        elif predicted_class == true_class and not correct_images:  # Keep only the first correct prediction
            correct_images.append((img, img_tensor, true_class))

        if incorrect_images and correct_images:  # Exit once one correct and one incorrect sample are found
            break




    def normalize_attacked_feature_map(feature_map):
        min_val = feature_map.min()
        max_val = feature_map.max()

        if max_val - min_val > 1e-6:  # Prevent extremely small denominator
            normalized_feature_map = (feature_map - min_val) / (max_val - min_val)
        else:
            normalized_feature_map = feature_map.clone()  # If max_val == min_val, keep unchanged

        return normalized_feature_map
    def visualize_samples_with_metrics(image_data, title_prefix, is_incorrect=True):
        if image_data is not None:
            img, img_tensor_original, target_class = image_data
            img_tensor = img_tensor_original.clone().detach()  # Ensure img_tensor is independent

            print(f"{title_prefix}: True Class = {true_class}, Predicted Class = {target_class}")

            # Use hook to get the original feature map
            _ = model(img_tensor)  # Ensure forward pass triggers the hook
            if features_before is None:
                print("Error: features_before is None.")
                return
            saved_feature_map = features_before.clone().detach()

            # 1. GradCAM for original prediction (Before attack)
            activation_map_true_before = cam_extractor(true_class, model(img_tensor))
            combined_map_before = activation_map_true_before[0]

            # Visualize GradCAM activation map (before attack)
            if title_prefix == "Correct Prediction":
                show_cam_on_image(img, combined_map_before.cpu(), f"{title_prefix} (GradCAM Before Attack)")

            # ------------------------ 1. PGD attack ------------------------
            if is_incorrect:
                activation_map_error_before = cam_extractor(target_class, model(img_tensor))
                combined_map_error_before = activation_map_error_before[0]

                # Visualize original activation map for incorrect prediction (no attack)
                show_cam_on_image(img, combined_map_error_before.cpu(), f"{title_prefix} (GradCAM Before Any Attack)")
                # Perform PGD attack
                labels = torch.tensor([true_class]).to(device)
                attacked_images_pgd = pgd_attack(model, img_tensor.clone(), labels)

                # GradCAM for PGD attack
                adv_activation_map_pgd = cam_extractor(true_class, model(attacked_images_pgd))
                adv_activation_map_tensor_pgd = adv_activation_map_pgd[0]

                delete_result = deletion_test(model, img_tensor, adv_activation_map_tensor_pgd, true_class, step=1)
                insertion_result = insertion_test(model, img_tensor, adv_activation_map_tensor_pgd, true_class,
                                                  step=1)
                print(f"pgd Deletion Average Score: {delete_result['deletion_avg_score']}")
                print(f"pgd Insertion Average Score: {insertion_result['insertion_avg_score']}")

                # Visualize activation map after PGD attack
                show_cam_on_image(img, adv_activation_map_tensor_pgd.cpu(), f"{title_prefix} (GradCAM After PGD Attack)")

                # Compute difference activation map after PGD attack
                difference_map_pgd = adv_activation_map_tensor_pgd - combined_map_error_before
                difference_map_pgd_relu = F.relu(difference_map_pgd)
                difference_map_normalized_pgd = normalize_tensor(difference_map_pgd_relu)

                # Visualize difference activation map after PGD attack
                show_cam_on_image(img, difference_map_normalized_pgd.cpu(), f"{title_prefix} (Difference Map After PGD Attack)")

                # ------------------------ 2. Reset img_tensor ------------------------
                img_tensor = img_tensor_original.clone().detach()  # Reinitialize img_tensor

                # ------------------------ 3. Feature replacement attack ------------------------
                # Feature replacement attack
                attacked_feature_map_fr = feature_replacement_attack(all_feature_units, saved_feature_map, img_tensor, true_class)

                # GradCAM for feature replacement attack
                adv_activation_map_fr = cam_extractor(true_class, model(img_tensor))
                adv_activation_map_tensor_fr = adv_activation_map_fr[0]
                attacked_feature_map_normalized = normalize_attacked_feature_map(attacked_feature_map_fr)
                delete_result = deletion_test(model, img_tensor, adv_activation_map_tensor_fr, true_class, step=1)
                insertion_result = insertion_test(model, img_tensor,  adv_activation_map_tensor_fr, true_class,
                                                  step=1)
                print(f"Deletion Average Score: {delete_result['deletion_avg_score']}")
                print(f"Insertion Average Score: {insertion_result['insertion_avg_score']}")

                shapley_map_after = compute_shapley_values(attacked_feature_map_normalized, model, img_tensor,
                                                           true_class)
                combined_map_after = adv_activation_map_fr[0] * shapley_map_after
                combined_map_normalized_after = normalize_tensor(combined_map_after)
                show_cam_on_image(img, combined_map_normalized_after.cpu(), f"{title_prefix} (shap After Feature Replacement Attack)")

                # Compute difference activation map after feature replacement attack

                shapley_map_before = compute_shapley_values(features_before, model, img_tensor,true_class)
                combined_map_error_before = shapley_map_before * activation_map_error_before[0]
                combined_map_normalized_before = normalize_tensor(combined_map_error_before)
                difference_map_fr =adv_activation_map_tensor_fr- combined_map_normalized_before
                difference_map_fr_relu = F.relu(difference_map_fr)
                difference_map_normalized_fr = normalize_tensor(difference_map_fr_relu)

                # Visualize difference activation map after feature replacement attack
                show_cam_on_image(img, difference_map_normalized_fr.cpu(), f"{title_prefix} (Difference shapMap After Feature Replacement Attack)")

        else:
            print(f"No {'incorrect' if is_incorrect else 'correct'} predictions found for visualization.")

    # Visualize correctly predicted image (only one result)
    if correct_images:
        visualize_samples_with_metrics(correct_images[0], "Correct Prediction", is_incorrect=False)

    # Visualize first incorrectly predicted image; run PGD and feature replacement attacks
    if incorrect_images:
        visualize_samples_with_metrics(incorrect_images[0], "Incorrect Prediction", is_incorrect=True)




# Extract feature map units from all categories
def extract_all_feature_units_from_all_categories(model, total_folder_path):
    for category_folder in os.listdir(total_folder_path):
        category_dir = os.path.join(total_folder_path, category_folder)
        if os.path.isdir(category_dir):
            extract_all_feature_units_from_category(model, category_dir)

# Process a single category
def process_single_category(model, category_dir, cam_extractor, class_index_mapping):
    predict_and_visualize_combined(model, category_dir, cam_extractor, class_index_mapping)

# Use GradCAM
cam_extractor = GradCAM(model, target_layer='layer4.2')


# Use GradCAM
total_folder = '/workspace/stdex/Images'
class_index_mapping = create_class_index_mapping(total_folder)
extract_all_feature_units_from_all_categories(model, total_folder)

category_folder = '/workspace/stdex/Images/n02096585-Boston_bull'
process_single_category(model, category_folder, cam_extractor, class_index_mapping)

# Manual cleanup at the end
del cam_extractor