import os
import re
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image, convert_image_dtype
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("amaye15/microsoft-resnet-50-batch32-lr0.0005-standford-dogs")
model = AutoModelForImageClassification.from_pretrained("amaye15/microsoft-resnet-50-batch32-lr0.0005-standford-dogs")
model = model.to(device)
model.eval()

def create_class_index_mapping(total_folder_path):
    class_index_mapping = {}
    folder_names = sorted(os.listdir(total_folder_path), key=lambda x: int(re.search(r'\d+', x).group()))
    for idx, folder_name in enumerate(folder_names):
        if os.path.isdir(os.path.join(total_folder_path, folder_name)):
            class_index_mapping[folder_name] = idx
    return class_index_mapping

def get_class_index(folder_name, class_index_mapping):
    if folder_name in class_index_mapping:
        return class_index_mapping[folder_name]
    else:
        raise ValueError(f"Folder name '{folder_name}' not found in class index mapping")

target_layer = model.resnet.encoder.stages[3].layers[2].layer[2].convolution

total_folder_path = '/workspace/stdex/Images/'

class_index_mapping = create_class_index_mapping(total_folder_path)

correct_count = 0
incorrect_count = 0

target_folder_name = 'n02087394-Rhodesian_ridgeback'
target_folder_path = os.path.join(total_folder_path, target_folder_name)

true_class_index = get_class_index(target_folder_name, class_index_mapping)

for image_name in os.listdir(target_folder_path):
    image_path = os.path.join(target_folder_path, image_name)
    try:
        img = read_image(image_path)

        img = img[:3]
        img = convert_image_dtype(img, dtype=torch.float)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_tensor = normalize(resize(img, (224, 224)), mean=mean, std=std).to(device)

        out = model(input_tensor.unsqueeze(0))

        logits = out.logits+1

        predicted_class = logits.argmax(dim=1).item()

        if predicted_class == true_class_index:
            correct_count += 1
        else:
            incorrect_count += 1

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

print(f"Correctly predicted images: {correct_count}")
print(f"Incorrectly predicted images: {incorrect_count}")
