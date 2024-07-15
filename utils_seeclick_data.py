import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import random
import action_type
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2Model

# BLIP2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

def extract_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    # image = image.resize((224, 224))
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        image_features = model.get_image_features(**inputs).pooler_output[0]
        image_features = image_features.detach().cpu()
    return image_features

def load_seeclick_data(split, data_dir):
    with open("/data/zzs800-0/wangyt/seeclick_web.json", 'r') as file:
        all_data = json.load(file)
    random.shuffle(all_data)
    split_index = int(len(all_data) * 0.8) 
    val_index = int(len(all_data) * 0.9)
 
    if split == 'train':
        data = all_data[:split_index]
    elif split == 'val':
        data = all_data[split_index:val_index]
    else:
        data = all_data[val_index:]

    processed_data = []
    for item in data:
        img_path = os.path.join(data_dir, item['img_filename'])
        image_features = extract_image_features(img_path)
        
        for element in item['elements']:
            bbox_center = [(element['bbox'][0] + element['bbox'][2]) / 2, (element['bbox'][1] + element['bbox'][3]) / 2]
            question = f"Goal: Click on the '{element['instruction']}' element."
            processed_data.append({
                'image': image_features,
                'instruction': question,
                'bbox': element['bbox'],
                'action_type': action_type.ActionType.DUAL_POINT.value,
                'touch_point': bbox_center,
                'lift_point': bbox_center
            })

    return processed_data


class SeeClickDatasetImg(Dataset):
    def __init__(self, data, tokenizer, source_len, target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_ids = torch.tensor(item['image']).squeeze()
        instruction = item['instruction']
        instruction = " ".join(instruction.split())

        # Tokenize text
        source_encoded = self.tokenizer.batch_encode_plus(
            [instruction],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        action_label = f'"action_type": "{item["action_type"]}", "touch_point": "{item["touch_point"]}", "lift_point": "{item["lift_point"]}"'
        target_encoded = self.tokenizer.batch_encode_plus(
            [action_label],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoded['input_ids'].squeeze(),
            "attention_mask": source_encoded['attention_mask'].squeeze(),
            "image_ids": image_ids,
            "labels": target_encoded['input_ids'].squeeze(),
            "target_act": torch.tensor(item['action_type']).squeeze(),
            "target_touch": torch.tensor(item['touch_point']).squeeze(),
            "target_lift": torch.tensor(item['lift_point']).squeeze()
        }
