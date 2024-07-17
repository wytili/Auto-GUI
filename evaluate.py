from PIL import Image
from transformers import AutoProcessor, Blip2Model, AutoTokenizer
from model import T5ForMultimodalGeneration
import torch
import json
import ast

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

blip2 = Blip2Model.from_pretrained("/data/zzs800-0/wangyt/blip2-opt-2.7b", torch_dtype=dtype).to(device)
processor = AutoProcessor.from_pretrained("/data/zzs800-0/wangyt/blip2-opt-2.7b")

def extract_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device, dtype)
        image_features = blip2.get_image_features(**inputs).pooler_output[0]
        image_features = image_features.detach().cpu()
    return image_features

# 示例文本和图片
image_width, image_height = 2400, 1636
# prompt = "Add an Apple iPhone 11 to the shopping cart."
# image_features = extract_image_features("./evaluate/output_data_v2_refine_batch/images/popupbox_phone_1b1i/modified_html_1717646565.8464.png")
# prompt = "Create a new account on the login form."
# image_features = extract_image_features("./evaluate/output_data_v2_refine_batch/images/popupbox_phone_1b1i/modified_html_1717647046.547312.png")

# 加载 T5 模型
# model = T5ForMultimodalGeneration.from_pretrained('/data1/models/Auto-UI-Base', img_dim=1408).to(device).half()
# tokenizer = AutoTokenizer.from_pretrained('/data1/models/Auto-UI-Base', trust_remote_code=True)

model = T5ForMultimodalGeneration.from_pretrained('/home/wangyt/Auto-GUI/experiments/seq_future_blip_axis_all0.1_hist8_future4_-data-zzs800-0-wangyt-Auto-UI-Auto-UI-Base_blip_lr0.0001_bs4_ip512_op128_ep10', img_dim=1408).to(device).half()
tokenizer = AutoTokenizer.from_pretrained('/home/wangyt/Auto-GUI/experiments/seq_future_blip_axis_all0.1_hist8_future4_-data-zzs800-0-wangyt-Auto-UI-Auto-UI-Base_blip_lr0.0001_bs4_ip512_op128_ep10', trust_remote_code=True)

# 使用tokenizer将文本转换为token ID
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# image_ids = torch.tensor(image_features).unsqueeze(0).to(device)

# outputs = model.test_step(tokenizer, batch={"input_ids": input_ids, "image_ids": image_ids})

# # 将生成的token ID解码为文本
# print("Generated Text:", outputs)

def point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def replace_extension(filename, new_extension):
    base_name, _ = os.path.splitext(filename)
    return f"{base_name}.{new_extension}"

def process_dataset(dataset_path):
    results = []
    gold = 0
    bad = 0
    with open(dataset_path, 'r') as file:
        for line in file:
            item = json.loads(line)

            prompt = item['goal']
            image_path = os.path.join('./1b1i/images/', item['target'], item['modified_file'].replace('.html', '.png'))
            # image_path = os.path.join('./2b/images/', item['target'], item['modified_file'].replace('.html', '.png'))
            # image_path = os.path.join('./form/images/', item['target'], item['modified_file'].replace('.html', '.png'))
            image_features = extract_image_features(image_path)
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            image_ids = torch.tensor(image_features).unsqueeze(0).to(device)

            output = model.test_step(tokenizer, batch={"input_ids": input_ids, "image_ids": image_ids})
            print(f"output: {output}")
            prediction_string = output['preds'][0]
            print(f"prediction_string: {prediction_string}")
            formatted_string = '{' + prediction_string + '}'
            prediction_dict = json.loads(formatted_string.replace("'", '"'))

            # 提取数据
            touch_point = prediction_dict['touch_point']

            coordinates = ast.literal_eval(touch_point)
            touch_x, touch_y = [float(coord) for coord in coordinates]

            pixel_touch_x = touch_x * 2400
            pixel_touch_y = touch_y * 1636

            is_in_gold = any(point_in_bbox((pixel_touch_x, pixel_touch_y), bbox) for _, bbox in item['label']['gold'])
            is_in_bad = any(point_in_bbox((pixel_touch_x, pixel_touch_y), bbox) for _, bbox in item['label']['bad'])

            if is_in_gold:
                gold += 1
            if is_in_bad:
                bad += 1
            result = {
                'prediction': prediction_dict,
                'is_in_gold': is_in_gold,
                'is_in_bad': is_in_bad
            }

            results.append(result)
            
            print("==============================")
            print(f"is_in_gold: {is_in_gold}")
            print(f"gold: {gold/len(results)}")
            print(f"is_in_bad: {is_in_bad}")
            print(f"bad: {bad/len(results)}")

    return results, gold, bad


dataset_path = './1b1i/output_popupbox_phone_1b1i.jsonl'
# dataset_path = './2b/output_popupbox_phone_2b.jsonl'
# dataset_path = './form/output_popupbox_phone_form.jsonl'

evaluation_results, gold, bad = process_dataset(dataset_path)
print(evaluation_results)
print(f"gold: {gold}")
print(f"bad: {bad}")