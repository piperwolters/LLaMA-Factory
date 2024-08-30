import os
import json
from PIL import Image
from peft import PeftModel


from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, AutoTokenizer, LlavaImageProcessor

model = LlavaForConditionalGeneration.from_pretrained('/data/piperw/projects/LLaMA-Factory/saves/aug28_llava_mm_ac_train_LL-1000/checkpoint-100')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/piperw/projects/LLaMA-Factory/saves/aug28_llava_mm_ac_train_LL-1000/checkpoint-100')

image_processor = LlavaImageProcessor(
    image_size=224,  # Example size, adjust to your model's requirement
    mean=[0.485, 0.456, 0.406],  # Example normalization mean values
    std=[0.229, 0.224, 0.225],   # Example normalization std values
    resample=2,  # Example resampling, adjust as needed
)

# Combine into a processor
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)


def apply_chat_template(user_input):
    vicuna_template = f"""USER: {user_input}\nASSISTANT:"""
    return vicuna_template


## Data
train_file = open('/data/piperw/projects/LLaMA-Factory/data/ac_train_LL-1000.json')
val_file = open('/data/piperw/projects/LLaMA-Factory/data/ac_val_LL.json')
val_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_ac_val_LL.json') 

json_files = [val_file]
splits = ['val']
for j,json_file in enumerate(json_files):
    split = splits[j]
    print("SPLIT:", split)

    data = json.load(json_file)

    results = []
    outputs, targets = [], []
    target_bb_centers, target_bb_smallests = [], []
    for i,dp in enumerate(data):
        print("Processing...", i)

        messages = dp['messages']
        messages = apply_chat_template(messages)

        metadata = dp['metadata']
        images = Image.open(dp['images'][0]) if 'images' in dp else None

        dp_idx = str(metadata['dp_idx'])
        datapath = '/data/piperw/data/osagent/unified/android_control_III/' + split + '/' + dp_idx + '/'
        screenshot = os.path.join(datapath, 'start_state.png')

        dim = metadata['dim']
        reduced_a11y = metadata['a11y']
        target = metadata['action']
        instruction = metadata['instruction']

        target_bb_center = metadata['target_bb_center']
        target_bb_centers.append(target_bb_center)
        target_bb_smallest = metadata['target_bb_smallest']
        target_bb_smallests.append(target_bb_smallest)
        targets.append(target)

        inputs = tokenizer(text=messages, images=images, return_tensors="pt")
        outputs = adapter_model(**inputs)
        predictions = outputs.logits.argmax(-1)
        decoded_output = processor.batch_decode(predictions, skip_special_tokens=True)
        print("output:", decoded_output)

        exit()
