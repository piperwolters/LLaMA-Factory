import os
import io
import json
import base64
from PIL import Image
from openai import OpenAI

from metric import compute_stepwise_accuracy
from vis import save_html


api_key = ''
client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

# Load in a dataset json and format messages for the model api.
json_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_ac_val_LL.json')
data = json.load(json_file)

results = []
outputs, targets = [], []
target_bb_centers, target_bb_smallests = [], []
for i,dp in enumerate(data):
    print("Processing...", i)

    messages = dp['messages']
    metadata = dp['metadata']

    image_path = dp['images'][0]
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((448, 448))  # Resize to a smaller resolution
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_data = image_buffer.getvalue()
    image_data = base64.b64encode(image_data).decode('utf-8')
    #image_file = open(image_path, 'rb')
    #image_data = image_file.read()

    dp_idx = str(metadata['dp_idx'])
    datapath = '/data/piperw/data/osagent/unified/android_control_III/val/' + dp_idx + '/'
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

    # Remove the target action from the prompt during inference.
    #messages = [m for m in messages if not (m.get("role") == "assistant")]
    messages = [{"role": "user", "content": image_data, " type": "image/png"}]
    #messages.append({"role": "user", "content": image_data, " type": "image/png"})

    chat_response = client.chat.completions.create(
        #model="llava-hf/llava-1.5-7b-hf",
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=20
    )

    output = chat_response.choices[0].message.content
    outputs.append(output)

    result = {
        'target_action': target,
        'target_bb_center': target_bb_center,
        'target_bb_smallest': target_bb_smallest,
        'model_output': output,
        'a11y': reduced_a11y, 
        'screenshot': screenshot,
        'instruction': instruction,
        'dims': dim
    }
    results.append(result)
  
    if len(results) == 100:
        break

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_centers)
print("Step-wise accuracy using centers:", accuracy)

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_smallests)
print("Step-wise accuracy using smallest:", accuracy)

for r,result in enumerate(results):
    result['metric'] = corrects[r]

save_html(results, 'aug27_gpt4omini_val100_justtext.html')
