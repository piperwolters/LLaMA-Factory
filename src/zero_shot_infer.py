import os
import json
from openai import OpenAI

from metric import compute_stepwise_accuracy
from vis import save_html


api_key = ''
client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

# Load in a dataset json and format messages for the model api.
#json_file = open('/data/piperw/projects/LLaMA-Factory/data/ac/ac_val_LL.json')
json_file = open('/data/piperw/projects/LLaMA-Factory/data/ac/ac_zeroshot_val_LL.json')
data = json.load(json_file)

results = []
outputs, targets = [], []
target_bb_centers, target_bb_smallests = [], []
for i,dp in enumerate(data):
    print("Processing...", i)

    messages = dp['messages']
    metadata = dp['metadata']

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
    messages = [m for m in messages if not (m.get("role") == "assistant")]

    chat_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.1,
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
   

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_centers)
print("Step-wise accuracy using centers:", accuracy)

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_smallests)
print("Step-wise accuracy using smallest:", accuracy)

for r,result in enumerate(results):
    result['metric'] = corrects[r]

save_html(results, 'aug23_gpt4_ac_zeroshot_val_LL.html')
