import os
import io
import json
import base64
from PIL import Image
from openai import OpenAI

from metric import compute_stepwise_accuracy
from vis import save_html


def add_string_after_instruction(original_string, string_to_add):
    insertion_point = original_string.find("Instruction: \n")
    if insertion_point == -1:
        return original_string
    instruction_end = original_string.find("\n", insertion_point + len("Instruction: \n")) + 1
    updated_string = original_string[:instruction_end] + string_to_add + "\n" + original_string[instruction_end:]
    return updated_string

def remove_screen_description(original_string):
    start_of_section = original_string.find("# Screen Description: \n")
    if start_of_section == -1:
        return original_string
    start_of_instruction = original_string.find("# Instruction", start_of_section)
    if start_of_instruction == -1:
        return original_string
    updated_string = original_string[:start_of_section] + original_string[start_of_instruction:]
    return updated_string


api_key = '' # DashScope API key
client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

# Load in a dataset json and format messages for the model api.
#train_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_train_LL_1000.json')
#val_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_val_LL.json')
test_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_test_LL.json')

json_file = test_file
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

    # Optionally add more content to the text input 
    messages[1]['content'][0]['text'] = add_string_after_instruction(messages[1]['content'][0]['text'], "Please output a single action and nothing else.")
    messages[1]['content'][0]['text'] = remove_screen_description(messages[1]['content'][0]['text'])  # code to remove a11y from input 
    #messages[1]['content'] = [c for c in messages[1]['content'] if not (c.get("type") == "image_url")]  # code to remove image from input

    system_message = [m for m in messages if m.get("role") == "system"][0]

    input_messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': messages[1]['content'][0]['text']
    ]

    chat_response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=input_messages,
        temperature=0.0,
        max_tokens=20
    )

    print("chat response:", chat_response)
    output = chat_response.choices[0].message.content
    outputs.append(output)

    print("output:", output, " & gt:", target)

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
  
    #if len(results) == 100:
    #    break

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_centers)
print("Step-wise accuracy using centers:", accuracy)

accuracy, corrects = compute_stepwise_accuracy(targets, outputs, target_bb_smallests)
print("Step-wise accuracy using smallest:", accuracy)

for r,result in enumerate(results):
    result['metric'] = corrects[r]

#save_html(results, 'aug27_gpt4omini_val100_justtext.html')
