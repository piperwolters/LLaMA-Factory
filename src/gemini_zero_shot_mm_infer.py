import os
import io
import json
import base64
import vertexai
from PIL import Image
from openai import OpenAI
from google.auth import default, transport

from metric import compute_stepwise_accuracy
from vis import save_html


# TODO(developer): Update and un-comment below lines
project_id = "ai2-vision"
location = "us-west1"

vertexai.init(project=project_id, location=location)

# Programmatically get an access token
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

# OpenAI Client
client = OpenAI(
    base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
    api_key=credentials.token,
)


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


# Load in a dataset json and format messages for the model api.
#train_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_train_LL_1000.json')
val_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_val_LL.json')
test_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_test_LL.json')

json_file = val_file
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

    chat_response = client.chat.completions.create(
        model="google/gemini-1.5-pro-001",
        messages=messages,
        temperature=0.0,
        max_tokens=20
    )

    try:
        output = chat_response.choices[0].message.content
    except:
        print("BAD")
        output = ""
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
