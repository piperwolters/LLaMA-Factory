import os
import json
from openai import OpenAI

from metric import compute_stepwise_accuracy
from vis import save_html


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Load in a dataset json and format messages for the model api.
#train_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_train_LL-1000.json')
#val_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_val_LL.json')
test_file = open('/data/piperw/projects/LLaMA-Factory/data/mm_v2_ac_test_LL.json')

json_files = [test_file] # test_file]
splits = ['test']
for j,json_file in enumerate(json_files):
    split = splits[j]
    print("SPLIT:", split)

    data = json.load(json_file)

    results = []
    outputs, targets = [], []
    target_bb_centers, target_bb_smallests = [], []
    for i,dp in enumerate(data):
        messages = dp['messages']
        metadata = dp['metadata']
        images = dp['images'] if 'images' in dp else None

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

        # Remove the target action from the prompt during inference.
        messages = [m for m in messages if not (m.get("role") == "assistant")]

        # Optionally add more content to the text input 
        messages[1]['content'][0]['text'] = add_string_after_instruction(messages[1]['content'][0]['text'], "Please output a single action and nothing else.")
        messages[1]['content'][0]['text'] = remove_screen_description(messages[1]['content'][0]['text'])  # code to remove a11y from input 
        #messages[1]['content'] = [c for c in messages[1]['content'] if not (c.get("type") == "image_url")]  # code to remove image from input

        chat_response = client.chat.completions.create(
            model="llava-hf/llava-1.5-7b-hf",
            messages=messages,
            temperature=0.0
        )

        output = chat_response.choices[0].message.content
        outputs.append(output)

        print("output:", output, " & target:", target)

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

    #save_dir = 'vis/sep9/'
    #os.makedirs(save_dir, exist_ok=True)
    #save_html(results, save_dir + 'ac_' + split + '_Llamaall-bs128-ckpt12500.html')
