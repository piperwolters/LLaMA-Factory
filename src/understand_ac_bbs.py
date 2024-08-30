import os
import re
import json
import base64
import random
from openai import OpenAI
from io import BytesIO
from PIL import Image, ImageDraw, ImageOps

from bb_utils import extract_bbs_from_a11y, within_bounding_box

def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


# Load in a dataset json and format messages for the model api.
json_file = open('/data/piperw/projects/LLaMA-Factory/data/ac_test_IDD_LL-100.json')
data = json.load(json_file)

# Load in the file that tells us which screenshot corresponds to which datapoint
json_datapaths = open('/data/piperw/projects/LLaMA-Factory/data/ac_test_IDD_LL-100_DATAPATHS.json')
datapaths = json.load(json_datapaths)

dp_to_dims_file = open('/data/piperw/data/osagent/unified/android_control_III/dp_to_dims.json')
dp_to_dims = json.load(dp_to_dims_file)

results = []
original_a11ys, a11ys = [], []
outputs, targets = [], []
for i,dp in enumerate(data):
    print("Processing...", i)

    datapath = datapaths[i]
    dp_idx = datapath.split('/')[-1]

    f = open(datapath + '/data.json')
    data = json.load(f)
    target = data['action']
    if target.startswith("Click"):
        gt_coords = re.findall(r'\d+', target)
    else:
        continue

    instruction = data['instruction']
    dim = dp_to_dims[dp_idx]

    img = Image.open(os.path.join(datapath, 'start_state.png'))
    draw = ImageDraw.Draw(img)
    draw_colors = []

    original_a11y = json.load(open(os.path.join(datapath, 'data.json')))['a11y']
    bbs, bb_centers, bb_sizes, metadata = extract_bbs_from_a11y(original_a11y, dim)

    bounding_boxes_with_metadata = []
    for i,bb in enumerate(bbs):
        if within_bounding_box(gt_coords, bb):
            bounding_boxes_with_metadata.append({
                'bbox': bb,
                'metadata': metadata[i],
                'size': bb_sizes[i]
            })


    bounding_boxes_with_metadata.sort(key=lambda item: item['size'])

    matches = []
    for i, item in enumerate(bounding_boxes_with_metadata):
        l, t, r, b = item['bbox']
        if len(matches) == 0:
            color = "#{:02x}{:02x}{:02x}".format(255, 0, 0)
        else:
            color = generate_random_color()
        draw_colors.append(color)
        matches.append(item['metadata'])

        draw.rectangle([l, t, r, b], outline=color, width=10)

    if len(draw_colors) > 0:
        img.save('tmp_drawing_' + str(dp_idx) + '.png')
        screenshot = 'tmp_drawing_' + str(dp_idx) + '.png'
    else:
        screenshot = os.path.join(datapath, 'start_state.png')

    best_match = matches[0] if len(matches) > 0 else None

    result = {
        'target_action': target,
        'all_matches': [matches, draw_colors],
        'best_match': best_match,
        'screenshot': screenshot,
        'instruction': instruction,
        'dims': dim
    }
    results.append(result)


# A version of save_html function that is specific to visualizating
# bounding boxes of elements that enclose target click/longpress coordinates.
def save_html(results, output_path):
    html_content = """
    <html>
    <head>
        <title>Model Evaluation Results</title>
        <style>
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid black; padding: 10px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 300px; max-height: 500px; }
        </style>
    </head>
    <body>
        <h1>Model Evaluation Results</h1>
        <table>
            <tr>
                <th>Input Text Prompt</th>
                <th>Target Action</th>
                <th>Best Match</th>
                <th>Bounding Boxes</th>
                <th>Screenshot</th>
            </tr>
    """

    def parse_coordinates(text):
        # Assuming coordinates are in the format 'x,y' or 'x1,y1,x2,y2'
        numbers = re.findall(r'\d+', text)
        if len(numbers) == 2:
            return [(int(numbers[0]), int(numbers[1]))]
        elif len(numbers) == 4:
            return [(int(numbers[0]), int(numbers[1])), (int(numbers[2]), int(numbers[3]))]
        else:
            return []

    def map_coordinates(coord, old_min, old_max, new_min, new_max):
        return int(((coord - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min)

    for result in results:
        # Parse coordinates from target_action and model_output
        target_coordinates = parse_coordinates(result['target_action'])

        screenshot = result['screenshot']

        # Load input images
        before_image = Image.open(screenshot)
        size = before_image.size

        # Draw colored points on before_image for target_coordinates and model_coordinates
        draw_before = ImageDraw.Draw(before_image)

        # Draw colored points on before_image for target_coordinates and model_coordinates
        max_x, max_y = int(result['dims'][0]), int(result['dims'][1])
        for coord in target_coordinates:
            mapped_x = map_coordinates(coord[0], 0, max_x, 0, size[0])
            mapped_y = map_coordinates(coord[1], 0, max_y, 0, size[1])
            draw_before.rectangle([mapped_x-10, mapped_y-10, mapped_x+10, mapped_y+10], fill='red')

        # Convert images back to base64 for HTML embedding
        buffered_before = BytesIO()
        before_image.save(buffered_before, format="PNG")
        encoded_before_image = base64.b64encode(buffered_before.getvalue()).decode('utf-8')
        result['screenshot'] = encoded_before_image

    for result in results:

        all_matches_colored = ""
        for string, color in zip(result['all_matches'][0], result['all_matches'][1]):
            all_matches_colored += f'<span style="color:{color};">{string}</span><br>'

        html_content += f"""
            <tr>
                <td>{result['instruction']}</td>
                <td>{result['target_action']}</td>
                <td><span style="color:red;">{result['best_match']}</span></td>
                <td>{all_matches_colored}</td>
                <td><img src="data:image/png;base64,{result['screenshot']}" alt="screenshot"></td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)


save_html(results, 'center_strategy.html')
