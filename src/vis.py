import re
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageOps


# Given a dictionary containing various inputs, outputs, and metadata, generate
# an HTML that organizes this information in a datapoint-per-row manner. 
# NOTE: can and is often edited for specific visualization needs.
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
                <th>A11Y</th>
                <th>Input (Before) Image </th>
                <th>Target Text Output</th>
                <th>Model Text Output</th>
                <th>Metric</th>
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
        model_coordinates = parse_coordinates(result['model_output'])

        before_image = Image.open(result['screenshot'])
        #before_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        size = before_image.size

        # Draw colored points on before_image for target_coordinates and model_coordinates
        draw_before = ImageDraw.Draw(before_image)

        # Draw bounding box for the GT center box and GT smallest box, if provided.
        if result['target_bb_center'] not in [None, '']:
            draw_before.rectangle(result['target_bb_center'], outline="#{:02x}{:02x}{:02x}".format(0, 0, 255), width=10)  # blue
        if result['target_bb_smallest'] not in [None, '']:
            draw_before.rectangle(result['target_bb_smallest'], outline="#{:02x}{:02x}{:02x}".format(0, 255, 0), width=10)  # green

        # Draw colored points on before_image for target_coordinates and model_coordinates
        max_x, max_y = int(result['dims'][0]), int(result['dims'][1])
        for coord in target_coordinates:
            mapped_x = map_coordinates(coord[0], 0, max_x, 0, size[0])
            mapped_y = map_coordinates(coord[1], 0, max_y, 0, size[1])
            draw_before.rectangle([mapped_x-10, mapped_y-10, mapped_x+10, mapped_y+10], fill='red')

        for coord in model_coordinates:
            mapped_x = map_coordinates(coord[0], 0, max_x, 0, size[0])
            mapped_y = map_coordinates(coord[1], 0, max_y, 0, size[1])
            draw_before.rectangle([mapped_x-10, mapped_y-10, mapped_x+10, mapped_y+10], fill='blue')

        # Convert images back to base64 for HTML embedding
        buffered_before = BytesIO()
        before_image.save(buffered_before, format="PNG")
        encoded_before_image = base64.b64encode(buffered_before.getvalue()).decode('utf-8')
        result['screenshot'] = encoded_before_image

    for result in results:
        #action_history_formatted = "<br>".join(result['action_history'])
        #action_history_formatted = result['action_history']
        a11y = result['a11y']
        metric = result['metric']
        html_content += f"""
            <tr>
                <td>{result['instruction']}</td>
                <td>{a11y}</td>
                <td><img src="data:image/png;base64,{result['screenshot']}" alt="screenshot"></td>
                <td>{result['target_action']}</td>
                <td>{result['model_output']}</td>
                <td>{metric}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

# Given an image path, encodes the image so that it can be visualized in an HTML.
def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
