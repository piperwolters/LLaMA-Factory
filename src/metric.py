import re

from bb_utils import extract_app_bb, extract_bbs_from_a11y, find_gt_box, within_bounding_box

# Metric function aiming to replicate the step-wise accuracy as described in Appendix D.3 of
# the AndroidControl paper. Very specific to the action space and edge cases in that dataset.
def compute_stepwise_accuracy(ground_truth, predictions, target_bbs):
    def parse_action(action_str):
        action_str = action_str.strip().lower()
        if action_str.startswith("click"):
            coords = re.findall(r'\d+', action_str)
            if len(coords) >= 2:
                return {"type": "click", "coords": (int(coords[0]), int(coords[1]))}
            else:
                return {"type": "click"}
        elif action_str.startswith("long press"):
            coords = re.findall(r'\d+', action_str)
            if len(coords) >= 2:
                return {"type": "long press", "coords": (int(coords[0]), int(coords[1]))}
            else:
                return {"type": "long press"}
        elif action_str.startswith("type"):
            text = action_str[5:]
            return {"type": "type", "text": text}
        elif action_str.startswith("scroll"):
            if len(action_str.split()) >= 2:
                direction = action_str.split()[1]
            else:
                direction = ''
            return {"type": "scroll", "direction": direction}
        elif action_str == "wait":
            return {"type": "wait"}
        elif action_str.startswith("open app"):
            app_name = action_str[9:]
            return {"type": "open_app", "app_name": app_name}
        elif action_str == "navigate home":
            return {"type": "navigate home"}
        elif action_str == "navigate back":
            return {"type": "navigate back"}
        return {"type": None}

    correct_predictions = 0
    metrics = []
    for gt_action, pred_action, gt_box in zip(ground_truth, predictions, target_bbs):
        metric = 'incorrect'  # default to a prediction being incorrect until proven otherwise

        gt_parsed = parse_action(gt_action)
        pred_parsed = parse_action(pred_action)

        if gt_parsed["type"] == pred_parsed["type"]:
            if gt_parsed["type"] in ["click", "long press"]:
                gt_coords = gt_parsed["coords"]
                if "coords" in pred_parsed and gt_box != None:
                    pred_coords = pred_parsed["coords"]
                    if within_bounding_box(pred_coords, gt_box):
                        correct_predictions += 1
                        metric = 'correct'

            elif gt_parsed["type"] == "type":
                if gt_parsed["text"] == pred_parsed["text"]:
                    correct_predictions += 1
                    metric = 'correct'

            elif gt_parsed["type"] == "scroll":
                if gt_parsed["direction"] == pred_parsed["direction"]:
                    correct_predictions += 1
                    metric = 'correct'

            elif gt_parsed["type"] in ["navigate home", "navigate back", "wait"]:
                correct_predictions += 1  # These actions have no parameters to compare
                metric = 'correct'

            elif gt_parsed["type"] == "open_app":
                if pred_parsed["app_name"] == gt_parsed["app_name"]:
                    correct_predictions += 1
                    metric = 'correct'

            else:
                if gt_parsed == pred_parsed:
                    correct_predictions += 1
                    metric = 'correct'

        else:
            # Consider open_app and click on app name equivalent
            if pred_parsed["type"] == "click" and gt_parsed["type"] == "open_app":
                if gt_box != None and "coords" in pred_parsed:
                    if within_bounding_box(pred_parsed['coords'], gt_box):
                        correct_predictions += 1
                        metric = 'correct'

        metrics.append(metric)

    return (correct_predictions / len(ground_truth)) * 100, metrics
