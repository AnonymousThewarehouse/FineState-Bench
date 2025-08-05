#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

from .model_clients import get_model_client
from .utils import ConfigLoader
from .Plug_and_play_model import ComponentDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLM_Eval")

def load_evaluation_data(file_path: str) -> List[Dict]:
    """Load evaluation data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(data)} evaluation cases from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {str(e)}")
        return []

def construct_prompt(base_prompt: str, instruction: str, instruction_type: str) -> str:
    """Construct prompt"""
    return f"{base_prompt}\n\nInstruction ({instruction_type}): {instruction}"

def enhance_prompt_with_component_info(prompt: str, component_info: Dict) -> str:
    if not component_info:
        return prompt

    if "COMPONENT INFORMATION:" in prompt:
        return prompt

    description = component_info.get("description", "")
    operation = component_info.get("operation", "")
    bbox = component_info.get("bbox")

    if not description and not operation and not bbox:
        return prompt

    enhanced_prompt = f"{prompt}\n\n"
    enhanced_prompt += "COMPONENT INFORMATION:\n"

    if description:
        enhanced_prompt += f"- Description: {description}\n"

    if operation:
        enhanced_prompt += f"- Suggested Operation: {operation}\n"

    if bbox and len(bbox) == 4:
        enhanced_prompt += f"- Bounding Box: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
        enhanced_prompt += "  (coordinates are normalized to 0-1 range, where (0,0) is top-left and (1,1) is bottom-right)\n"
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        enhanced_prompt += f"- Component Center: ({center_x:.3f}, {center_y:.3f})\n"
            
    enhanced_prompt += "\nPlease use this information to assist your analysis. Focus on the component described above when responding to the instruction."
    
    return enhanced_prompt

def generate_prompt(instruction: str, instruction_type: str, component_info: Optional[Dict] = None) -> str:

    base_prompt = """Please analyze the UI element in the image based on the given instruction.

Task Requirements:
1. Locate the target UI element (could be a button, slider, switch, etc.)
2. Determine the precise interaction point with the element

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- You MUST provide coordinates in EXACTLY this format: [x, y]
- Coordinates MUST be decimals between 0 and 1 (e.g., [0.237, 0.456])
- (0,0) represents the top-left corner, (1,1) represents the bottom-right corner
- NO EXCEPTIONS: Always include numerical coordinates, even if uncertain
- If you cannot determine exact coordinates, provide your best estimate as [x, y]

MANDATORY RESPONSE FORMAT (follow exactly):
1. Component Description: [Brief description of the target UI element]
2. Interaction Coordinates: [0.XXX, 0.YYY]
3. Reasoning: [Explain why you chose this location]

EXAMPLE RESPONSE:
1. Component Description: A horizontal slider with a movable handle
2. Interaction Coordinates: [0.443, 0.483]
3. Reasoning: This is the center point of the slider control

CRITICAL: You MUST replace [0.XXX, 0.YYY] with actual decimal numbers. Do NOT use descriptive text for coordinates. Do NOT write "The element is located at..." - write the numbers directly like [0.443, 0.483]."""

    prompt = f"{base_prompt}\n\nInstruction ({instruction_type}): {instruction}"

    if component_info:
        prompt = enhance_prompt_with_component_info(prompt, component_info)
    
    return prompt

def is_point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    if not point or not bbox or len(bbox) != 4:
        return False
    
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    
    return xmin <= x <= xmax and ymin <= y <= ymax

def evaluate_location_accuracy(predicted_coords: Tuple[float, float], bbox: List[float], tolerance: float = 0.0) -> int:

    if not predicted_coords or not bbox:
        return 0
        
    if is_point_in_bbox(predicted_coords, bbox):
        return 1
    
    return 0  

def evaluate_state_awareness(predicted_coords: Tuple[float, float], now_coords: List[float], tolerance: float = 0.0) -> int:
    if not predicted_coords or not now_coords:
        return 0
    if is_point_in_bbox(predicted_coords, now_coords):
        return 1
    return 0

def run_single_evaluation(model_client: Any, test_case: Dict, base_prompt: str = "", use_component_detector: bool = False, use_actual_bbox: bool = False, model_name: str = None) -> Dict:
    try:
        image_path = test_case.get("image")
        component = test_case.get("component", {})
        instructions = component.get("instructions", {})
        processing_details = test_case.get("processing_details", {})
        
        actual_bbox = processing_details.get("bbox") if use_actual_bbox else None
        now_coords = processing_details.get("now_coords") if use_actual_bbox else None  
        target_coords = processing_details.get("target_coords") if use_actual_bbox else None
        
        if image_path and not os.path.exists(image_path):
            alt_path = os.path.join("element_detection", os.path.basename(image_path))
            if os.path.exists(alt_path):
                image_path = alt_path
                logger.info(f"Using alternative image path: {image_path}")
            else:
                logger.warning(f"Image does not exist: {image_path}, using empty path")
                image_path = None
        
        results = {
            "image": image_path,
            "component_name": component.get("name"),
            "component_type": component.get("type"),
            "locate_results": None,
            "interact_results": None,
            "component_detection": None
        }
        
        if "locate" in instructions:
            component_info = {"bbox": actual_bbox} if use_actual_bbox else None
            locate_prompt = generate_prompt(instructions["locate"], "LOCATE", component_info if use_component_detector else None)
            locate_response = model_client.get_response(image_path, locate_prompt)
            locate_result = parse_model_response(locate_response, model_name)
            locate_coords = locate_result.get("coordinates")
            
            locate_results = {
                "prompt": locate_prompt,
                "response": locate_response,
                "predicted_coords": locate_coords,
                "accuracy": 0,
                "state_awareness": 0
            }
            
            if locate_coords and actual_bbox:
                locate_results["accuracy"] = evaluate_location_accuracy(locate_coords, actual_bbox)
                if now_coords:
                    locate_results["state_awareness"] = evaluate_state_awareness(locate_coords, now_coords)
                else:
                    locate_results["state_awareness"] = evaluate_state_awareness(locate_coords, actual_bbox)
                
            results["locate_results"] = locate_results
        
        if "interact" in instructions:
            component_info = {"bbox": actual_bbox} if use_actual_bbox else None
            interact_prompt = generate_prompt(instructions["interact"], "INTERACT", component_info if use_component_detector else None)
            interact_response = model_client.get_response(image_path, interact_prompt)
            interact_result = parse_model_response(interact_response, model_name)
            interact_coords = interact_result.get("coordinates")
            
            interact_results = {
                "prompt": interact_prompt,
                "response": interact_response,
                "predicted_coords": interact_coords,
                "target_accuracy": 0,
                "state_awareness": 0
            }
            
            if interact_coords and actual_bbox:
                if target_coords:
                    interact_results["target_accuracy"] = evaluate_location_accuracy(interact_coords, target_coords)
                    interact_results["state_awareness"] = evaluate_state_awareness(interact_coords, target_coords)
                else:
                    interact_results["target_accuracy"] = evaluate_location_accuracy(interact_coords, actual_bbox)
                    interact_results["state_awareness"] = evaluate_state_awareness(interact_coords, actual_bbox)
                
            results["interact_results"] = interact_results
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {
            "error": str(e),
            "image": image_path,
            "locate_results": None,
            "interact_results": None
        }


def is_point_near_bbox(point: Tuple[float, float], bbox: List[float], tolerance: float = 0.2) -> bool:
    if not point or not bbox or len(bbox) != 4:
        return False
    
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    
    width = xmax - xmin
    height = ymax - ymin
    
    extended_bbox = [
        xmin - width * tolerance,
        ymin - height * tolerance,
        xmax + width * tolerance,
        ymax + height * tolerance
    ]
    
    return is_point_in_bbox(point, extended_bbox)

def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    if not bbox or len(bbox) != 4:
        return None
    
    xmin, ymin, xmax, ymax = bbox
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)

def parse_model_response(response: str, model_name: str = None) -> Dict:
    try:
        result = {
            "coordinates": None,
            "description": "",
            "reasoning": "",
            "success": False,
            "error": None
        }
        
        if isinstance(response, dict):
            if "error" in response:
                result["error"] = response["error"]
                return result
            return response
            
        if not isinstance(response, str):
            result["error"] = f"Unexpected response type: {type(response)}"
            return result
        
        if model_name and 'OS-Atlas-Pro-7B' in model_name:
            lines = response.split('\n')
            unique_lines = []
            seen_actions = set()
            
            for line in lines:
                line = line.strip()
                if line.startswith('actions:'):
                    if line not in seen_actions:
                        unique_lines.append(line)
                        seen_actions.add(line)
                else:
                    unique_lines.append(line)
            
            response = '\n'.join(unique_lines)
        
        if model_name and 'Qwen2.5-VL-3B-UI-R1' in model_name:
            lines = response.split('\n')
            unique_lines = []
            seen_lines = set()
            
            for line in lines:
                line = line.strip()
                if line and line not in seen_lines:
                    unique_lines.append(line)
                    seen_lines.add(line)
            
            response = '\n'.join(unique_lines)

        

        coord_patterns = [
            r'<point>\[\[(\d+)\s*,\s*(\d+)\]\]</point>',
            r'<points\s+x1="(\d+)"\s+y1="(\d+)"', 
            r'<\|box_start\|\>\[\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*\d*\.?\d+,\s*\d*\.?\d+\]\]<\|box_end\|\>', 
            r'Interaction\s+Coordinates[：:]\s*\[(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]', 
            r'2\.\s*Interaction\s+Coordinates[：:]\s*\[(\d*\.?\d+)\s*,?\s*(\d*\.?\d+)?\]?',  
            r'2\.\s*Interaction\s+Coordinates[：:]\s*\[(\d*\.?\d+)',  
            r'\[(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]', 
            r'\((\d*\.?\d+)\s*,\s*(\d*\.?\d+)\)',  
            r'坐标[：:]\s*\[?(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]?',  
            r'Coordinates[：:]\s*\[?(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]?',  
            r'\((\d+),\s*(\d+)\)',                 
            r'(\d*\.?\d+)\s*,\s*(\d*\.?\d+)', 
        ]
        
        for i, pattern in enumerate(coord_patterns):
            coords_match = re.search(pattern, response)
            if coords_match:
                try:
                    x = float(coords_match.group(1))

                    try:
                        y = float(coords_match.group(2)) if coords_match.group(2) else None
                    except (IndexError, AttributeError):
                        y = None

                    if y is None:
                        logger.debug(f"Y coordinate missing in pattern {i}, trying next pattern")
                        continue  

                    if i == 0:  
                        max_coord = max(x, y)
                        if max_coord > 1024:
                            image_width = 1920
                            image_height = 1920
                        else:
                            
                            image_width = 1024
                            image_height = 1024
                        x = x / image_width
                        y = y / image_height
                    elif i == 1:  

                        if x > 1 or y > 1:  
                            if model_name and 'Jedi' in model_name and '1080p' in model_name:
                                
                                x = x / 1920.0  
                                y = y / 1080.0
                            else:
                                
                                x = x / 1024.0
                                y = y / 1024.0
                    elif i == 2: 
                        
                        pass
                    elif i == 3:  

                        if x > 1 or y > 1:  

                            if model_name and 'InfiGUI' in model_name:
                                max_coord = max(x, y)
                                if max_coord > 1080:
                                    
                                    x = x / 1920.0
                                    y = y / 1920.0
                                else:
                                    
                                    x = x / 1920.0
                                    y = y / 1080.0
                            else:
                                
                                x = x / 1920.0
                                y = y / 1080.0
                    elif i == 11:  
                        
                        if model_name and 'OS-Atlas' in model_name:

                            if x > 1 or y > 1:  
                                x = x / 1000.0  
                                y = y / 1000.0
                        elif model_name and ('UGround' in model_name or 'Qwen2-VL' in model_name):

                            x = x / 1000.0  
                            y = y / 1000.0
                        elif model_name and 'Qwen2.5-VL-3B-UI-R1' in model_name:

                            if x > 1 or y > 1:  

                                x = x / 1024.0  
                                y = y / 1024.0

                    if i == len(coord_patterns) - 1:  

                        if x > 1 or y > 1:
                            if model_name and 'OS-Atlas' in model_name:

                                x = x / 1000.0
                                y = y / 1000.0
                            elif model_name and ('UGround' in model_name or 'Qwen2-VL' in model_name):

                                x = x / 1000.0
                                y = y / 1000.0
                            else:

                                x = x / 1024.0
                                y = y / 1024.0

                    if 0 <= x <= 1 and 0 <= y <= 1:
                        result["coordinates"] = (x, y)
                        result["success"] = True
                        break
                except (ValueError, IndexError):
                    continue

        description_patterns = [
            r'(?:Component Description|组件描述)[：:]\s*(.+?)(?=\n|$)',
            r'1\.\s*(.+?)(?=\n|2\.|\Z)'
        ]
        
        reasoning_patterns = [
            r'(?:Reasoning|判断依据)[：:]\s*(.+?)(?=\n|$)',
            r'3\.\s*(.+?)(?=\n|\Z)'
        ]

        for pattern in description_patterns:
            desc_match = re.search(pattern, response)
            if desc_match:
                result["description"] = desc_match.group(1).strip()
                break

        for pattern in reasoning_patterns:
            reason_match = re.search(pattern, response)
            if reason_match:
                result["reasoning"] = reason_match.group(1).strip()
                break

        if not result["coordinates"] and model_name and ('OS-Atlas-Pro-7B' in model_name or 'Qwen2.5-VL-3B-UI-R1' in model_name):

            location_patterns = [
                r'located at the (\w+)\s*(\w+)',  
                r'at the (\w+)\s*(\w+)',        
                r'in the (\w+)\s*(\w+)',         
            ]
            
            for pattern in location_patterns:
                location_match = re.search(pattern, response, re.IGNORECASE)
                if location_match:
                    pos1 = location_match.group(1).lower()
                    pos2 = location_match.group(2).lower()

                    x, y = 0.5, 0.5  

                    if 'upper' in pos1 or 'top' in pos1:
                        y = 0.25
                    elif 'lower' in pos1 or 'bottom' in pos1:
                        y = 0.75
                    elif 'middle' in pos1 or 'center' in pos1:
                        y = 0.5

                    if 'left' in pos2:
                        x = 0.25
                    elif 'right' in pos2:
                        x = 0.75
                    elif 'center' in pos2 or 'middle' in pos2:
                        x = 0.5
                    
                    if 'upper' in pos2 or 'top' in pos2:
                        y = 0.25
                    elif 'lower' in pos2 or 'bottom' in pos2:
                        y = 0.75
                    
                    result["coordinates"] = (x, y)
                    result["success"] = True
                    result["description"] = f"Estimated from location description: {pos1} {pos2}"
                    result["reasoning"] = "Coordinates estimated from textual location description"
                    break

        if not result["coordinates"]:
            result["error"] = "No valid coordinates found in response"
            
        return result
        
    except Exception as e:
        logger.error(f"Error parsing model response: {str(e)}")
        return {
            "coordinates": None,
            "description": "",
            "reasoning": "",
            "success": False,
            "error": str(e)
        }

def run_model_evaluation(
    model_name: str,
    config_loader: ConfigLoader,
    evaluation_data: List[Dict],
    base_prompt: str,
    use_component_detector: bool = False,
    use_actual_bbox: bool = False
) -> Tuple[List[Dict], Dict]:

    model_config = config_loader.get_model_config(model_name)
    if not model_config:
        logger.error(f"No configuration found for model {model_name}")
        return [], {}
    

    model_type = model_config.get("type", "online")
    api_key = None if model_type == "offline" else model_config.get("api_key", "")
    model_client = get_model_client(
        model_name, 
        api_key, 
        model_config,
        use_component_detector=use_component_detector
    )

    logger.info(f"Running evaluation for model {model_name}")
    logger.info(f"- Use component detector: {use_component_detector}")
    logger.info(f"- Use actual bounding box: {use_actual_bbox}")

    results = []
    for i, test_case in enumerate(evaluation_data):
        try:

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processing test case {i+1}/{len(evaluation_data)}")
            
            result = run_single_evaluation(
                model_client, 
                test_case, 
                base_prompt,
                use_component_detector=use_component_detector,
                use_actual_bbox=use_actual_bbox,
                model_name=model_name
            )

            result["test_case_id"] = i
            result["model_name"] = model_name

            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in test case {i}: {str(e)}")
            results.append({
                "test_case_id": i,
                "model_name": model_name,
                "error": str(e)
            })

    metrics = calculate_metrics(results)
    metrics["model_name"] = model_name
    metrics["total_cases"] = len(evaluation_data)
    metrics["timestamp"] = datetime.now().isoformat()
    
    return results, metrics

def calculate_metrics(results: List[Dict]) -> Dict:

    total_cases = len(results)
    if total_cases == 0:
        return {
            "locate_success_rate": 0.0,
            "interaction_success_rate": 0.0,
            "state_awareness_rate_locate": 0.0,
            "state_awareness_rate_interaction": 0.0
        }

    locate_success = 0
    locate_state_aware = 0
    interact_success = 0
    interact_state_aware = 0
    locate_total = 0
    interact_total = 0
    
    for result in results:

        locate_results = result.get("locate_results")
        if locate_results:
            locate_total += 1
            if locate_results.get("accuracy", 0) > 0:
                locate_success += 1
            if locate_results.get("state_awareness", 0) > 0:
                locate_state_aware += 1

        interact_results = result.get("interact_results")
        if interact_results:
            interact_total += 1
            if interact_results.get("target_accuracy", 0) > 0:
                interact_success += 1
            if interact_results.get("state_awareness", 0) > 0:
                interact_state_aware += 1

    metrics = {
        "locate_success_rate": locate_success / locate_total if locate_total > 0 else 0.0,
        "interaction_success_rate": interact_success / interact_total if interact_total > 0 else 0.0,
        "state_awareness_rate_locate": locate_state_aware / locate_total if locate_total > 0 else 0.0,
        "state_awareness_rate_interaction": interact_state_aware / interact_total if interact_total > 0 else 0.0
    }
    
    return metrics

def save_evaluation_results(model_name: str, results: List[Dict], metrics: Dict, output_dir: str = "results") -> None:

    try:

        os.makedirs(output_dir, exist_ok=True)

        safe_model_name = model_name.replace("/", "_").replace(":", "_")

        results_file = os.path.join(output_dir, f"{safe_model_name}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        metrics_file = os.path.join(output_dir, f"{safe_model_name}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved evaluation results to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")

def main(config_path: str, eval_data_path: str, output_dir: str = "results", 
         use_component_detector: bool = False, use_actual_bbox: bool = False) -> None:

    config_loader = ConfigLoader(config_path)

    evaluation_data = load_evaluation_data(eval_data_path)
    if not evaluation_data:
        logger.error(f"Failed to load evaluation data from {eval_data_path}")
        return

    models = config_loader.get_models()
    if not models:
        logger.error("No models found in configuration")
        return

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Evaluation parameters:")
    logger.info(f"- Use component detector: {use_component_detector}")
    logger.info(f"- Use actual bounding box: {use_actual_bbox}")

    for model_name in models:
        logger.info(f"Running evaluation for model: {model_name}")

        model_config = config_loader.get_model_config(model_name)
        if not model_config:
            logger.error(f"No configuration found for model {model_name}")
            continue

        model_dir_name = model_name.replace("/", "-")
        if use_actual_bbox:
            model_dir_name += "_bygroundtruth"
        model_dir = os.path.join(output_dir, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)

        results, metrics = run_model_evaluation(
            model_name,
            config_loader,
            evaluation_data,
            "",  
            use_component_detector=use_component_detector,
            use_actual_bbox=use_actual_bbox
        )

        save_evaluation_results(model_name, results, metrics, model_dir)
        
        logger.info(f"Evaluation completed for model {model_name}")
        logger.info(f"Metrics: {metrics}")
    
    logger.info("All evaluations completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Evaluation Script")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--eval_data", required=True, help="Path to evaluation data")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--use_component_detector", action="store_true", help="Use component detector")
    parser.add_argument("--use_actual_bbox", action="store_true", help="Use actual bounding box")
    
    args = parser.parse_args()
    main(args.config, args.eval_data, args.output_dir, args.use_component_detector, args.use_actual_bbox)