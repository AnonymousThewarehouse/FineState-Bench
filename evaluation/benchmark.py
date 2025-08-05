import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from evaluation.LLM_eval import save_evaluation_results, enhance_prompt_with_component_info, run_single_evaluation
from evaluation.Plug_and_play_model import ComponentDetector
from evaluation.utils import ConfigLoader
from evaluation.model_clients import get_model_client, ModelClient

class BenchmarkEvaluator:
    def __init__(self, config_path: str = "config/models_config.yaml", test_case_file: str = "pipline/slider_instructions.jsonl", 
                 use_cache: bool = False, cache_source_dir: Optional[str] = None, data_root: str = "desktop_en"):
        self.config_path = config_path
        self.test_case_file = test_case_file
        self.data_root = data_root
        self.use_cache = use_cache
        self.cache_source_dir = cache_source_dir
        self.logger = logging.getLogger("BenchmarkEvaluator")
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def run_evaluation(self, model_name: str, limit: int = 5, use_ground_truth: bool = True, detector_model: Optional[str] = None, scenario: int = 1) -> None:

        self.scenario = scenario
        config_loader = ConfigLoader(self.config_path)
        model_config = config_loader.get_model_config(model_name)
        
        if not model_config:
            self.logger.error(f"Model configuration not found for {model_name}")
            return
        
        model_type = model_config.get("model_type", "online")
        
        if model_type == "offline":

            base_dir = Path("results") / f"offline_scenario{scenario}"
            result_dir = base_dir / model_name
        else:

            model_parts = model_name.split('/')
            if len(model_parts) > 1:
                provider, model_name_clean = model_parts[0], model_parts[1]
                base_dir = Path("results") / provider
            else:
                base_dir = Path("results")
                model_name_clean = model_name

            dir_components = [model_name_clean]

            if detector_model:
                detector_parts = detector_model.split('/')
                if len(detector_parts) > 1:
                    detector_name = detector_parts[1]
                else:
                    detector_name = detector_parts[0]
                dir_components.append(f"by{detector_name}")

            if use_ground_truth:
                dir_components.append("groundtruth")

            result_dir = base_dir / "_".join(dir_components)
        result_dir.mkdir(parents=True, exist_ok=True)

        config_info = {
            "model_name": model_name,
            "detector_model": detector_model,
            "use_ground_truth": use_ground_truth,
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        with open(result_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        if "desktop_en" in self.data_root:
            test_cases = self._load_desktop_test_cases(limit, result_dir)
        else:
            test_cases = self._load_test_cases(limit, result_dir)
            
        if not test_cases:
            self.logger.error("Test cases not found")
            return

        config_loader = ConfigLoader(self.config_path)
        model_config = config_loader.get_model_config(model_name)
        
        if not model_config:
            self.logger.error(f"Model configuration not found for {model_name}")
            return

        model_type = model_config.get("model_type", "online")

        api_key = model_config.get("api_key", "")
        if model_type == "online" and not api_key:
            self.logger.error(f"API key is empty for online model {model_name}")
            return

        if model_type == "online":
            masked_key = f"{api_key[:5]}...{api_key[-5:]}" if api_key and len(api_key) > 10 else "Not set"
            self.logger.info(f"Using API key: {masked_key}, length: {len(api_key)}")
        else:
            self.logger.info(f"Using offline model {model_name}, no API key required")

        model_client = get_model_client(model_name, api_key, model_config, use_component_detector=detector_model is not None)

        if self.use_cache:
            cache_dir = self.cache_source_dir or self._determine_cache_source_dir(model_name, detector_model, use_ground_truth)
            if cache_dir:
                try:
                    from .patch import create_cached_model_client
                    original_client = model_client
                    
                    model_client = create_cached_model_client(original_client, cache_dir, model_name, self.scenario)
                    self.logger.info(f"Cache enabled, cache source directory: {cache_dir}, scenario: {self.scenario}")
                except Exception as e:
                    self.logger.warning(f"Failed to enable cache, using original model client: {str(e)}")
            else:
                self.logger.warning("No suitable cache source directory found, using original model client")
        
        detector = None
        if detector_model:

            detector_config = config_loader.get_model_config(detector_model)
            if not detector_config:
                self.logger.error(f"Detector model configuration not found for {detector_model}")
                return

            detector_model_type = detector_config.get("type", "online")
            detector_api_key = None if detector_model_type == "offline" else detector_config.get("api_key", "")
            
            if detector_model_type == "online":
                masked_key = f"{detector_api_key[:5]}...{detector_api_key[-5:]}" if detector_api_key and len(detector_api_key) > 10 else "Not set"
                self.logger.info(f"Detector using API key: {masked_key}, length: {len(detector_api_key)}")
            
            detector = ComponentDetector(model=detector_model, api_key=detector_api_key, config=detector_config)
            detector.set_use_ground_truth(use_ground_truth)
            self.logger.info(f"Component detector set to use ground truth bounding boxes: {use_ground_truth}")
        else:
            self.logger.info("Not using component detector")
        
        results = self._process_test_cases(test_cases, detector, result_dir, model_client)
        
        if self.use_cache and hasattr(model_client, 'get_cache_stats'):
            cache_stats = model_client.get_cache_stats()
            self.logger.info(f"Cache statistics: {cache_stats}")
        

        save_evaluation_results(
            model_name=model_name,
            results=results,
            metrics=self._calculate_metrics(results),
            output_dir=str(result_dir)
        )
        
        self.logger.info(f"Evaluation completed, results saved in {result_dir}")
        
    def evaluate_models(
        self,
        models: List[str],
        limit: int = 0,
        mock: bool = False,
        detector_model: Optional[str] = None,
        use_ground_truth: bool = True
    ) -> None:
        os.makedirs("element_detection", exist_ok=True)
        
        for model in models:
            self.logger.info(f"Starting evaluation for model: {model}")
            try:
                # Run evaluation
                self.run_evaluation(
                    model, 
                    limit=limit, 
                    use_ground_truth=use_ground_truth,
                    detector_model=detector_model if not mock else None
                )
            except Exception as e:
                self.logger.error(f"Error evaluating model {model}: {str(e)}")
                continue

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        total_cases = len(results)
        successful_locates = 0
        successful_interactions = 0
        state_awareness_locate_count = 0
        state_awareness_interact_count = 0

        valid_locate_cases = 0
        valid_interact_cases = 0

        for result in results:
            interaction_prediction = result.get("interaction_prediction", {})
            locate_results = interaction_prediction.get("locate_results", {})
            interact_results = interaction_prediction.get("interact_results", {})

            if not locate_results:
                locate_results = result.get("locate_results", {})
            if not interact_results:
                interact_results = result.get("interact_results", {})

            if locate_results:
                valid_locate_cases += 1
            if interact_results:
                valid_interact_cases += 1

            if locate_results and locate_results.get("accuracy", 0) > 0:
                successful_locates += 1

            if locate_results and locate_results.get("state_awareness", 0) > 0:
                state_awareness_locate_count += 1

            if interact_results and interact_results.get("target_accuracy", 0) > 0:
                successful_interactions += 1

            if interact_results and interact_results.get("state_awareness", 0) > 0:
                state_awareness_interact_count += 1

        metrics = {
            "total_cases": total_cases,
            "valid_locate_cases": valid_locate_cases,
            "valid_interact_cases": valid_interact_cases,
            "locate_success_rate": successful_locates / valid_locate_cases if valid_locate_cases > 0 else 0.0,
            "interaction_success_rate": successful_interactions / valid_interact_cases if valid_interact_cases > 0 else 0.0,
            "state_awareness_rate_locate": state_awareness_locate_count / valid_locate_cases if valid_locate_cases > 0 else 0.0,
            "state_awareness_rate_interaction": state_awareness_interact_count / valid_interact_cases if valid_interact_cases > 0 else 0.0,
            "timestamp": datetime.now().isoformat()
        }

        return metrics

    def _process_test_cases(
        self, 
        test_cases: List[Dict[str, Any]], 
        detector: ComponentDetector,
        result_dir: Path,
        model_client: ModelClient
    ) -> List[Dict[str, Any]]:
        results = []

        if hasattr(model_client, 'batch_predict') and len(test_cases) > 1:
            self.logger.info(f"Using batch processing mode for {len(test_cases)} test cases")
            results = self._process_test_cases_batch(test_cases, detector, result_dir, model_client)
        else:
            self.logger.info(f"Using single processing mode for {len(test_cases)} test cases")
            results = self._process_test_cases_single(test_cases, detector, result_dir, model_client)
        
        return results
    
    def _process_test_cases_batch(self, test_cases: List[Dict[str, Any]], detector: ComponentDetector, result_dir: Path, model_client: ModelClient) -> List[Dict[str, Any]]:
        results = []
        batch_size = 4  
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(test_cases) + batch_size - 1)//batch_size
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")

            if self.progress_callback:
                self.progress_callback(i, len(test_cases), f"Batch {batch_num}/{total_batches}")
            
            try:
                batch_results = []
                for j, test_case in enumerate(batch):
                    case_index = i + j
                    original_index = test_case.get('_original_index', case_index)
                    
                    try:
                        if self.progress_callback:
                            image_name = os.path.basename(test_case.get("image", ""))
                            self.progress_callback(case_index, len(test_cases), f"Processing {image_name} (#{original_index})")
                        
                        result = self._process_single_test_case(original_index, test_case, detector, model_client, model_client.model_name)
                        if result:
                            batch_results.append(result)
                            output_file = result_dir / f"result_{original_index}.json"
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        self.logger.error(f"Error processing test case {original_index}: {str(e)}")
                
                results.extend(batch_results)

                if hasattr(model_client, 'clear_cache'):
                    model_client.clear_cache()
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")

        if self.progress_callback:
            self.progress_callback(len(test_cases), len(test_cases), "Batch processing completed")
        
        return results
    
    def _process_test_cases_single(self, test_cases: List[Dict[str, Any]], detector: ComponentDetector, result_dir: Path, model_client: ModelClient) -> List[Dict[str, Any]]:
        results = []
        for i, test_case in enumerate(test_cases):
            try:
                original_index = test_case.get('_original_index', i)

                if self.progress_callback:
                    image_name = os.path.basename(test_case.get("image", ""))
                    self.progress_callback(i, len(test_cases), f"Processing {image_name} (#{original_index})")
                
                result = self._process_single_test_case(original_index, test_case, detector, model_client, model_client.model_name)
                if result:
                    results.append(result)
                    output_file = result_dir / f"result_{original_index}.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
                if self.progress_callback:
                    self.progress_callback(i + 1, len(test_cases), f"Completed {image_name} (#{original_index})")
                    
            except Exception as e:
                original_index = test_case.get('_original_index', i)
                self.logger.error(f"Error processing test case {original_index}: {str(e)}")

        if self.progress_callback:
            self.progress_callback(len(test_cases), len(test_cases), "Single processing mode completed")
            
        return results

    def _process_single_test_case(
        self,
        index: int,
        test_case: Dict[str, Any],
        detector: ComponentDetector,
        model_client: ModelClient,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        image_path = test_case.get("image")
        if not image_path:
            self.logger.warning(f"Test case {index} missing image path")
            return None

        normalized_path = self._normalize_image_path(image_path)
        
        if not os.path.exists(normalized_path):
            if '_image_path' in test_case and os.path.exists(test_case['_image_path']):
                image_path = test_case['_image_path']
            else:
                alt_path = os.path.join('element_detection', os.path.basename(normalized_path))
                if os.path.exists(alt_path):
                    image_path = alt_path
                else:
                    self.logger.warning(f'Image does not exist: {normalized_path} (original path: {image_path})')
                    return None
        else:
            image_path = normalized_path
            
        self.logger.info(f"Processing image {image_path}")

        instructions = test_case.get("component", {}).get("instructions", {})
        original_locate_prompt = instructions.get("locate", "")
        original_interact_prompt = instructions.get("interact", "")

        locate_prompt = original_locate_prompt
        interact_prompt = original_interact_prompt
        formatted_info = {}
        if self.use_cache and hasattr(model_client, 'prompt_cache'):
            try:
                cached_locate_prompt = model_client.prompt_cache.get_cached_prompt(image_path, "enhanced_locate_prompt")
                cached_interact_prompt = model_client.prompt_cache.get_cached_prompt(image_path, "enhanced_interact_prompt")
                
                if cached_locate_prompt:
                    locate_prompt = cached_locate_prompt
                    self.logger.info(f"Loaded enhanced locate prompt from cache: {len(cached_locate_prompt)} characters")
                if cached_interact_prompt:
                    interact_prompt = cached_interact_prompt  
                    self.logger.info(f"Loaded enhanced interact prompt from cache: {len(cached_interact_prompt)} characters")
            except Exception as e:
                self.logger.warning(f"Failed to get enhanced prompts from cache: {str(e)}")

        elif detector:
            try:
                component_info = detector.detect_component(image_path)
                formatted_info = detector.format_component_info(component_info)

                if formatted_info["description"]:
                    self.logger.info(f"Component description: {formatted_info['description'][:50]}...")
                if formatted_info["bbox"]:
                    self.logger.info(f"Detected bounding box coordinates: {formatted_info['bbox']}")

                locate_prompt = enhance_prompt_with_component_info(original_locate_prompt, formatted_info) if original_locate_prompt else ""
                interact_prompt = enhance_prompt_with_component_info(original_interact_prompt, formatted_info) if original_interact_prompt else ""
                
            except Exception as e:
                self.logger.warning(f"Component detection failed: {str(e)}")
                formatted_info = {}
                locate_prompt = original_locate_prompt
                interact_prompt = original_interact_prompt
        
        interaction_result = {}

        if locate_prompt:
            locate_test_case = {
                "image": image_path,
                "component": {
                    "instructions": {
                        "locate": locate_prompt
                    }
                }
            }

            processing_details = test_case.get("processing_details", {})
            if processing_details:
                locate_test_case["processing_details"] = processing_details
            
            locate_result = run_single_evaluation(
                model_client=model_client,
                test_case=locate_test_case,
                base_prompt="",
                use_component_detector=bool(detector),
                use_actual_bbox=True,  
                model_name=model_name
            )
            interaction_result = locate_result
        
        if interact_prompt:

            interact_test_case = {
                "image": image_path,
                "component": {
                    "instructions": {
                        "interact": interact_prompt
                    }
                }
            }

            processing_details = test_case.get("processing_details", {})
            if processing_details:
                interact_test_case["processing_details"] = processing_details
            

            interact_result = run_single_evaluation(
                model_client=model_client,
                test_case=interact_test_case,
                base_prompt="",
                use_component_detector=bool(detector),
                use_actual_bbox=True,  
                model_name=model_name
            )

            if interaction_result:
                interaction_result["interact_results"] = interact_result.get("interact_results")
            else:
                interaction_result = interact_result

        processing_details = test_case.get("processing_details", {})
        ground_truth_bbox = processing_details.get("bbox")
        ground_truth_now_coords = processing_details.get("now_coords") 
        ground_truth_target_coords = processing_details.get("target_coords")

        result = {
            "image_path": image_path,
            "original_locate_prompt": original_locate_prompt,
            "original_interact_prompt": original_interact_prompt,
            "enhanced_locate_prompt": locate_prompt if locate_prompt != original_locate_prompt else "",
            "enhanced_interact_prompt": interact_prompt if interact_prompt != original_interact_prompt else "",
            "component_detection": formatted_info if detector else {},
            "interaction_prediction": interaction_result,
            "ground_truth": {
                "bbox": ground_truth_bbox,
                "now_coords": ground_truth_now_coords, 
                "target_coords": ground_truth_target_coords,
                "original_data": test_case.get("processing_details", None)  
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _determine_cache_source_dir(self, model_name: str, detector_model: Optional[str], use_ground_truth: bool) -> Optional[str]:

        if self.cache_source_dir:
            if os.path.exists(self.cache_source_dir):
                return self.cache_source_dir
            else:
                self.logger.warning(f"Specified cache source directory does not exist: {self.cache_source_dir}")
                return None

        potential_sources = [
            "results/google/gemini-2.5-flash-preview_bygemini-2.5-flash-preview_groundtruth",
            "results/google/gemini-2.5-flash-preview_bygemini-2.5-flash-preview",
            "results/google/gemini-2.5-flash-preview_groundtruth",
            "results/google/gemini-2.5-flash-preview"
        ]
        
        for source_dir in potential_sources:
            if os.path.exists(source_dir):

                result_files = list(Path(source_dir).glob("result_*.json"))
                if result_files:
                    self.logger.info(f"Auto-detected cache source directory: {source_dir}")
                    return source_dir
        
        self.logger.warning("No suitable cache source directory found")
        return None

    def _check_completed_tests(self, result_dir: Path) -> set:

        completed_indices = set()
        
        if not result_dir.exists():
            return completed_indices
            
        for result_file in result_dir.glob("result_*.json"):
            try:

                filename = result_file.stem
                if filename.startswith("result_"):
                    index_str = filename[7:]  
                    index = int(index_str)
                    
                    if self._validate_result_file(result_file):
                        completed_indices.add(index)
                        
            except (ValueError, Exception) as e:
                self.logger.warning(f"Error checking result file {result_file}: {e}")
                
        return completed_indices
    
    def _validate_result_file(self, file_path: Path) -> bool:

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            required_fields = ['image_path', 'interaction_prediction', 'ground_truth', 'timestamp']
            if not all(field in data for field in required_fields):
                return False

            ip = data['interaction_prediction']
            if 'locate_results' not in ip or 'interact_results' not in ip:
                return False

            locate_response = ip['locate_results'].get('response', '')
            interact_response = ip['interact_results'].get('response', '')
            
            if (locate_response.startswith('Error:') and 
                interact_response.startswith('Error:') and
                'Model weights error' in locate_response):
                return False
            
            return True
            
        except (json.JSONDecodeError, Exception):
            return False

    def _load_test_cases(self, limit: int = 0, result_dir: Path = None) -> List[Dict[str, Any]]:
        """Load test cases, support skipping completed tests"""
        test_cases = []

        if not os.path.exists(self.test_case_file):
            self.logger.error(f"Test case file does not exist: {self.test_case_file}")
            return test_cases

        completed_indices = set()
        if result_dir and result_dir.exists():
            completed_indices = self._check_completed_tests(result_dir)
            if completed_indices:
                self.logger.info(f"Detected {len(completed_indices)} completed test cases")

        all_test_cases = []
        with open(self.test_case_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    test_case = json.loads(line)
                    all_test_cases.append(test_case)
                    if limit > 0 and len(all_test_cases) >= limit:
                        break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing line {line_num}: {e}")
                    continue

        indexed_test_cases = []
        for i, test_case in enumerate(all_test_cases):
            if i not in completed_indices:

                test_case['_original_index'] = i
                indexed_test_cases.append(test_case)
        
        skipped_count = len(completed_indices)
        total_loaded = len(indexed_test_cases)
        
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} completed test cases, need to process {total_loaded} test cases")
        else:
            self.logger.info(f"Successfully loaded {total_loaded} test cases")
        
        return indexed_test_cases
        
    def _load_desktop_test_cases(self, limit: int = 0, result_dir: Path = None) -> List[Dict[str, Any]]:
        test_cases = []
        
        categories = {
            'A': 'ANumericRange240',
            'B': 'BToggleOption200', 
            'C': 'CSpecificData200',
            'D': 'DViewManipilation160'
        }
        
        completed_indices = set()
        if result_dir and result_dir.exists():
            completed_indices = self._check_completed_tests(result_dir)
            if completed_indices:
                self.logger.info(f"Detected {len(completed_indices)} completed test cases")
        
        case_index = 0
        for category_key, category_name in categories.items():
            image_dir = Path(self.data_root) / category_name
            json_dir = Path(self.data_root) / f"{category_name}json"
            
            if not image_dir.exists() or not json_dir.exists():
                self.logger.warning(f"Skipping non-existent directories: {image_dir} or {json_dir}")
                continue

            json_files = list(json_dir.glob("*.json"))
            
            for json_file in json_files:
                subcategory = json_file.stem 

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        if category_key == 'C':
                            
                            json_data = json.load(f)
                        else:
                            
                            json_data = []
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    json_data.append(json.loads(line))
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"Error parsing line {line_num} in {json_file}: {e}")
                                    continue

                    for test_case in json_data:
                        if case_index in completed_indices:
                            case_index += 1
                            continue

                        image_path = test_case.get('image', '')
                        if not image_path:
                            self.logger.warning(f"Test case missing image path: {test_case}")
                            case_index += 1
                            continue
                        

                        image_path = image_path.replace('\\', '/')
                        if not image_path.startswith(self.data_root):
                            image_path = os.path.join(self.data_root, image_path.split('/')[-3], image_path.split('/')[-2], image_path.split('/')[-1])
                        
                        if not os.path.exists(image_path):
                            self.logger.warning(f"Image file does not exist: {image_path}")
                            case_index += 1
                            continue

                        converted_case = self._convert_desktop_case_to_standard(test_case, image_path, case_index)
                        if converted_case:
                            test_cases.append(converted_case)
                        
                        case_index += 1

                        if limit > 0 and len(test_cases) >= limit:
                            break
                    
                    if limit > 0 and len(test_cases) >= limit:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error reading JSON file {json_file}: {e}")
                    continue
            
            if limit > 0 and len(test_cases) >= limit:
                break
        
        self.logger.info(f"Successfully loaded {len(test_cases)} desktop test cases")
        return test_cases
    
    def _convert_desktop_case_to_standard(self, desktop_case: Dict[str, Any], image_path: str, index: int) -> Dict[str, Any]:

        try:
            component = desktop_case.get('component', {})
            processing_details = desktop_case.get('processing_details', {})
            

            width = 1920  
            height = 1080  

            standard_case = {
                "image": os.path.basename(image_path),
                "width": width,
                "height": height,
                "component": {
                    "name": component.get('name', 'Unknown'),
                    "type": component.get('type', 'UNKNOWN').lower(),
                    "current_state": component.get('current_state', 'initial'),
                    "target_state": component.get('target_state', 'modified'),
                    "instructions": component.get('instructions', {})
                },
                "processing_details": {
                    "component_type_detected": component.get('type', 'UNKNOWN'),
                    "bbox": processing_details.get('locate', []),
                    "now_coords": processing_details.get('loactebbox', []),
                    "target_coords": processing_details.get('interactbbox', []),
                    "locate": processing_details.get('locate', []),
                    "loactebbox": processing_details.get('loactebbox', []),
                    "interact": processing_details.get('interact', []),
                    "interactbbox": processing_details.get('interactbbox', [])
                },
                "_original_index": index,
                "_image_path": image_path
            }
            
            return standard_case
            
        except Exception as e:
            self.logger.error(f"Error converting test case: {e}")
            return None
    
    def _normalize_image_path(self, image_path: str) -> str:

        if not image_path:
            return image_path
            

        normalized_path = image_path.replace('\\', '/')

        if self.data_root == "mobile_en":

            if normalized_path.startswith('mobile/'):
                normalized_path = normalized_path[7:]  

            full_path = os.path.join(self.data_root, normalized_path)

            if not os.path.exists(full_path):

                name, ext = os.path.splitext(full_path)
                if ext.lower() in ['.png', '.jpg', '.jpeg']:
                    upper_ext_path = name + ext.upper()
                    if os.path.exists(upper_ext_path):
                        return upper_ext_path

                    lower_ext_path = name + ext.lower()
                    if os.path.exists(lower_ext_path):
                        return lower_ext_path
            
            return full_path
        else:
            return normalized_path