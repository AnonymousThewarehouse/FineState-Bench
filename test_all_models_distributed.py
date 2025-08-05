import os
import sys
import subprocess
import time
import json
import threading
import queue
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.getcwd())

os.environ['TEST_NON_INTERACTIVE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

ALL_MODELS = [
    "AgentCPM-GUI",
    "cogagent-9b-20241220", 
    "Holo1-7B",
    "Jedi-7B-1080p",
    "MobileVLM_V2-3B",
    "MobileVLM_V2-7B", 
    "OS-Atlas-Base-7B",
    "ShowUI-2B",
    "UGround-V1-7B",
    "UI-TARS-1.5-7B",
]

MIN_FREE_MEMORY_GB = 8   
MAX_MEMORY_USAGE_PERCENT = 85  
GPU_EXCLUDE_LIST = []  
GPUS_PER_MODEL = 2  

MODEL_MEMORY_REQUIREMENTS = {
    "MobileVLM_V2-7B": 16,  
    "cogagent-9b-20241220": 16,  
    "Jedi-7B-1080p": 16, 
    "OS-Atlas-Base-7B": 13,
    "UGround-V1-7B": 13, 
    "UI-TARS-1.5-7B": 12, 
    "Holo1-7B": 14,
    "AgentCPM-GUI": 10,
    "MobileVLM_V2-3B": 8, 
    "ShowUI-2B": 6 
}

class GPUMonitor:
    
    def __init__(self, exclude_gpus=None):
        self.exclude_gpus = exclude_gpus or []
        self.all_gpus = self._get_all_gpus()
        self.available_gpus = [gpu for gpu in self.all_gpus if gpu not in self.exclude_gpus]
        self.gpu_locks = {gpu: threading.Lock() for gpu in self.available_gpus}
        self.gpu_usage = {gpu: False for gpu in self.available_gpus}
        
    def _get_all_gpus(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            return [int(gpu.strip()) for gpu in result.stdout.strip().split('\n') if gpu.strip()]
        except Exception as e:
            print(f"Error getting GPU list: {e}")
            return []
    
    def get_gpu_info(self, gpu_id):
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits', f'--id={gpu_id}'
            ], capture_output=True, text=True, check=True)
            
            parts = result.stdout.strip().split(', ')
            used_mb = int(parts[0])
            total_mb = int(parts[1])
            free_mb = int(parts[2])
            utilization = int(parts[3]) if parts[3] != 'N/A' else 0
            
            return {
                'used_mb': used_mb,
                'total_mb': total_mb,
                'free_mb': free_mb,
                'free_gb': free_mb / 1024,
                'utilization': utilization,
                'usage_percent': (used_mb / total_mb) * 100
            }
        except Exception as e:
            print(f"Error getting GPU {gpu_id} info: {e}")
            return None
    
    def find_best_gpus(self, model_name=None, num_gpus=GPUS_PER_MODEL):
        suitable_gpus = []
        
        required_memory = MIN_FREE_MEMORY_GB
        if model_name and model_name in MODEL_MEMORY_REQUIREMENTS:
            required_memory = max(required_memory, MODEL_MEMORY_REQUIREMENTS[model_name])
        
        for gpu_id in self.available_gpus:
            if self.gpu_usage[gpu_id]:
                continue
                
            info = self.get_gpu_info(gpu_id)
            if info is None:
                continue

            if (info['free_gb'] >= required_memory and 
                info['usage_percent'] <= 85):  
                
                suitable_gpus.append((gpu_id, info['free_gb']))
        
        suitable_gpus.sort(key=lambda x: x[1], reverse=True)
        selected_gpus = [gpu_id for gpu_id, _ in suitable_gpus[:num_gpus]]
        
        return selected_gpus if len(selected_gpus) >= num_gpus else []
    
    def reserve_gpus(self, gpu_ids):
        for gpu_id in gpu_ids:
            if gpu_id not in self.gpu_usage or self.gpu_usage[gpu_id]:
                return False
        
        for gpu_id in gpu_ids:
            self.gpu_usage[gpu_id] = True
        return True
    
    def release_gpus(self, gpu_ids):
        for gpu_id in gpu_ids:
            if gpu_id in self.gpu_usage:
                self.gpu_usage[gpu_id] = False
    
    def wait_for_free_gpus(self, model_name=None, num_gpus=GPUS_PER_MODEL, timeout=1500):
        start_time = time.time()
        required_memory = MIN_FREE_MEMORY_GB
        if model_name and model_name in MODEL_MEMORY_REQUIREMENTS:
            required_memory = max(required_memory, MODEL_MEMORY_REQUIREMENTS[model_name])
            
        while time.time() - start_time < timeout:
            gpu_ids = self.find_best_gpus(model_name, num_gpus)
            if len(gpu_ids) >= num_gpus:
                if self.reserve_gpus(gpu_ids):
                    gpu_list = ', '.join(map(str, gpu_ids))
                    print(f"🔒 Reserved GPUs [{gpu_list}] for {model_name} ({required_memory}GB required per GPU)")
                    return gpu_ids
            print(f"Waiting for {num_gpus} free GPUs for {model_name}... (need {required_memory}GB each)")
            time.sleep(10)
        return None

class ModelTester:
    
    def __init__(self, gpu_monitor):
        self.gpu_monitor = gpu_monitor
        self.results = {}
        self.results_lock = threading.Lock()
        
    def cleanup_gpu_memory(self, gpu_ids):
        cleanup_code = """
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
"""
        gpu_list = ','.join(map(str, gpu_ids))
        try:
            subprocess.run([
                'python', '-c', cleanup_code
            ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_list}, 
            capture_output=True, timeout=30)
        except Exception as e:
            print(f"Warning: GPUs {gpu_list} cleanup failed: {e}")
    
    def check_model_completion(self, model_name):
        model_dir = Path(f"results/mobile/{model_name}")
        if not model_dir.exists():
            return False, 0
            
        result_files = list(model_dir.glob("result_*.json"))
        metrics_file = model_dir / f"{model_name}_metrics.json"
        
        if model_name == "UI-TARS-1.5-7B":
            expected_files = 230
            min_valid_files = 220  
        else:
            expected_files = 821
            min_valid_files = 800  
        if not metrics_file.exists() or len(result_files) < min_valid_files:
            return False, len(result_files)
        
        valid_files = 0
        for result_file in result_files:
            if self._validate_result_file(result_file):
                valid_files += 1
        return valid_files >= min_valid_files, valid_files
    
    def _validate_result_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
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
            
            if 'bbox' not in data['ground_truth']:
                return False
            
            return True
        
        except json.JSONDecodeError:
            return False
        except Exception:
            return False
    
    def test_model(self, model_name, max_retries=2):
        """Test a single model with automatic GPU allocation"""
        print(f"\n🔍 Starting test for {model_name}...")
        
        model_path = Path(f"models/{model_name}")
        if not model_path.exists():
            result = {"success": False, "error": "Model directory not found", "gpu": None}
            with self.results_lock:
                self.results[model_name] = result
            return result
        
        is_complete, result_count = self.check_model_completion(model_name)
        if is_complete:
            print(f"✅ {model_name} already completed ({result_count} valid results)")
            result = {"success": True, "error": None, "gpu": "cached", "result_count": result_count}
            with self.results_lock:
                self.results[model_name] = result
            return result
        elif result_count > 0:
            print(f"🔄 {model_name} partially completed ({result_count} valid results, need 90+)")
        else:
            print(f"🆕 {model_name} starting fresh")
        
        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1}/{max_retries} for {model_name}")
            
            gpu_ids = self.gpu_monitor.wait_for_free_gpus(model_name=model_name, num_gpus=GPUS_PER_MODEL, timeout=600)
            if gpu_ids is None:
                required_memory = MODEL_MEMORY_REQUIREMENTS.get(model_name, MIN_FREE_MEMORY_GB)
                print(f"❌ No free GPUs available for {model_name} (requires {GPUS_PER_MODEL} GPUs with {required_memory}GB each)")
                result = {"success": False, "error": f"No free GPUs available (requires {GPUS_PER_MODEL} GPUs with {required_memory}GB each)", "gpus": None}
                with self.results_lock:
                    self.results[model_name] = result
                return result
            
            gpu_list = ', '.join(map(str, gpu_ids))
            print(f"🎯 Testing {model_name} on GPUs [{gpu_list}]")
            
            self.cleanup_gpu_memory(gpu_ids)
            
            test_samples = 230 if model_name == "UI-TARS-1.5-7B" else 821 
            result = self._run_test(model_name, gpu_ids, test_samples)

            if gpu_ids is not None:
                self.gpu_monitor.release_gpus(gpu_ids)
                print(f"🔓 Released GPUs [{gpu_list}]")
            
            if result["success"]:
                with self.results_lock:
                    self.results[model_name] = result
                return result
            else:
                print(f"⚠️ Attempt {attempt + 1} failed for {model_name}: {result['error']}")
                if attempt < max_retries - 1:
                    time.sleep(30)  

        result = {"success": False, "error": "All attempts failed", "gpus": gpu_ids if 'gpu_ids' in locals() else None}
        with self.results_lock:
            self.results[model_name] = result
        return result
    
    def _run_test(self, model_name, gpu_ids, test_samples=821):
        test_code = f"""
import sys
import os
import time
import json
from pathlib import Path
sys.path.append(os.getcwd())

# Set environment variables  
os.environ['TEST_NON_INTERACTIVE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def log_progress(message):
    print(f"[{{time.strftime('%H:%M:%S')}}] {model_name}: {{message}}", flush=True)

class RealTimeMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results_dir = Path(f"results/mobile/{{model_name}}")
        self.last_check = 0
        self.check_interval = 10  # Check every 10 samples or when total samples are reached
        
    def check_progress(self, current_index, total_samples):
        # Check success rate every check_interval samples or when total samples are reached
        if current_index % self.check_interval == 0 or current_index == total_samples:
            self.calculate_and_report_success_rate(current_index, total_samples)
    
    def calculate_and_report_success_rate(self, processed, total):
        if not self.results_dir.exists():
            log_progress(f"Progress: {{processed}}/{{total}} ({{processed/total*100:.1f}}%) - Results directory not created")
            return
            
        # Count completed result files
        result_files = list(self.results_dir.glob("result_*.json"))
        
        if not result_files:
            log_progress(f"Progress: {{processed}}/{{total}} ({{processed/total*100:.1f}}%) - No result files yet")
            return
            
        locate_success = 0
        interact_success = 0
        valid_results = 0
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                interaction_pred = result.get('interaction_prediction', {{}})
                locate_results = interaction_pred.get('locate_results', {{}})
                interact_results = interaction_pred.get('interact_results', {{}})
                
                # Check locate success rate (accuracy > 0 means success)
                if locate_results.get('accuracy', 0) > 0:
                    locate_success += 1
                    
                # Check interact success rate (target_accuracy > 0 means success)
                if interact_results.get('target_accuracy', 0) > 0:
                    interact_success += 1
                    
                valid_results += 1
                
            except Exception as e:
                continue
        
        if valid_results > 0:
            locate_rate = locate_success / valid_results * 100
            interact_rate = interact_success / valid_results * 100
            overall_rate = (locate_rate + interact_rate) / 2
            
            log_progress(f"📊 Progress: {{processed}}/{{total}} ({{processed/total*100:.1f}}%) | "
                        f"Completed: {{valid_results}} | "
                        f"Locate Success Rate: {{locate_rate:.1f}}% | "
                        f"Interact Success Rate: {{interact_rate:.1f}}% | "
                        f"Overall Success Rate: {{overall_rate:.1f}}%")
        else:
            log_progress(f"Progress: {{processed}}/{{total}} ({{processed/total*100:.1f}}%) - No valid results yet")

# Create real-time monitor
monitor = RealTimeMonitor('{model_name}')

try:
    log_progress("Starting model loading...")
    from evaluation.benchmark import BenchmarkEvaluator
    log_progress("Model loaded, starting inference test...")
    
    # Create custom progress callback function
    def progress_callback(current, total, message):
        monitor.check_progress(current, total)
    
    # Directly use BenchmarkEvaluator and specify using desktop_en data
    evaluator = BenchmarkEvaluator(data_root="desktop_en")
    evaluator.set_progress_callback(progress_callback)
    evaluator.run_evaluation(
        model_name='{model_name}',
        limit={test_samples},
        scenario=1,
        detector_model=None,
        use_ground_truth=True
    )
    
    # Final statistics
    monitor.calculate_and_report_success_rate({test_samples}, {test_samples})
    log_progress("Inference test completed, result: True")
    print('SUCCESS: True')
    exit(0)
except Exception as e:
    log_progress(f"Error occurred: {{e}}")
    print(f'ERROR: {{e}}')
    import traceback
    traceback.print_exc()
    exit(1)
"""
        
        gpu_list = ','.join(map(str, gpu_ids))
        start_time = time.time()
        try:
            result = subprocess.run([
                'python', '-c', test_code
            ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_list},
            capture_output=True, text=True, timeout=None)  
            
            duration = time.time() - start_time
            
            if result.returncode == 0 and "SUCCESS: True" in result.stdout:
                print(f"✅ {model_name} - PASSED ({duration:.1f}s) on GPUs [{gpu_list}]")
                
                metrics = self._get_model_metrics(model_name)
                
                return {
                    "success": True,
                    "error": None,
                    "gpus": gpu_ids,
                    "duration": duration,
                    "metrics": metrics
                }
            else:
                error_msg = result.stderr[-500:] if result.stderr else result.stdout[-500:]
                print(f"❌ {model_name} - FAILED ({duration:.1f}s) on GPUs [{gpu_list}]")
                print(f"   Error: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "gpus": gpu_ids,
                    "duration": duration
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} - TIMEOUT on GPUs [{gpu_list}]")
            return {
                "success": False,
                "error": "Unexpected timeout",
                "gpus": gpu_ids,
                "duration": time.time() - start_time
            }
        except Exception as e:
            print(f"💥 {model_name} - ERROR: {e}")
            return {
                "success": False,
                "error": str(e),
                "gpus": gpu_ids,
                "duration": time.time() - start_time
            }
    
    def _get_model_metrics(self, model_name):
        metrics_file = Path(f"results/mobile/{model_name}/{model_name}_metrics.json")
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    return {
                        'locate_rate': metrics.get('locate_success_rate', 0) * 100,
                        'interact_rate': metrics.get('interaction_success_rate', 0) * 100
                    }
            except Exception as e:
                print(f"Warning: Could not read metrics for {model_name}: {e}")
        return None

def cleanup_incomplete_results():
    print("🧹 Cleaning up incomplete results...")
    
    results_dir = Path("results/mobile")
    if not results_dir.exists():
        return
    
    class TempValidator:
        def _validate_result_file(self, file_path):
            try:
                with open(file_path, 'r') as f:
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

                if 'bbox' not in data['ground_truth']:
                    return False
                
                return True
            
            except json.JSONDecodeError:
                return False
            except Exception:
                return False
    
    validator = TempValidator()
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        if model_name not in ALL_MODELS:
            continue
            
        result_files = list(model_dir.glob("result_*.json"))
        metrics_file = model_dir / f"{model_name}_metrics.json"
        
        if model_name == "UI-TARS-1.5-7B":
            min_files_needed = 220  
        else:
            min_files_needed = 800  

        if not metrics_file.exists() or len(result_files) < min_files_needed:
            print(f"🗑️ Removing incomplete results for {model_name} ({len(result_files)} files, need {min_files_needed}+)")
            shutil.rmtree(model_dir)
            continue

        valid_files = 0
        for result_file in result_files:
            if validator._validate_result_file(result_file):
                valid_files += 1

        if valid_files < min_files_needed:
            print(f"🗑️ Removing incomplete results for {model_name} ({valid_files}/{len(result_files)} valid files, need {min_files_needed}+)")
            shutil.rmtree(model_dir)

def main():
    global GPUS_PER_MODEL
    
    parser = argparse.ArgumentParser(description='Distributed inference test script for models')
    parser.add_argument('--model', '-m', type=str, help='Test specific model only')
    parser.add_argument('--list', '-l', action='store_true', help='List all available models')
    parser.add_argument('--gpus', '-g', type=int, default=3, help=f'Number of GPUs per model (default: 3)')
    parser.add_argument('--cleanup', action='store_true', help='Clean up incomplete results before testing (default: disabled)')
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for i, model in enumerate(ALL_MODELS, 1):
            memory_req = MODEL_MEMORY_REQUIREMENTS.get(model, 10)
            print(f"  {i:2d}. {model} (~{memory_req}GB)")
        return True
    
    GPUS_PER_MODEL = args.gpus
    
    print("🚀 Starting distributed inference testing...")
    
    gpu_monitor = GPUMonitor(exclude_gpus=GPU_EXCLUDE_LIST)
    print(f"📊 Available GPUs: {gpu_monitor.available_gpus}")
    
    if not gpu_monitor.available_gpus:
        print("❌ No GPUs available for testing!")
        return False

    if args.cleanup:
        cleanup_incomplete_results()
    else:
        print("🔒 Skipping cleanup - incomplete results will be preserved")
    
    model_tester = ModelTester(gpu_monitor)
    
    if args.model:
        if args.model not in ALL_MODELS:
            print(f"❌ Model '{args.model}' not found in available models!")
            print("Use --list to see available models.")
            return False
        models_to_test = [args.model]
        print(f"\n🎯 Testing single model: {args.model}")
    else:
        models_to_test = ALL_MODELS
        print(f"\n🎯 Testing all {len(models_to_test)} models")

    sorted_models = sorted(models_to_test, key=lambda x: MODEL_MEMORY_REQUIREMENTS.get(x, 10))
    print(f"\n📋 Models to test (ordered by memory requirement):")
    for model in sorted_models:
        memory_req = MODEL_MEMORY_REQUIREMENTS.get(model, 10)
        print(f"   📊 {model}: ~{memory_req}GB")

    max_parallel_models = len(gpu_monitor.available_gpus) // GPUS_PER_MODEL
    max_workers = max(1, min(max_parallel_models, 3))  
    successful = []
    failed = []
    
    print(f"🔧 Using {max_workers} parallel workers ({GPUS_PER_MODEL} GPUs per model, {len(gpu_monitor.available_gpus)} total GPUs)")

    if len(gpu_monitor.available_gpus) < GPUS_PER_MODEL:
        print(f"❌ Not enough GPUs available! Need {GPUS_PER_MODEL} GPUs per model, but only have {len(gpu_monitor.available_gpus)}")
        return False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_model = {executor.submit(model_tester.test_model, model): model for model in sorted_models}
        
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                if result["success"]:
                    successful.append(model_name)
                    if result.get("metrics"):
                        metrics = result["metrics"]
                        print(f"   📊 {model_name}: Locate {metrics['locate_rate']:.1f}%, Interact {metrics['interact_rate']:.1f}%")
                else:
                    failed.append(model_name)
                    print(f"   ❌ {model_name}: {result['error'][:100]}")
            except Exception as e:
                failed.append(model_name)
                print(f"   💥 {model_name}: Exception during testing: {e}")

    print(f"\n{'='*60}")
    print("📋 FINAL SUMMARY")
    print('='*60)
    
    print(f"\n✅ SUCCESSFUL MODELS ({len(successful)}):")    
    for model in successful:
        result = model_tester.results.get(model, {})
        if result.get("gpus") == "cached":
            print(f"   ✅ {model} (already completed)")
        else:
            gpus = result.get('gpus')
            if gpus:
                gpu_list = ', '.join(map(str, gpus))
                print(f"   ✅ {model} (GPUs [{gpu_list}])")
            else:
                print(f"   ✅ {model} (unknown GPUs)")
    
    print(f"\n❌ FAILED MODELS ({len(failed)}):")    
    for model in failed:
        result = model_tester.results.get(model, {})
        error = result.get('error', 'Unknown error')[:100]
        gpus = result.get('gpus')
        if gpus:
            gpu_list = ', '.join(map(str, gpus))
            print(f"   ❌ {model} (GPUs [{gpu_list}]): {error}")
        else:
            print(f"   ❌ {model}: {error}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'test_results_distributed_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': {
                'total_models': len(sorted_models),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(sorted_models) * 100
            },
            'results': model_tester.results
        }, f, indent=2)
    
    print(f"\n📊 Success Rate: {len(successful)}/{len(sorted_models)} ({len(successful)/len(sorted_models)*100:.1f}%)")
    print(f"📁 Detailed results saved to: {results_file}")
    
    return len(successful) == len(sorted_models)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
