import sys
import os
import json
import logging
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  建议安装 tqdm 以显示进度条: pip install tqdm")

sys.path.append(os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OFFLINE_SCENARIO2_MODELS = [
    "ShowUI-2B",
    "OS-Atlas-Base-7B"
]

AVAILABLE_DATASETS = [
    "desktop_en",
    "mobile_en", 
    "web_en"
]

SCENARIO2_CONFIGS = {
    "gpt4o": {
        "name": "Component Detection Enhancement (GPT-4o)",
        "description": "Enhanced prompt using GPT-4o cache",
        "detector_model": "openai/gpt-4o-2024-11-20",
        "use_ground_truth": False,
        "use_cache": True,
        "cache_source_dir": "openai/gpt-4o-2024-11-20_bygpt-4o-2024-11-20"
    },
    "gemini": {
        "name": "Component Detection Enhancement (Gemini-2.5-Flash)",
        "description": "Enhanced prompt using Gemini-2.5-Flash cache",
        "detector_model": "google/gemini-2.5-flash",
        "use_ground_truth": False,
        "use_cache": True,
        "cache_source_dir": "google/gemini-2.5-flash_bygemini-2.5-flash"
    }
}

def clear_gpu_memory():
    """Clean up GPU memory - only on GPUs 6 and 7"""
    try:
        cleanup_code = """
import torch
import gc
import os

# Perform multiple garbage collections
for _ in range(3):
    gc.collect()
    
if torch.cuda.is_available():
    # Clear cache for all visible GPUs
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    
# Force release of cached pool memory
torch.cuda.empty_cache()
"""
        subprocess.run([
            'python', '-c', cleanup_code
        ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': '6,7'}, 
        capture_output=True, timeout=30)
        logger.info("GPU 6,7 memory has been actively cleaned up")
    except Exception as e:
        logger.warning(f"Failed to clean up GPU memory: {e}")

def check_model_completion(model_name: str, scenario: int, dataset: str) -> bool:
    result_dir = f"results/{dataset}_scenario{scenario}" if scenario == 2 else f"results/offline_scenario{scenario}"
    metrics_path = f"{result_dir}/{model_name}/{model_name}_metrics.json"
    
    if os.path.exists(metrics_path):
        logger.info(f"✅ Found completed results: {metrics_path}")
        return True
    else:
        logger.info(f"❌ Result file not found: {metrics_path}")
        return False

def list_existing_results(dataset: str, scenario: int) -> dict:
    result_dir = f"results/{dataset}_scenario{scenario}" if scenario == 2 else f"results/offline_scenario{scenario}"
    existing_results = {}
    
    logger.info(f"🔍 Checking result directory: {result_dir}")
    
    if os.path.exists(result_dir):
        for model_name in OFFLINE_SCENARIO2_MODELS:
            model_dir = f"{result_dir}/{model_name}"
            metrics_file = f"{model_dir}/{model_name}_metrics.json"
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)

                    mod_time = os.path.getmtime(metrics_file)
                    mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
                    
                    existing_results[model_name] = {
                        'file_path': metrics_file,
                        'modification_time': mod_time_str,
                        'total_cases': data.get('total_cases', 'Unknown'),
                        'locate_success_rate': data.get('locate_success_rate', 0),
                        'interaction_success_rate': data.get('interaction_success_rate', 0)
                    }
                    logger.info(f"✅ {model_name}: Completed (Modification time: {mod_time_str})")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name}: Result file exists but failed to read: {e}")
            else:
                logger.info(f"❌ {model_name}: Not completed")
    else:
        logger.info(f"📁 Result directory does not exist, will create: {result_dir}")
    
    return existing_results

def read_metrics_file_with_dataset(model_name: str, scenario: int, dataset: str) -> dict:
    if scenario == 2:
        result_dir = f"results/{dataset}_scenario2"
    else:
        if dataset == "desktop_en":
            result_dir = f"results/offline_scenario{scenario}"
        elif dataset == "mobile_en":
            result_dir = "results/mobile"
        elif dataset == "web_en":
            result_dir = "results/web"
        else:
            result_dir = f"results/offline_scenario{scenario}"
    
    metrics_path = f"{result_dir}/{model_name}/{model_name}_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

def wait_for_gpu_availability(gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = [6, 7]
    
    gpu_list = ','.join(map(str, gpu_ids))
    logger.info(f"🔍 Checking GPU availability for {gpu_list}...")
    
    while True:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
                '--format=csv,noheader,nounits', f'--id={gpu_list}'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            gpus_available = 0
            
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.split(', ')
                    used_mb = int(parts[0])
                    total_mb = int(parts[1])
                    free_mb = int(parts[2])
                    
                    usage_percent = (used_mb / total_mb) * 100
                    free_gb = free_mb / 1024
                    
                    gpu_id = gpu_ids[i]
                    logger.info(f"GPU {gpu_id}: Usage {usage_percent:.1f}%, Free memory {free_gb:.1f}GB")

                    if usage_percent < 70 and free_gb > 8:
                        gpus_available += 1
            
            if gpus_available >= len(gpu_ids):
                logger.info(f"✅ All GPUs {gpu_list} are available, starting test")
                break
            else:
                logger.info(f"⏳ Waiting for GPU availability (currently available: {gpus_available}/{len(gpu_ids)}), retrying in 60 seconds...")
                time.sleep(60)
                
        except Exception as e:
            logger.warning(f"Failed to check GPU status: {e}, retrying in 30 seconds...")
            time.sleep(30)

def select_detector_model():
    print("\n" + "="*50)
    print("Please select the bounding box prediction model:")
    print("  1. GPT-4o (openai/gpt-4o-2024-11-20)")
    print("  2. Gemini-2.5-Flash (google/gemini-2.5-flash)")
    print("="*50)
    
    while True:
        try:
            choice = int(input("Please enter the model number (1-2): "))
            if choice == 1:
                return "gpt4o"
            elif choice == 2:
                return "gemini"
            else:
                print("❌ Invalid choice, please enter 1-2")
        except ValueError:
            print("❌ Please enter a valid number")

def select_dataset():
    print("\n" + "="*50)
    print("Please select the dataset:")
    print("  1. Desktop (desktop_en) - 628 samples")
    print("  2. Mobile (mobile_en) - 648 samples")
    print("  3. Web (web_en) - 697 samples")
    print("="*50)
    
    while True:
        try:
            choice = int(input("Please enter the dataset number (1-3): "))
            if choice == 1:
                return "desktop_en", 628
            elif choice == 2:
                return "mobile_en", 648
            elif choice == 3:
                return "web_en", 697
            else:
                print("❌ Invalid choice, please enter 1-3")
        except ValueError:
            print("❌ Please enter a valid number")

def run_scenario_test_with_dataset(model_name: str, scenario: int, dataset: str, limit: int = 10, gpu_id: int = None, detector_key: str = "gpt4o") -> bool:
    scenario_info = SCENARIO2_CONFIGS[detector_key] 

    if gpu_id is None:
        gpu_ids = [6, 7]
        cuda_visible_devices = '6,7'
    else:
        gpu_ids = [gpu_id]
        cuda_visible_devices = str(gpu_id)

    non_interactive = os.environ.get('TEST_NON_INTERACTIVE', '0') == '1'
    
    if not non_interactive:
        logger.info("="*60)
        logger.info(f"🚀 Starting test")
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Scenario: {scenario} ({scenario_info['name']})")
        logger.info(f"Description: {scenario_info['description']}")
        logger.info(f"Number of test cases: {limit}")
        logger.info(f"🎯 Specified GPU: {cuda_visible_devices}")
        logger.info("="*60)

    wait_for_gpu_availability(gpu_ids)
    
    try:
        old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        old_tokenizers_parallelism = os.environ.get('TOKENIZERS_PARALLELISM')
        old_omp_threads = os.environ.get('OMP_NUM_THREADS')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        os.environ['TEST_NON_INTERACTIVE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

        test_code = f'''
#!/usr/bin/env python3
import os
import sys
import logging
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Performance optimization environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'  # Optimize for modern GPUs
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Asynchronous CUDA launch

# Add project path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clean up GPU memory - enhanced version"""
    try:
        import torch
        import gc
        
        # Perform multiple garbage collections
        for _ in range(3):
            gc.collect()
            
        if torch.cuda.is_available():
            # Clear current device cache
            torch.cuda.empty_cache()
            # Clear IPC cache
            torch.cuda.ipc_collect()
            # Synchronize all streams
            torch.cuda.synchronize()
            logger.info("GPU memory has been deeply cleaned up")
    except Exception as e:
        logger.warning(f"Failed to clean up GPU memory: {{e}}")

def run_scenario_test(model_name: str, scenario: int, dataset: str, limit: int) -> bool:
    """Run tests for a specified scenario"""
    scenario_config = {{
        "name": "{scenario_info['name']}",
        "description": "{scenario_info['description']}",
        "detector_model": "{scenario_info['detector_model']}",
        "use_ground_truth": {scenario_info['use_ground_truth']},
        "use_cache": {scenario_info['use_cache']},
        "cache_source_dir": "{scenario_info['cache_source_dir']}"
    }}
    
    logger.info("="*60)
    logger.info(f"🚀 Starting test")
    logger.info(f"Model: {{model_name}}")
    logger.info(f"Dataset: {{dataset}}")
    logger.info(f"Scenario: {{scenario}} ({{scenario_config['name']}})")  
    logger.info(f"Description: {{scenario_config['description']}}")
    logger.info(f"Number of test cases: {{limit}}")
    logger.info("="*60)
    
    # Simple progress callback
    def simple_progress(current: int, total: int, message: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {{current}}/{{total}} ({{percentage:.1f}}%) - {{message}}")
    
    try:
        # Import evaluator
        from evaluation.benchmark import BenchmarkEvaluator
        
        # Create evaluator, using specified dataset - add performance optimization parameters
        evaluator = BenchmarkEvaluator(
            data_root=dataset,
            use_cache=scenario_config['use_cache'], 
            cache_source_dir=scenario_config['cache_source_dir']
        )
        
        # Set performance optimization parameters (if supported)
        if hasattr(evaluator, 'set_batch_processing'):
            evaluator.set_batch_processing(True)
        if hasattr(evaluator, 'set_memory_optimization'):
            evaluator.set_memory_optimization(True)
        
        # Set progress callback
        evaluator.set_progress_callback(simple_progress)
        
        start_time = time.time()
        
        # Run evaluation
        evaluator.run_evaluation(
            model_name=model_name,
            limit=limit,
            scenario=scenario,
            detector_model=scenario_config['detector_model'],
            use_ground_truth=scenario_config['use_ground_truth']
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ {{model_name}} on {{dataset}} Scenario {{scenario}} test completed - Duration: {{duration:.1f}} seconds")
        return True
        
    except Exception as e:
        logger.error(f"❌ {{model_name}} on {{dataset}} Scenario {{scenario}} test failed: {{str(e)}}")
        return False
    finally:
        clear_gpu_memory()

def main():
    """Main function"""
    model_name = "{model_name}"
    dataset = "{dataset}"
    scenario = {scenario}
    limit = {limit}
    
    # Check environment
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ Model configuration file not found: config/models_config.yaml")
        return False
    
    # Check if dataset exists
    if not os.path.exists(dataset):
        logger.error(f"❌ Test data directory not found: {{dataset}}")
        return False
    
    # Run test
    success = run_scenario_test(model_name, scenario, dataset, limit)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''

        script_filename = f"run_test_{model_name.replace('-', '_')}_{dataset}_{scenario}.py"
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        logger.info(f"🚀 Starting model evaluation...")
        logger.info(f"📦 Model: {model_name}")
        logger.info(f"📊 Dataset: {dataset}")
        logger.info(f"🎯 Scenario: {scenario} ({scenario_info['name']})")
        logger.info(f"📊 Number of test cases: {limit}")
        logger.info(f"🔧 Using GPU: {cuda_visible_devices} (optimized mode)")
        logger.info(f"⚡ Performance optimization: Enabled cache, multi-threading limit, memory optimization")
        
        start_time = time.time()

        result = subprocess.run(
            ['python', script_filename],
            capture_output=True,
            text=True,
            timeout=None
        )
        
        duration = time.time() - start_time

        if os.path.exists(script_filename):
            os.remove(script_filename)
        
        if result.returncode == 0:
            metrics = read_metrics_file_with_dataset(model_name, scenario, dataset)
            
            if not non_interactive:
                logger.info("\n" + "="*60)
                logger.info("🎯 Test completed!")
                logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                
                if metrics:
                    logger.info("\n📊 Performance metrics:")
                    logger.info(f"  Total test cases: {metrics.get('total_cases', 0)}")
                    logger.info(f"  Locate success rate: {metrics.get('locate_success_rate', 0):.1%}")
                    logger.info(f"  Interaction success rate: {metrics.get('interaction_success_rate', 0):.1%}")
                    logger.info(f"  Locate state awareness rate: {metrics.get('state_awareness_rate_locate', 0):.1%}")
                    logger.info(f"  Interaction state awareness rate: {metrics.get('state_awareness_rate_interaction', 0):.1%}")

                    locate_rate = metrics.get('locate_success_rate', 0)
                    interact_rate = metrics.get('interaction_success_rate', 0)
                    total_success_rate = (locate_rate + interact_rate) / 2
                    
                    logger.info(f"\n🎯 Overall success rate: {total_success_rate:.1%}")
                    logger.info(f"📊 Dataset: {dataset}")
                    logger.info(f"\n✅ Scenario {scenario} test completed, enhanced effects reflected in success rate")
                else:
                    logger.warning("⚠️  Could not read metrics file, but test is completed")
                
                logger.info("="*60)
            else:
                print(f"✅ {model_name} on {dataset} Scenario {scenario} test completed - Duration: {duration:.1f} seconds")
            return True
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            if not non_interactive:
                logger.error(f"❌ Test failed: {error_msg}")
            else:
                print(f"❌ {model_name} on {dataset} Scenario {scenario} test failed: {error_msg}")
            return False
        
    except Exception as e:
        if not non_interactive:
            logger.error(f"❌ Test failed: {str(e)}")
        else:
            print(f"❌ {model_name} on {dataset} Scenario {scenario} test failed: {str(e)}")
        return False
    finally:
        if old_cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            
        if old_tokenizers_parallelism is not None:
            os.environ['TOKENIZERS_PARALLELISM'] = old_tokenizers_parallelism
        else:
            os.environ.pop('TOKENIZERS_PARALLELISM', None)
            
        if old_omp_threads is not None:
            os.environ['OMP_NUM_THREADS'] = old_omp_threads
        else:
            os.environ.pop('OMP_NUM_THREADS', None)
            
        os.environ.pop('MKL_NUM_THREADS', None)
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
        
        if 'script_filename' in locals() and os.path.exists(script_filename):
            os.remove(script_filename)
            
        clear_gpu_memory()

def run_single_model_test(model_name: str, dataset: str, limit: int, gpu_id: int, detector_key: str = "gpt4o") -> bool:
    logger.info(f"🔄 Starting test: {model_name} on {dataset} (GPU {gpu_id}) using {SCENARIO2_CONFIGS[detector_key]['name']}")
    success = run_scenario_test_with_dataset(model_name, 2, dataset, limit, gpu_id, detector_key)
    
    if success:
        logger.info(f"✅ {model_name} on {dataset} (GPU {gpu_id}) test completed")
    else:
        logger.error(f"❌ {model_name} on {dataset} (GPU {gpu_id}) test failed")
    
    return success

def run_offline_scenario2_on_dataset(dataset: str, limit: int, detector_key: str = "gpt4o"):
    print("\n" + "="*60)
    print(f"🚀 Offline Scenario 2 Test - {dataset}")
    print("📋 Test Configuration:")
    print(f"  Models: {', '.join(OFFLINE_SCENARIO2_MODELS)}")
    print(f"  Dataset: {dataset}")
    print(f"  Scenario: 2 ({SCENARIO2_CONFIGS[detector_key]['name']})")
    print(f"  Detector: {SCENARIO2_CONFIGS[detector_key]['detector_model']}")
    print(f"  Number of samples: {limit}")
    print(f"  Parallel execution: ShowUI-2B(GPU 6), OS-Atlas-7B(GPU 7)")
    print("="*60)
    
    print("\n🔍 Checking existing result files...")
    existing_results = list_existing_results(dataset, 2)
    
    if existing_results:
        print(f"\n📊 Found completed model results:")
        for model_name, info in existing_results.items():
            print(f"  ✅ {model_name}: Completed at {info['modification_time']}")
            print(f"     Locate success rate: {info['locate_success_rate']:.1%}, Interaction success rate: {info['interaction_success_rate']:.1%}")
        
        print(f"\n⚠️  Checkpoint protection: Found {len(existing_results)} completed models")
        rerun_choice = input("Do you want to re-run completed models? (y/N): ").strip().lower()
        
        if rerun_choice not in ['y', 'yes']:
            print("✅ Skipping completed models, only running incomplete models")
        else:
            print("🔄 Re-running all models")
            existing_results = {} 
    else:
        print("📭 No completed results found, will run all models")

    print(f"\n{'='*60}")
    if existing_results:
        remaining_models = [m for m in OFFLINE_SCENARIO2_MODELS if m not in existing_results]
        if remaining_models:
            print(f"📋 Models to be run: {', '.join(remaining_models)}")
            confirm = input("Confirm to start testing incomplete models? (y/N): ").strip().lower()
        else:
            print("🎉 All models are already completed!")
            return
    else:
        confirm = input("Confirm to start testing all models? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print("❌ Test cancelled")
        return

    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ Model configuration file not found: config/models_config.yaml")
        return

    if not os.path.exists(dataset):
        logger.error(f"❌ Dataset directory not found: {dataset}")
        return
    
    remaining_models = [m for m in OFFLINE_SCENARIO2_MODELS if m not in existing_results]
    
    if not remaining_models:
        logger.info("🎉 All models are already completed, no need to run!")
        return

    required_gpus = []
    for model_name in remaining_models:
        if model_name == "ShowUI-2B":
            required_gpus.append(6)
        elif model_name == "OS-Atlas-Base-7B":
            required_gpus.append(7)
    
    logger.info(f"🎯 Required GPUs: {required_gpus}")
    wait_for_gpu_availability(required_gpus)
    
    total_tests = len(remaining_models)
    total_original = len(OFFLINE_SCENARIO2_MODELS)
    completed_already = len(existing_results)
    logger.info(f"🚀 Starting parallel tests, {total_tests} models to be run (completed: {completed_already}/{total_original})")

    tasks = []
    for model_name in OFFLINE_SCENARIO2_MODELS:
        if model_name in remaining_models:
            gpu_id = 6 if model_name == "ShowUI-2B" else 7
            tasks.append((model_name, dataset, limit, gpu_id))
            logger.info(f"📋 Preparing task: {model_name} → GPU {gpu_id}")
    
    completed_tests = completed_already  
    failed_tests = 0

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_to_task = {
            executor.submit(run_single_model_test, model_name, dataset, limit, gpu_id, detector_key): (model_name, gpu_id)
            for model_name, dataset, limit, gpu_id in tasks
        }
        
        for future in as_completed(future_to_task):
            model_name, gpu_id = future_to_task[future]
            try:
                success = future.result()
                if success:
                    completed_tests += 1
                else:
                    failed_tests += 1
            except Exception as e:
                logger.error(f"❌ {model_name} (GPU {gpu_id}) execution exception: {e}")
                failed_tests += 1

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Parallel test completion statistics - {dataset}:")
    logger.info(f"  Total models: {total_original}")
    logger.info(f"  Completed: {completed_tests}")
    logger.info(f"  Models run in this run: {len(tasks)}")
    logger.info(f"  Failed tests: {failed_tests}")
    logger.info(f"  Overall success rate: {completed_tests/total_original*100:.1f}%")

    logger.info(f"\n📁 Result saving path:")
    result_base_dir = f"results/{dataset}_scenario2"
    for model_name in OFFLINE_SCENARIO2_MODELS:
        model_dir = f"{result_base_dir}/{model_name}"
        metrics_file = f"{model_dir}/{model_name}_metrics.json"
        if os.path.exists(metrics_file):
            logger.info(f"  ✅ {model_name}: {metrics_file}")
        else:
            logger.info(f"  ❌ {model_name}: {metrics_file} (Incomplete)")
    
    logger.info(f"{'='*80}")
    
    if completed_tests == total_original:
        logger.info(f"🎉 All models in {dataset} test completed successfully!")
    elif failed_tests > 0:
        logger.warning(f"⚠️  {dataset} has {failed_tests} failed tests")
    else:
        logger.info(f"✅ {len(tasks)} models in {dataset} test completed this run!")

def main():

    print("=" * 60)
    print("🎯 SpiderBench Offline Scenario 2 Test Tool")
    print("📦 Test Models: ShowUI-2B, OS-Atlas-7B")
    print("🎯 Test Scenario: 2 (Component Detection Enhancement)")
    print("🔧 Specified GPU: 6,7")
    print("=" * 60)
    
    if not TQDM_AVAILABLE:
        print("💡 Hint: Installing the tqdm library can display a better progress bar:")
        print("   pip install tqdm")
        print()
    
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ Model configuration file not found: config/models_config.yaml")
        return
    
    try:
        detector_key = select_detector_model()
        
        dataset, limit = select_dataset()
        
        run_offline_scenario2_on_dataset(dataset, limit, detector_key)
            
    except KeyboardInterrupt:
        print("\n❌ User interrupted test")
    except Exception as e:
        logger.error(f"❌ Program exception: {str(e)}")

if __name__ == "__main__":
    main()