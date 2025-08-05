#!/bin/bash

# Proxy configuration function
setup_proxy() {
    echo "Configuring proxy..." | tee -a $LOG_FILE
    
    # Start proxy
    clashon
    sleep 2
    
    # Set global mode
    curl -X PATCH http://127.0.0.1:9090/configs -d '{"mode": "Global"}' 2>/dev/null
    sleep 1
    
    # Get available nodes
    echo "Getting available proxy nodes..." | tee -a $LOG_FILE
    available_proxies=$(curl -s http://127.0.0.1:9090/proxies/GLOBAL | python3 -c "
import json, sys
data = json.load(sys.stdin)
proxies = data.get('all', [])
for proxy in proxies[:5]:  # Display first 5 nodes
    print(f'Available node: {proxy}')
# Use the first available node
if proxies:
    print(f'Using node: {proxies[0]}')
    sys.exit(0)
else:
    sys.exit(1)
")
    
    # Set proxy node (using the first available node)
    first_proxy=$(curl -s http://127.0.0.1:9090/proxies/GLOBAL | python3 -c "
import json, sys
data = json.load(sys.stdin)
proxies = data.get('all', [])
if proxies:
    print(proxies[0])
")
    
    if [ ! -z "$first_proxy" ]; then
        curl -X PUT http://127.0.0.1:9090/proxies/GLOBAL -d "{\"name\": \"$first_proxy\"}" 2>/dev/null
        echo "Proxy set: $first_proxy" | tee -a $LOG_FILE
        
        # Set environment variables
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export HTTPS_PROXY=http://127.0.0.1:7890
    else
        echo "Warning: Could not get proxy node, continuing with direct connection" | tee -a $LOG_FILE
    fi
}

# Check network connection
check_network() {
    echo "Checking network connection..." | tee -a $LOG_FILE
    if ! curl -s --connect-timeout 10 https://www.google.com > /dev/null; then
        echo "Network connection failed, attempting to configure proxy..." | tee -a $LOG_FILE
        setup_proxy
        
        # Check again
        if ! curl -s --connect-timeout 10 https://www.google.com > /dev/null; then
            echo "Proxy configuration failed, please check network settings" | tee -a $LOG_FILE
        else
            echo "Proxy configured successfully" | tee -a $LOG_FILE
        fi
    else
        echo "Network connection normal" | tee -a $LOG_FILE
    fi
}

# Set log file
LOG_FILE="benchmark_mobile_test.log"
echo "Starting test $(date)" > $LOG_FILE

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check and configure network
check_network

# Interactive model selection
select_model() {
    echo "Please select the model to test:"
    echo "1) google/gemini-2.5-flash"
    echo "2) google/gemini-2.5-pro"
    echo "3) anthropic/claude-3.5-sonnet"
    echo "4) openai/gpt-4o-2024-11-20"
    echo "5) Exit"
    
    while true; do
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1)
                SELECTED_MODEL="google/gemini-2.5-flash"
                echo "✅ Selected: $SELECTED_MODEL"
                break
                ;;
            2)
                SELECTED_MODEL="google/gemini-2.5-pro"
                echo "✅ Selected: $SELECTED_MODEL"
                break
                ;;
            3)
                SELECTED_MODEL="anthropic/claude-3.5-sonnet"
                echo "✅ Selected: $SELECTED_MODEL"
                break
                ;;
            4)
                SELECTED_MODEL="openai/gpt-4o-2024-11-20"
                echo "✅ Selected: $SELECTED_MODEL"
                break
                ;;
            5)
                echo "Exiting script..."
                exit 0
                ;;
            *)
                echo "❌ Invalid choice, please enter 1-5"
                ;;
        esac
    done
}

# Select model
select_model

# Define models to test
MODELS=("$SELECTED_MODEL")

# Set progress file based on selected model
MODEL_SAFE_NAME=$(echo "$SELECTED_MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
PROGRESS_FILE="evaluation_progress_mobile_${MODEL_SAFE_NAME}.json"

echo "Current model: $SELECTED_MODEL" | tee -a $LOG_FILE
echo "Progress file: $PROGRESS_FILE" | tee -a $LOG_FILE

if [ -f "$PROGRESS_FILE" ]; then
    echo "Progress file found for this model, resuming from checkpoint..." | tee -a $LOG_FILE
else
    echo "Progress file not found for this model, starting fresh evaluation..." | tee -a $LOG_FILE
fi

# Run Python script for evaluation
echo "Starting online model evaluation..." | tee -a $LOG_FILE

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR"
export TEST_NON_INTERACTIVE=1
export SELECTED_MODEL="$SELECTED_MODEL"
export PROGRESS_FILE="$PROGRESS_FILE"

# Create Python evaluation script
cat > run_online_evaluation.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import logging
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Progress file path - will be passed by shell script
PROGRESS_FILE = os.environ.get('PROGRESS_FILE', 'evaluation_progress.json')

def load_progress():
    """Load progress file"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
    return {}

def save_progress(progress_data):
    """Save progress to file"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save progress file: {e}")

def is_task_completed(model_name, scenario, progress_data):
    """Check if task is completed"""
    task_key = f"{model_name}_scenario_{scenario}"
    return progress_data.get(task_key, {}).get('completed', False)

def mark_task_completed(model_name, scenario, progress_data):
    """Mark task as completed"""
    task_key = f"{model_name}_scenario_{scenario}"
    progress_data[task_key] = {
        'completed': True,
        'completion_time': datetime.now().isoformat(),
        'model': model_name,
        'scenario': scenario
    }
    save_progress(progress_data)

def clear_gpu_memory():
    """Clear GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def run_scenario_test(model_name: str, scenario: int, limit: int = 650) -> bool:
    """Run test for a specific scenario"""
    scenario_configs = {
        1: {
            "name": "Baseline Test",
            "description": "Direct model evaluation, no enhancement",
            "detector_model": None,
            "use_ground_truth": False,
            "use_cache": False,
            "cache_source_dir": None
        },
        2: {
            "name": "Component Detection Enhancement (GPT-4o)", 
            "description": "Enhanced prompt using GPT-4o for component detection",
            "detector_model": "openai/gpt-4o-2024-11-20",
            "use_ground_truth": False,
            "use_cache": True,
            "cache_source_dir": "openai/gpt-4o-2024-11-20_bygpt-4o-2024-11-20"
        },
        3: {
            "name": "Full Enhancement",
            "description": "Using component detector and real bounding boxes",
            "detector_model": "google/gemini-2.5-flash", 
            "use_ground_truth": True,
            "use_cache": True,
            "cache_source_dir": "google/gemini-2.5-flash_bygemini-2.5-flash_groundtruth"
        }
    }
    
    scenario_info = scenario_configs[scenario]
    
    logger.info("="*60)
    logger.info(f"🚀 Starting test")
    logger.info(f"Model: {model_name}")
    logger.info(f"Scenario: {scenario} ({scenario_info['name']})")
    logger.info(f"Description: {scenario_info['description']}")
    logger.info(f"Test cases: {limit}")
    logger.info("="*60)
    
    # Simple progress callback
    def simple_progress(current: int, total: int, message: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) - {message}")
    
    try:
        # Import evaluator
        from evaluation.benchmark import BenchmarkEvaluator
        
        # Create evaluator based on scenario, using mobile_en data
        if scenario == 1:
            evaluator = BenchmarkEvaluator(data_root="mobile_en", result_prefix="mobile")
        elif scenario == 2:
            evaluator = BenchmarkEvaluator(
                data_root="mobile_en",
                use_cache=scenario_info['use_cache'], 
                cache_source_dir=scenario_info['cache_source_dir'],
                result_prefix="mobile"
            )
        elif scenario == 3:
            evaluator = BenchmarkEvaluator(
                data_root="mobile_en",
                use_cache=scenario_info['use_cache'], 
                cache_source_dir=scenario_info['cache_source_dir'],
                result_prefix="mobile"
            )
        
        # Set progress callback
        evaluator.set_progress_callback(simple_progress)
        
        start_time = time.time()
        
        # Run evaluation
        evaluator.run_evaluation(
            model_name=model_name,
            limit=limit,
            scenario=scenario,
            detector_model=scenario_info['detector_model'],
            use_ground_truth=scenario_info['use_ground_truth']
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ {model_name} Scenario {scenario} test completed - Duration: {duration:.1f} seconds")
        
        # Mark task as completed
        progress_data = load_progress()
        mark_task_completed(model_name, scenario, progress_data)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name} Scenario {scenario} test failed: {str(e)}")
        return False
    finally:
        clear_gpu_memory()

def main():
    """Main function (parallel, supports checkpoint recovery)"""
    # Get model name from environment variables
    selected_model = os.environ.get('SELECTED_MODEL', 'google/gemini-2.5-flash')
    models = [selected_model]
    
    # Set scenarios based on model type
    if selected_model.startswith('google/gemini'):
        scenarios = [1, 2]  # Gemini models run scenarios 1 and 2
        max_workers = min(2, os.cpu_count())  # Two scenarios can run in parallel
    else:
        scenarios = [1]  # Claude and GPT only run scenario 1
        max_workers = 1  # Single scenario, single process
    
    limit = 650

    # Check environment
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ Model configuration file not found: config/models_config.yaml")
        return False
    
    # Check if mobile_en data exists
    if not os.path.exists("mobile_en"):
        logger.error("❌ Test data directory not found: mobile_en")
        return False

    # Load progress data
    progress_data = load_progress()
    
    # Filter out completed tasks
    all_tasks = [(m, s) for m in models for s in scenarios]
    pending_tasks = [(m, s) for m, s in all_tasks if not is_task_completed(m, s, progress_data)]
    
    total_tests = len(all_tasks)
    completed_tests = total_tests - len(pending_tasks)
    success_count = completed_tests

    if completed_tests > 0:
        logger.info(f"⏭️  Found completed tasks: {completed_tests}/{total_tests}")
        logger.info(f"📝 Skipping completed tasks, continuing with {len(pending_tasks)} remaining tasks")
    else:
        logger.info(f"🚀 Starting fresh evaluation, total {total_tests} tasks")

    if not pending_tasks:
        logger.info("🎉 All tasks completed!")
        return True

    logger.info(f"Executing {len(pending_tasks)} remaining tasks")

    # Checkpoint protection: add confirmation mechanism
    if len(pending_tasks) > 0:
        logger.info("⚠️  Checkpoint protection: model evaluation about to begin")
        logger.info(f"Model: {pending_tasks[0][0]}")
        logger.info(f"Scenario: {pending_tasks[0][1]}")
        logger.info("If you want to stop, press Ctrl+C within 5 seconds")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("User interrupted, exiting safely")
            return False

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(run_scenario_test, m, s, limit): (m, s)
            for m, s in pending_tasks
        }

        for future in as_completed(future_map):
            m, s = future_map[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                    logger.info(f"✅ {m} Scenario {s} completed")
                else:
                    logger.error(f"❌ {m} Scenario {s} failed")
            except Exception as e:
                logger.error(f"❌ {m} Scenario {s} exception: {e}")

    final_success_count = success_count
    logger.info(f"📊 Evaluation statistics:")
    logger.info(f"    Total tasks: {total_tests}")
    logger.info(f"    Completed: {final_success_count}")
    logger.info(f"    Success rate: {final_success_count/total_tests*100:.1f}%")
    
    return final_success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Run Python evaluation script
python run_online_evaluation.py

# Check evaluation results
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully $(date)" | tee -a $LOG_FILE
    echo "Progress file saved: $PROGRESS_FILE" | tee -a $LOG_FILE
else
    echo "❌ Error during evaluation $(date)" | tee -a $LOG_FILE
    echo "Progress saved, can re-run script to continue execution" | tee -a $LOG_FILE
fi

# Record completion time
echo "Test completed $(date)" | tee -a $LOG_FILE

# Clean up temporary files
rm -f run_online_evaluation.py