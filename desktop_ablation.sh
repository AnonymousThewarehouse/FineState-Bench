#!/bin/bash

# Desktop ablation experiment script - only run Description-only and Coordinate-only experiments

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
for proxy in proxies[:5]:  # Display the first 5 nodes
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
        
        # Re-check
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
LOG_FILE="benchmark_desktop_ablation.log"
echo "Starting Desktop ablation experiment $(date)" > $LOG_FILE

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check and configure network
check_network

# Fixed use of gemini-2.5-flash model
SELECTED_MODEL="google/gemini-2.5-flash"
echo "Using model: $SELECTED_MODEL" | tee -a $LOG_FILE

# Set progress file based on selected model
MODEL_SAFE_NAME=$(echo "$SELECTED_MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
PROGRESS_FILE="evaluation_progress_desktop_ablation_${MODEL_SAFE_NAME}.json"

echo "Progress file: $PROGRESS_FILE" | tee -a $LOG_FILE

if [ -f "$PROGRESS_FILE" ]; then
    echo "Progress file found, resuming from checkpoint..." | tee -a $LOG_FILE
else
    echo "Progress file not found, starting fresh evaluation..." | tee -a $LOG_FILE
fi

# Run Python script for evaluation
echo "Starting ablation experiment..." | tee -a $LOG_FILE

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR"
export TEST_NON_INTERACTIVE=1
export SELECTED_MODEL="$SELECTED_MODEL"
export PROGRESS_FILE="$PROGRESS_FILE"

# Create Python evaluation script
cat > run_desktop_ablation.py << 'EOF'
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
PROGRESS_FILE = os.environ.get('PROGRESS_FILE', 'evaluation_progress_desktop_ablation.json')

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

def is_task_completed(model_name, ablation_type, progress_data):
    """Check if task is completed"""
    task_key = f"{model_name}_ablation_{ablation_type}"
    return progress_data.get(task_key, {}).get('completed', False)

def mark_task_completed(model_name, ablation_type, progress_data):
    """Mark task as completed"""
    task_key = f"{model_name}_ablation_{ablation_type}"
    progress_data[task_key] = {
        'completed': True,
        'completion_time': datetime.now().isoformat(),
        'model': model_name,
        'ablation_type': ablation_type
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

def run_ablation_test(model_name: str, ablation_type: str, limit: int = 821) -> bool:
    """Run ablation test of a specified type"""
    ablation_configs = {
        "description_only": {
            "name": "Description ablation experiment",
            "description": "Only use component description, no coordinate information",
            "use_cache": False,
            "cache_source_dir": None
        },
        "coordinate_only": {
            "name": "Coordinate ablation experiment", 
            "description": "Only use coordinate prediction, no description information",
            "use_cache": False,
            "cache_source_dir": None
        }
    }
    
    ablation_info = ablation_configs[ablation_type]
    
    logger.info("="*60)
    logger.info(f"🚀 Starting ablation experiment")
    logger.info(f"Model: {model_name}")
    logger.info(f"Type: {ablation_type} ({ablation_info['name']})")
    logger.info(f"Description: {ablation_info['description']}")
    logger.info(f"Number of test cases: {limit}")
    logger.info("="*60)
    
    # Simple progress callback
    def simple_progress(current: int, total: int, message: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) - {message}")
    
    try:
        # Import evaluator
        from evaluation.benchmark import BenchmarkEvaluator
        
        # Create evaluator, using desktop_en data
        evaluator = BenchmarkEvaluator(
            data_root="desktop_en",
            use_cache=ablation_info['use_cache'], 
            cache_source_dir=ablation_info['cache_source_dir']
        )
        
        # Set progress callback
        evaluator.set_progress_callback(simple_progress)
        
        start_time = time.time()
        
        # Run evaluation - using scenario 2 (component detection enhancement) settings
        evaluator.run_evaluation(
            model_name=model_name,
            limit=limit,
            scenario=2,  # Use scenario 2 framework
            detector_model="google/gemini-2.5-flash",
            use_ground_truth=False,
            ablation_type=ablation_type  # Pass ablation type
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ {model_name} {ablation_type} test completed - Duration: {duration:.1f} seconds")
        
        # Mark task as completed
        progress_data = load_progress()
        mark_task_completed(model_name, ablation_type, progress_data)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name} {ablation_type} test failed: {str(e)}")
        return False
    finally:
        clear_gpu_memory()

def main():
    """Main function (ablation version)"""
    # Get model name from environment variables
    selected_model = os.environ.get('SELECTED_MODEL', 'google/gemini-2.5-flash')
    models = [selected_model]
    
    # Ablation experiment types
    ablation_types = ["description_only", "coordinate_only"]
    
    limit = 821  # Total number of desktop_en dataset

    # Check environment
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ Model configuration file not found: config/models_config.yaml")
        return False
    
    # Check if desktop_en data exists
    if not os.path.exists("desktop_en"):
        logger.error("❌ Test data directory not found: desktop_en")
        return False

    # Load progress data
    progress_data = load_progress()
    
    # Filter out completed tasks
    all_tasks = [(m, a) for m in models for a in ablation_types]
    pending_tasks = [(m, a) for m, a in all_tasks if not is_task_completed(m, a, progress_data)]
    
    total_tests = len(all_tasks)
    completed_tests = total_tests - len(pending_tasks)
    success_count = completed_tests

    if completed_tests > 0:
        logger.info(f"⏭️  Found completed tasks: {completed_tests}/{total_tests}")
        logger.info(f"📝 Skipping completed tasks, continuing with {len(pending_tasks)} tasks")
    else:
        logger.info(f"🚀 Starting fresh ablation experiment, total {total_tests} tasks")

    if not pending_tasks:
        logger.info("🎉 All ablation experiments completed!")
        return True

    logger.info(f"Executing {len(pending_tasks)} remaining tasks")

    # Checkpoint protection: add confirmation mechanism
    if len(pending_tasks) > 0:
        logger.info("⚠️  Checkpoint protection: ablation experiment about to begin")
        logger.info(f"Model: {pending_tasks[0][0]}")
        logger.info(f"Ablation type: {pending_tasks[0][1]}")
        logger.info("If you want to stop, press Ctrl+C within 5 seconds")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("User interrupted, exiting safely")
            return False

    # Run ablation experiments sequentially (to avoid cache conflicts)
    for m, a in pending_tasks:
        try:
            ok = run_ablation_test(m, a, limit)
            if ok:
                success_count += 1
                logger.info(f"✅ {m} {a} completed")
            else:
                logger.error(f"❌ {m} {a} failed")
        except Exception as e:
            logger.error(f"❌ {m} {a} exception: {e}")

    final_success_count = success_count
    logger.info(f"📊 Ablation experiment completion statistics:")
    logger.info(f"    Total tasks: {total_tests}")
    logger.info(f"    Completed: {final_success_count}")
    logger.info(f"    Success rate: {final_success_count/total_tests*100:.1f}%")
    
    return final_success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Run Python evaluation script
python run_desktop_ablation.py

# Check evaluation results
if [ $? -eq 0 ]; then
    echo "✅ Desktop ablation experiment completed successfully $(date)" | tee -a $LOG_FILE
    echo "Progress file saved: $PROGRESS_FILE" | tee -a $LOG_FILE
else
    echo "❌ Error during Desktop ablation experiment $(date)" | tee -a $LOG_FILE
    echo "Progress saved, can re-run script to continue execution" | tee -a $LOG_FILE
fi

# Record completion time
echo "Desktop ablation experiment completed $(date)" | tee -a $LOG_FILE

# Clean up temporary files
rm -f run_desktop_ablation.py