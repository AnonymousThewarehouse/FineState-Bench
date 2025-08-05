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
        "name": "组件检测增强(GPT-4o)",
        "description": "使用GPT-4o缓存的增强prompt",
        "detector_model": "openai/gpt-4o-2024-11-20",
        "use_ground_truth": False,
        "use_cache": True,
        "cache_source_dir": "openai/gpt-4o-2024-11-20_bygpt-4o-2024-11-20"
    },
    "gemini": {
        "name": "组件检测增强(Gemini-2.5-Flash)",
        "description": "使用Gemini-2.5-Flash缓存的增强prompt",
        "detector_model": "google/gemini-2.5-flash",
        "use_ground_truth": False,
        "use_cache": True,
        "cache_source_dir": "google/gemini-2.5-flash_bygemini-2.5-flash"
    }
}

def clear_gpu_memory():
    """清理GPU内存 - 仅在GPU 6和7上"""
    try:
        cleanup_code = """
import torch
import gc
import os

# 多次垃圾回收
for _ in range(3):
    gc.collect()
    
if torch.cuda.is_available():
    # 清理所有可见GPU的缓存
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    
# 强制释放未使用的缓存池内存
torch.cuda.empty_cache()
"""
        subprocess.run([
            'python', '-c', cleanup_code
        ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': '6,7'}, 
        capture_output=True, timeout=30)
        logger.info("GPU 6,7 内存已积极清理")
    except Exception as e:
        logger.warning(f"清理GPU内存失败: {e}")

def check_model_completion(model_name: str, scenario: int, dataset: str) -> bool:
    result_dir = f"results/{dataset}_scenario{scenario}" if scenario == 2 else f"results/offline_scenario{scenario}"
    metrics_path = f"{result_dir}/{model_name}/{model_name}_metrics.json"
    
    if os.path.exists(metrics_path):
        logger.info(f"✅ 发现已完成的结果: {metrics_path}")
        return True
    else:
        logger.info(f"❌ 未找到结果文件: {metrics_path}")
        return False

def list_existing_results(dataset: str, scenario: int) -> dict:
    result_dir = f"results/{dataset}_scenario{scenario}" if scenario == 2 else f"results/offline_scenario{scenario}"
    existing_results = {}
    
    logger.info(f"🔍 检查结果目录: {result_dir}")
    
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
                    logger.info(f"✅ {model_name}: 已完成 (修改时间: {mod_time_str})")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name}: 结果文件存在但读取失败: {e}")
            else:
                logger.info(f"❌ {model_name}: 未完成")
    else:
        logger.info(f"📁 结果目录不存在，将创建: {result_dir}")
    
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
    logger.info(f"🔍 检查GPU {gpu_list}的可用性...")
    
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
                    logger.info(f"GPU {gpu_id}: 使用率 {usage_percent:.1f}%, 空闲内存 {free_gb:.1f}GB")

                    if usage_percent < 70 and free_gb > 8:
                        gpus_available += 1
            
            if gpus_available >= len(gpu_ids):
                logger.info(f"✅ GPU {gpu_list}都可用，开始测试")
                break
            else:
                logger.info(f"⏳ 等待GPU可用 (当前可用: {gpus_available}/{len(gpu_ids)})，60秒后重试...")
                time.sleep(60)
                
        except Exception as e:
            logger.warning(f"检查GPU状态失败: {e}，30秒后重试...")
            time.sleep(30)

def select_detector_model():
    print("\n" + "="*50)
    print("请选择边框预测模型:")
    print("  1. GPT-4o (openai/gpt-4o-2024-11-20)")
    print("  2. Gemini-2.5-Flash (google/gemini-2.5-flash)")
    print("="*50)
    
    while True:
        try:
            choice = int(input("请输入模型编号 (1-2): "))
            if choice == 1:
                return "gpt4o"
            elif choice == 2:
                return "gemini"
            else:
                print("❌ 无效选择，请输入 1-2")
        except ValueError:
            print("❌ 请输入有效数字")

def select_dataset():
    print("\n" + "="*50)
    print("请选择数据集:")
    print("  1. Desktop (desktop_en) - 628个样本")
    print("  2. Mobile (mobile_en) - 648个样本")
    print("  3. Web (web_en) - 697个样本")
    print("="*50)
    
    while True:
        try:
            choice = int(input("请输入数据集编号 (1-3): "))
            if choice == 1:
                return "desktop_en", 628
            elif choice == 2:
                return "mobile_en", 648
            elif choice == 3:
                return "web_en", 697
            else:
                print("❌ 无效选择，请输入 1-3")
        except ValueError:
            print("❌ 请输入有效数字")

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
        logger.info(f"🚀 开始测试")
        logger.info(f"模型: {model_name}")
        logger.info(f"数据集: {dataset}")
        logger.info(f"场景: {scenario} ({scenario_info['name']})")
        logger.info(f"描述: {scenario_info['description']}")
        logger.info(f"测试用例数: {limit}")
        logger.info(f"🎯 指定GPU: {cuda_visible_devices}")
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

# 性能优化环境变量设置
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告
os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
os.environ['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'  # 针对现代GPU优化
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步CUDA启动

# 添加项目路径
sys.path.append(os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """清理GPU内存 - 增强版"""
    try:
        import torch
        import gc
        
        # 多次垃圾回收
        for _ in range(3):
            gc.collect()
            
        if torch.cuda.is_available():
            # 清理当前设备缓存
            torch.cuda.empty_cache()
            # 清理IPC缓存
            torch.cuda.ipc_collect()
            # 同步所有流
            torch.cuda.synchronize()
            logger.info("GPU内存已深度清理")
    except Exception as e:
        logger.warning(f"清理GPU内存失败: {{e}}")

def run_scenario_test(model_name: str, scenario: int, dataset: str, limit: int) -> bool:
    """运行指定场景的测试"""
    scenario_config = {{
        "name": "{scenario_info['name']}",
        "description": "{scenario_info['description']}",
        "detector_model": "{scenario_info['detector_model']}",
        "use_ground_truth": {scenario_info['use_ground_truth']},
        "use_cache": {scenario_info['use_cache']},
        "cache_source_dir": "{scenario_info['cache_source_dir']}"
    }}
    
    logger.info("="*60)
    logger.info(f"🚀 开始测试")
    logger.info(f"模型: {{model_name}}")
    logger.info(f"数据集: {{dataset}}")
    logger.info(f"场景: {{scenario}} ({{scenario_config['name']}})")  
    logger.info(f"描述: {{scenario_config['description']}}")
    logger.info(f"测试用例数: {{limit}}")
    logger.info("="*60)
    
    # 简单进度回调
    def simple_progress(current: int, total: int, message: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"进度: {{current}}/{{total}} ({{percentage:.1f}}%) - {{message}}")
    
    try:
        # 导入评估器
        from evaluation.benchmark import BenchmarkEvaluator
        
        # 创建评估器，使用指定数据集 - 添加性能优化参数
        evaluator = BenchmarkEvaluator(
            data_root=dataset,
            use_cache=scenario_config['use_cache'], 
            cache_source_dir=scenario_config['cache_source_dir']
        )
        
        # 设置性能优化参数（如果支持）
        if hasattr(evaluator, 'set_batch_processing'):
            evaluator.set_batch_processing(True)
        if hasattr(evaluator, 'set_memory_optimization'):
            evaluator.set_memory_optimization(True)
        
        # 设置进度回调
        evaluator.set_progress_callback(simple_progress)
        
        start_time = time.time()
        
        # 运行评估
        evaluator.run_evaluation(
            model_name=model_name,
            limit=limit,
            scenario=scenario,
            detector_model=scenario_config['detector_model'],
            use_ground_truth=scenario_config['use_ground_truth']
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ {{model_name}} on {{dataset}} 场景{{scenario}} 测试完成 - 耗时: {{duration:.1f}}秒")
        return True
        
    except Exception as e:
        logger.error(f"❌ {{model_name}} on {{dataset}} 场景{{scenario}} 测试失败: {{str(e)}}")
        return False
    finally:
        clear_gpu_memory()

def main():
    """主函数"""
    model_name = "{model_name}"
    dataset = "{dataset}"
    scenario = {scenario}
    limit = {limit}
    
    # 检查环境
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ 找不到模型配置文件: config/models_config.yaml")
        return False
    
    # 检查数据集是否存在
    if not os.path.exists(dataset):
        logger.error(f"❌ 找不到测试数据目录: {{dataset}}")
        return False
    
    # 运行测试
    success = run_scenario_test(model_name, scenario, dataset, limit)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''

        script_filename = f"run_test_{model_name.replace('-', '_')}_{dataset}_{scenario}.py"
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        logger.info(f"🚀 开始模型评估...")
        logger.info(f"📦 模型: {model_name}")
        logger.info(f"📊 数据集: {dataset}")
        logger.info(f"🎯 场景: {scenario} ({scenario_info['name']})")
        logger.info(f"📊 测试用例数: {limit}")
        logger.info(f"🔧 使用GPU: {cuda_visible_devices} (优化模式)")
        logger.info(f"⚡ 性能优化: 启用缓存, 多线程限制, 内存优化")
        
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
                logger.info("🎯 测试完成!")
                logger.info(f"⏱️  耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
                
                if metrics:
                    logger.info("\n📊 性能指标:")
                    logger.info(f"  总测试用例: {metrics.get('total_cases', 0)}")
                    logger.info(f"  定位成功率: {metrics.get('locate_success_rate', 0):.1%}")
                    logger.info(f"  交互成功率: {metrics.get('interaction_success_rate', 0):.1%}")
                    logger.info(f"  定位状态感知率: {metrics.get('state_awareness_rate_locate', 0):.1%}")
                    logger.info(f"  交互状态感知率: {metrics.get('state_awareness_rate_interaction', 0):.1%}")

                    locate_rate = metrics.get('locate_success_rate', 0)
                    interact_rate = metrics.get('interaction_success_rate', 0)
                    total_success_rate = (locate_rate + interact_rate) / 2
                    
                    logger.info(f"\n🎯 总体成功率: {total_success_rate:.1%}")
                    logger.info(f"📊 数据集: {dataset}")
                    logger.info(f"\n✅ 场景{scenario}测试完成，增强效果已体现在成功率中")
                else:
                    logger.warning("⚠️  无法读取metrics文件，但测试已完成")
                
                logger.info("="*60)
            else:
                print(f"✅ {model_name} on {dataset} 场景{scenario} 测试完成 - 耗时: {duration:.1f}秒")
            return True
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            if not non_interactive:
                logger.error(f"❌ 测试失败: {error_msg}")
            else:
                print(f"❌ {model_name} on {dataset} 场景{scenario} 测试失败: {error_msg}")
            return False
        
    except Exception as e:
        if not non_interactive:
            logger.error(f"❌ 测试失败: {str(e)}")
        else:
            print(f"❌ {model_name} on {dataset} 场景{scenario} 测试失败: {str(e)}")
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
    logger.info(f"🔄 开始测试: {model_name} on {dataset} (GPU {gpu_id}) 使用 {SCENARIO2_CONFIGS[detector_key]['name']}")
    success = run_scenario_test_with_dataset(model_name, 2, dataset, limit, gpu_id, detector_key)
    
    if success:
        logger.info(f"✅ {model_name} on {dataset} (GPU {gpu_id}) 测试完成")
    else:
        logger.error(f"❌ {model_name} on {dataset} (GPU {gpu_id}) 测试失败")
    
    return success

def run_offline_scenario2_on_dataset(dataset: str, limit: int, detector_key: str = "gpt4o"):
    print("\n" + "="*60)
    print(f"🚀 离线场景2测试 - {dataset}")
    print("📋 测试配置:")
    print(f"  模型: {', '.join(OFFLINE_SCENARIO2_MODELS)}")
    print(f"  数据集: {dataset}")
    print(f"  场景: 2 ({SCENARIO2_CONFIGS[detector_key]['name']})")
    print(f"  检测器: {SCENARIO2_CONFIGS[detector_key]['detector_model']}")
    print(f"  样本数: {limit}")
    print(f"  并行执行: ShowUI-2B(GPU 6), OS-Atlas-7B(GPU 7)")
    print("="*60)
    
    print("\n🔍 检查现有结果文件...")
    existing_results = list_existing_results(dataset, 2)
    
    if existing_results:
        print(f"\n📊 发现已完成的模型结果:")
        for model_name, info in existing_results.items():
            print(f"  ✅ {model_name}: 完成时间 {info['modification_time']}")
            print(f"     定位成功率: {info['locate_success_rate']:.1%}, 交互成功率: {info['interaction_success_rate']:.1%}")
        
        print(f"\n⚠️  断点保护: 发现 {len(existing_results)} 个已完成的模型")
        rerun_choice = input("是否重新运行已完成的模型? (y/N): ").strip().lower()
        
        if rerun_choice not in ['y', 'yes']:
            print("✅ 将跳过已完成的模型，只运行未完成的模型")
        else:
            print("🔄 将重新运行所有模型")
            existing_results = {} 
    else:
        print("📭 未发现已完成的结果，将运行所有模型")

    print(f"\n{'='*60}")
    if existing_results:
        remaining_models = [m for m in OFFLINE_SCENARIO2_MODELS if m not in existing_results]
        if remaining_models:
            print(f"📋 待运行模型: {', '.join(remaining_models)}")
            confirm = input("确认开始测试未完成的模型? (y/N): ").strip().lower()
        else:
            print("🎉 所有模型都已完成！")
            return
    else:
        confirm = input("确认开始测试所有模型? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print("❌ 测试已取消")
        return

    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ 找不到模型配置文件: config/models_config.yaml")
        return

    if not os.path.exists(dataset):
        logger.error(f"❌ 找不到数据集目录: {dataset}")
        return
    
    remaining_models = [m for m in OFFLINE_SCENARIO2_MODELS if m not in existing_results]
    
    if not remaining_models:
        logger.info("🎉 所有模型都已完成，无需运行!")
        return

    required_gpus = []
    for model_name in remaining_models:
        if model_name == "ShowUI-2B":
            required_gpus.append(6)
        elif model_name == "OS-Atlas-Base-7B":
            required_gpus.append(7)
    
    logger.info(f"🎯 需要使用的GPU: {required_gpus}")
    wait_for_gpu_availability(required_gpus)
    
    total_tests = len(remaining_models)
    total_original = len(OFFLINE_SCENARIO2_MODELS)
    completed_already = len(existing_results)
    logger.info(f"🚀 开始并行测试，共 {total_tests} 个待运行模型 (已完成: {completed_already}/{total_original})")

    tasks = []
    for model_name in OFFLINE_SCENARIO2_MODELS:
        if model_name in remaining_models:
            gpu_id = 6 if model_name == "ShowUI-2B" else 7
            tasks.append((model_name, dataset, limit, gpu_id))
            logger.info(f"📋 准备任务: {model_name} → GPU {gpu_id}")
    
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
                logger.error(f"❌ {model_name} (GPU {gpu_id}) 执行异常: {e}")
                failed_tests += 1

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 并行测试完成统计 - {dataset}:")
    logger.info(f"  总模型数: {total_original}")
    logger.info(f"  已完成: {completed_tests}")
    logger.info(f"  本次运行: {len(tasks)}")
    logger.info(f"  失败测试: {failed_tests}")
    logger.info(f"  总体成功率: {completed_tests/total_original*100:.1f}%")

    logger.info(f"\n📁 结果保存路径:")
    result_base_dir = f"results/{dataset}_scenario2"
    for model_name in OFFLINE_SCENARIO2_MODELS:
        model_dir = f"{result_base_dir}/{model_name}"
        metrics_file = f"{model_dir}/{model_name}_metrics.json"
        if os.path.exists(metrics_file):
            logger.info(f"  ✅ {model_name}: {metrics_file}")
        else:
            logger.info(f"  ❌ {model_name}: {metrics_file} (未完成)")
    
    logger.info(f"{'='*80}")
    
    if completed_tests == total_original:
        logger.info(f"🎉 {dataset} 所有模型测试成功完成!")
    elif failed_tests > 0:
        logger.warning(f"⚠️ {dataset} 有 {failed_tests} 个测试失败")
    else:
        logger.info(f"✅ {dataset} 本次运行的 {len(tasks)} 个模型测试完成!")

def main():

    print("=" * 60)
    print("🎯 SpiderBench 离线场景2测试工具")
    print("📦 测试模型: ShowUI-2B, OS-Atlas-7B")
    print("🎯 测试场景: 2 (组件检测增强)")
    print("🔧 指定GPU: 6,7")
    print("=" * 60)
    
    if not TQDM_AVAILABLE:
        print("💡 提示: 安装 tqdm 库可以显示更好的进度条:")
        print("   pip install tqdm")
        print()
    
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ 找不到模型配置文件: config/models_config.yaml")
        return
    
    try:
        detector_key = select_detector_model()
        
        dataset, limit = select_dataset()
        
        run_offline_scenario2_on_dataset(dataset, limit, detector_key)
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断测试")
    except Exception as e:
        logger.error(f"❌ 程序异常: {str(e)}")

if __name__ == "__main__":
    main()