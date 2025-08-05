#!/bin/bash

# Desktop 消融实验脚本 - 只运行 Description-only 和 Coordinate-only 实验

# 代理配置函数
setup_proxy() {
    echo "配置代理..." | tee -a $LOG_FILE
    
    # 启动代理
    clashon
    sleep 2
    
    # 设置全局模式
    curl -X PATCH http://127.0.0.1:9090/configs -d '{"mode": "Global"}' 2>/dev/null
    sleep 1
    
    # 获取可用节点
    echo "获取可用代理节点..." | tee -a $LOG_FILE
    available_proxies=$(curl -s http://127.0.0.1:9090/proxies/GLOBAL | python3 -c "
import json, sys
data = json.load(sys.stdin)
proxies = data.get('all', [])
for proxy in proxies[:5]:  # 显示前5个节点
    print(f'可用节点: {proxy}')
# 使用第一个可用节点
if proxies:
    print(f'使用节点: {proxies[0]}')
    sys.exit(0)
else:
    sys.exit(1)
")
    
    # 设置代理节点（使用第一个可用节点）
    first_proxy=$(curl -s http://127.0.0.1:9090/proxies/GLOBAL | python3 -c "
import json, sys
data = json.load(sys.stdin)
proxies = data.get('all', [])
if proxies:
    print(proxies[0])
")
    
    if [ ! -z "$first_proxy" ]; then
        curl -X PUT http://127.0.0.1:9090/proxies/GLOBAL -d "{\"name\": \"$first_proxy\"}" 2>/dev/null
        echo "代理设置完成: $first_proxy" | tee -a $LOG_FILE
        
        # 设置环境变量
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export HTTPS_PROXY=http://127.0.0.1:7890
    else
        echo "警告: 无法获取代理节点，继续使用直连" | tee -a $LOG_FILE
    fi
}

# 检查网络连接
check_network() {
    echo "检查网络连接..." | tee -a $LOG_FILE
    if ! curl -s --connect-timeout 10 https://www.google.com > /dev/null; then
        echo "网络连接失败，尝试配置代理..." | tee -a $LOG_FILE
        setup_proxy
        
        # 再次检查
        if ! curl -s --connect-timeout 10 https://www.google.com > /dev/null; then
            echo "代理配置后仍无法连接，请检查网络设置" | tee -a $LOG_FILE
        else
            echo "代理配置成功" | tee -a $LOG_FILE
        fi
    else
        echo "网络连接正常" | tee -a $LOG_FILE
    fi
}

# 设置日志文件
LOG_FILE="benchmark_desktop_ablation.log"
echo "开始 Desktop 消融实验 $(date)" > $LOG_FILE

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查并配置网络
check_network

# 固定使用 gemini-2.5-flash 模型
SELECTED_MODEL="google/gemini-2.5-flash"
echo "使用模型: $SELECTED_MODEL" | tee -a $LOG_FILE

# 根据选择的模型设置进度文件
MODEL_SAFE_NAME=$(echo "$SELECTED_MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
PROGRESS_FILE="evaluation_progress_desktop_ablation_${MODEL_SAFE_NAME}.json"

echo "进度文件: $PROGRESS_FILE" | tee -a $LOG_FILE

if [ -f "$PROGRESS_FILE" ]; then
    echo "发现进度文件，将从断点继续运行..." | tee -a $LOG_FILE
else
    echo "未找到进度文件，将开始全新评估..." | tee -a $LOG_FILE
fi

# 运行Python脚本进行评估
echo "开始运行消融实验..." | tee -a $LOG_FILE

# 设置环境变量
export PYTHONPATH="$SCRIPT_DIR"
export TEST_NON_INTERACTIVE=1
export SELECTED_MODEL="$SELECTED_MODEL"
export PROGRESS_FILE="$PROGRESS_FILE"

# 创建Python评估脚本
cat > run_desktop_ablation.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import logging
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
sys.path.append(os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 进度文件路径 - 将由shell脚本传入
PROGRESS_FILE = os.environ.get('PROGRESS_FILE', 'evaluation_progress_desktop_ablation.json')

def load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载进度文件失败: {e}")
    return {}

def save_progress(progress_data):
    """保存进度到文件"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        logger.error(f"保存进度文件失败: {e}")

def is_task_completed(model_name, ablation_type, progress_data):
    """检查任务是否已完成"""
    task_key = f"{model_name}_ablation_{ablation_type}"
    return progress_data.get(task_key, {}).get('completed', False)

def mark_task_completed(model_name, ablation_type, progress_data):
    """标记任务为已完成"""
    task_key = f"{model_name}_ablation_{ablation_type}"
    progress_data[task_key] = {
        'completed': True,
        'completion_time': datetime.now().isoformat(),
        'model': model_name,
        'ablation_type': ablation_type
    }
    save_progress(progress_data)

def clear_gpu_memory():
    """清理GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU内存已清理")
    except Exception as e:
        logger.warning(f"清理GPU内存失败: {e}")

def run_ablation_test(model_name: str, ablation_type: str, limit: int = 821) -> bool:
    """运行指定类型的消融测试"""
    ablation_configs = {
        "description_only": {
            "name": "仅描述消融实验",
            "description": "只使用组件描述，不包含坐标信息",
            "use_cache": False,
            "cache_source_dir": None
        },
        "coordinate_only": {
            "name": "仅坐标消融实验", 
            "description": "只使用坐标预测，不包含描述信息",
            "use_cache": False,
            "cache_source_dir": None
        }
    }
    
    ablation_info = ablation_configs[ablation_type]
    
    logger.info("="*60)
    logger.info(f"🚀 开始消融实验")
    logger.info(f"模型: {model_name}")
    logger.info(f"类型: {ablation_type} ({ablation_info['name']})")
    logger.info(f"描述: {ablation_info['description']}")
    logger.info(f"测试用例数: {limit}")
    logger.info("="*60)
    
    # 简单进度回调
    def simple_progress(current: int, total: int, message: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"进度: {current}/{total} ({percentage:.1f}%) - {message}")
    
    try:
        # 导入评估器
        from evaluation.benchmark import BenchmarkEvaluator
        
        # 创建评估器，使用desktop_en数据
        evaluator = BenchmarkEvaluator(
            data_root="desktop_en",
            use_cache=ablation_info['use_cache'], 
            cache_source_dir=ablation_info['cache_source_dir']
        )
        
        # 设置进度回调
        evaluator.set_progress_callback(simple_progress)
        
        start_time = time.time()
        
        # 运行评估 - 使用场景2（组件检测增强）的设置
        evaluator.run_evaluation(
            model_name=model_name,
            limit=limit,
            scenario=2,  # 使用场景2的框架
            detector_model="google/gemini-2.5-flash",
            use_ground_truth=False,
            ablation_type=ablation_type  # 传递消融类型
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ {model_name} {ablation_type} 测试完成 - 耗时: {duration:.1f}秒")
        
        # 标记任务完成
        progress_data = load_progress()
        mark_task_completed(model_name, ablation_type, progress_data)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name} {ablation_type} 测试失败: {str(e)}")
        return False
    finally:
        clear_gpu_memory()

def main():
    """主函数（消融实验版）"""
    # 从环境变量获取模型名称
    selected_model = os.environ.get('SELECTED_MODEL', 'google/gemini-2.5-flash')
    models = [selected_model]
    
    # 消融实验类型
    ablation_types = ["description_only", "coordinate_only"]
    
    limit = 821  # desktop_en数据集总数

    # 检查环境
    if not os.path.exists("config/models_config.yaml"):
        logger.error("❌ 找不到模型配置文件: config/models_config.yaml")
        return False
    
    # 检查desktop_en数据是否存在
    if not os.path.exists("desktop_en"):
        logger.error("❌ 找不到测试数据目录: desktop_en")
        return False

    # 加载进度数据
    progress_data = load_progress()
    
    # 过滤掉已完成的任务
    all_tasks = [(m, a) for m in models for a in ablation_types]
    pending_tasks = [(m, a) for m, a in all_tasks if not is_task_completed(m, a, progress_data)]
    
    total_tests = len(all_tasks)
    completed_tests = total_tests - len(pending_tasks)
    success_count = completed_tests

    if completed_tests > 0:
        logger.info(f"⏭️  发现已完成的任务: {completed_tests}/{total_tests}")
        logger.info(f"📝 跳过已完成任务，继续执行剩余 {len(pending_tasks)} 个任务")
    else:
        logger.info(f"🚀 开始全新消融实验，共 {total_tests} 个任务")

    if not pending_tasks:
        logger.info("🎉 所有消融实验已完成！")
        return True

    logger.info(f"将执行 {len(pending_tasks)} 个剩余任务")

    # 断点保护：添加确认机制
    if len(pending_tasks) > 0:
        logger.info("⚠️  断点保护：即将开始消融实验")
        logger.info(f"模型: {pending_tasks[0][0]}")
        logger.info(f"消融类型: {pending_tasks[0][1]}")
        logger.info("如需停止，请在5秒内按 Ctrl+C")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("用户中断，安全退出")
            return False

    # 串行运行消融实验（避免缓存冲突）
    for m, a in pending_tasks:
        try:
            ok = run_ablation_test(m, a, limit)
            if ok:
                success_count += 1
                logger.info(f"✅ {m} {a} 完成")
            else:
                logger.error(f"❌ {m} {a} 失败")
        except Exception as e:
            logger.error(f"❌ {m} {a} 异常: {e}")

    final_success_count = success_count
    logger.info(f"📊 消融实验完成统计:")
    logger.info(f"   总任务数: {total_tests}")
    logger.info(f"   已完成: {final_success_count}")
    logger.info(f"   成功率: {final_success_count/total_tests*100:.1f}%")
    
    return final_success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# 运行Python评估脚本
python run_desktop_ablation.py

# 检查评估结果
if [ $? -eq 0 ]; then
    echo "✅ Desktop 消融实验成功完成 $(date)" | tee -a $LOG_FILE
    echo "进度文件保存在: $PROGRESS_FILE" | tee -a $LOG_FILE
else
    echo "❌ Desktop 消融实验过程中出现错误 $(date)" | tee -a $LOG_FILE
    echo "进度已保存，可重新运行脚本继续执行" | tee -a $LOG_FILE
fi

# 记录完成时间
echo "Desktop 消融实验完成 $(date)" | tee -a $LOG_FILE

# 清理临时文件
rm -f run_desktop_ablation.py