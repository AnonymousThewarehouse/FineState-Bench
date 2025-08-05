import os
import time
import concurrent.futures
from huggingface_hub import hf_hub_download, list_repo_files, HfFolder
from tqdm import tqdm
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7890')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7890')

MAX_CONCURRENT_DOWNLOADS = 5

models_to_download = {
    "OS-Copilot/OS-Atlas-Base-7B": "OS-Atlas-Base-7B",
    "THUDM/cogagent-9b-20241220": "cogagent-9b-20241220",
    "osunlp/UGround-V1-7B": "UGround-V1-7B",
    "ByteDance-Seed/UI-TARS-1.5-7B": "UI-TARS-1.5-7B",
    "xlangai/Jedi-7B-1080p": "Jedi-7B-1080p",
    "SenseLLM/SpiritSight-Agent-8B": "SpiritSight-Agent-8B",
    "showlab/ShowUI-2B": "ShowUI-2B",
    "mtgv/MobileVLM_V2-3B": "MobileVLM_V2-3B",
    "mtgv/MobileVLM_V2-7B": "MobileVLM_V2-7B",
    "openbmb/AgentCPM-GUI": "AgentCPM-GUI",
    "Hcompany/Holo1-7B": "Holo1-7B",
    "xhan77/web-llama2-13b-adapter": "web-llama2-13b-adapter",
    "caca9527/GUIExplorer": "GUIExplorer",
    "meta-llama/Llama-2-13b-hf": "Llama-2-13b-hf",
}

base_model_path = "/home/yjp/spiderbench/models"

def download_file_with_retry(repo_id, filename, local_dir, pbar):

    max_retries = 5
    base_wait_time = 10

    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            tqdm.write(f"  ✅ File processing completed: {filename}")
            pbar.update(1) 
            return True
        except Exception as e:
            tqdm.write(f"  ⚠️ (Attempt {attempt + 1}/{max_retries}) Failed to download file {filename}: {e}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (attempt + 2)
                tqdm.write(f"  ...Network error, retrying in {wait_time} seconds.")
                time.sleep(wait_time)
            else:
                tqdm.write(f"  ❌ File {filename} still failed after {max_retries} retries.")
                pbar.update(1) 
                return False

def download_repo_file_by_file(repo_id, local_folder_name):
    """
    Get the repository file list and download all files concurrently using a thread pool.
    """
    tqdm.write(f"\n=================================================")
    tqdm.write(f"📁 Starting to process repository: {repo_id}")
    tqdm.write(f"=================================================")
    
    local_path = os.path.join(base_model_path, local_folder_name)
    os.makedirs(local_path, exist_ok=True)

    try:
        filenames = list_repo_files(repo_id=repo_id)
        tqdm.write(f"  - Found {len(filenames)} files in repository. Will use up to {MAX_CONCURRENT_DOWNLOADS} threads for concurrent download.")
    except Exception as e:
        tqdm.write(f"❌ Critical error: Unable to get file list for repository {repo_id}. Error: {e}")
        tqdm.write(f"  Please confirm your server can access the mirror site: {os.environ.get('HF_ENDPOINT')}")
        return

    with tqdm(total=len(filenames), desc=f"Repository {repo_id[:20]}...", unit="file") as pbar:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        
            futures = {
                executor.submit(download_file_with_retry, repo_id, filename, local_path, pbar)
                for filename in filenames
            }

if __name__ == "__main__":
    print(f"Preparing to start download tasks, processing models one by one, files within models will be downloaded concurrently.")
    print(f"All models will be downloaded to root directory: {base_model_path}")
    print(f"Using mirror site: {os.environ.get('HF_ENDPOINT')}")
    print(f"Force offline mode: {os.environ.get('HF_HUB_OFFLINE')}")
    print(f"Maximum concurrent file downloads: {MAX_CONCURRENT_DOWNLOADS}")

    for repo_id, local_folder in models_to_download.items():
        download_repo_file_by_file(repo_id, local_folder)

    print("\n🎉 All specified model download tasks have been completed. Please check the logs above to confirm the download status of each file.")