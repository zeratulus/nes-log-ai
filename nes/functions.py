import logging
import torch
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import nvidia_smi
import math

def torch_info():
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        for gpu_idx in range(torch.cuda.device_count()):
            logging.info(f"GPU:{gpu_idx} > name: {torch.cuda.get_device_name(gpu_idx)}")


def get_cuda_devices_for_llm_inference() -> None | int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()

    return None

def bytes_to_megabytes(bytes):
    return bytes / 1024 / 1024

def get_nvidia_free_gpu_memory_bytes(gpu_idx = 0):
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    free_mem = info.free
    nvidia_smi.nvmlShutdown()

    return free_mem

def get_logical_cpu_cores() -> int:
    cpu_logical_cores_count = os.cpu_count()
    logging.debug(f'CPU logical cores count: {cpu_logical_cores_count}')
    return cpu_logical_cores_count

def get_max_available_physical_cpu_cores() -> None | int:
    cpu_logical_cores_count = get_logical_cpu_cores()
    if cpu_logical_cores_count:
        cpu_cores_count = round(cpu_logical_cores_count / 2) - 1
        logging.debug(f'CPU cores count: {cpu_cores_count}')
        return cpu_cores_count

    return None

def get_approximated_ai_layers_count(default_ai_layer_size = 1200):
    free_mem = bytes_to_megabytes(get_nvidia_free_gpu_memory_bytes())
    apx_ai_layer_size = math.floor(free_mem / default_ai_layer_size)
    return apx_ai_layer_size

def init_llm(override_llm_model: str = '', override_max_tokens: int = 0, is_predict_ai_layers: bool = False) -> ChatOllama | ChatOpenAI:
    llm = None
    is_local_ollama_preferred = False
    if bool(os.environ.get("IS_LOCAL_OLLAMA_PREFERRED")):
        is_local_ollama_preferred = True

    if is_local_ollama_preferred and os.environ.get("LOCAL_OLLAMA_MODEL_CODER"):
        model = os.environ.get("LOCAL_OLLAMA_MODEL_CODER")
        num_ctx = (os.environ.get("LOCAL_OLLAMA_MODEL_CODER_NUM_CTX") or 16000)
        num_predict = (os.environ.get("LOCAL_OLLAMA_NUM_PREDICT") or -1)
        max_cpu_cores = get_logical_cpu_cores() - 4 if bool(os.environ.get("IS_LOCAL_OLLAMA_MAX_PERFORMANCE")) else get_logical_cpu_cores() / 2

        if len(override_llm_model) > 0:
            model = override_llm_model

        if override_max_tokens != 0:
            num_predict = override_max_tokens

        ollama_params = {
            "model": model,
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_thread": max_cpu_cores,
        }

        if is_predict_ai_layers:
            # num_gpu = get_cuda_devices_for_llm_inference()
            num_gpu = get_approximated_ai_layers_count()
            ollama_params["num_gpu"] = num_gpu

        logging.info(f"Current Ollama settings: {str(ollama_params)}", )

        llm = ChatOllama(**ollama_params)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not is_local_ollama_preferred and openai_api_key and os.environ.get("OPENAI_MODEL"):
        model = os.environ.get("OPENAI_MODEL")
        if len(override_llm_model) > 0:
            model = override_llm_model

        openai_params = {
            "model": model,
            "temperature": 0,
            "api_key": openai_api_key,
        }

        # if override_max_tokens != 0:
        #     openai_params['max_tokens'] = override_max_tokens

        logging.info(f"Current OpenAI settings: {str(openai_params)}", )
        llm = ChatOpenAI(**openai_params)

    if llm is None:
        raise Exception("LLM wasn't initialized, check your environment")

    return llm

def simple_error_detector(line: str) -> bool:
    error_markers = [
        'WARNING',
        'ERROR',
        'CRITICAL',
        'EMERGENCY',
    ]

    for marker in error_markers:
        if line.find(marker) >= 0:
            return True

        if line.find(marker.lower()) >= 0:
            return True

    return False