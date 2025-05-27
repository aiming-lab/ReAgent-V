import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from pathlib import Path

# 直接设置路径而不是通过命令行参数
BASE_MODEL_PATH = "/home/yaofeng/GRAPE/OpenVLA-7B-SFT-Simpler"
LORA_ADAPTER_PATH = "/home/yaofeng/GRAPE/123/trial2-ada/OpenVLA-7B-SFT-Simpler+rlds_np_rollout+b4+lr-2e-05+lora-r32+dropout-0.0/d1121_check_epoch_7"
OUTPUT_DIR = "/home/yaofeng/GRAPE/merged3/"

def merge_lora_weights(base_model_path, lora_adapter_path, output_dir):
    """
    Merges LoRA weights with the base model and saves the resulting full model.
    
    Args:
        base_model_path: Path to the base model
        lora_adapter_path: Path to the LoRA adapter weights
        output_dir: Directory to save the merged model
    """
    print(f"Loading base model from {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA weights from {lora_adapter_path}")
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging weights...")
    # Merge weights
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving merged model to {output_dir}")
    # Save merged model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("Merge complete!")

if __name__ == "__main__":
    # 使用直接定义的路径而不是命令行参数
    merge_lora_weights(BASE_MODEL_PATH, LORA_ADAPTER_PATH, OUTPUT_DIR)