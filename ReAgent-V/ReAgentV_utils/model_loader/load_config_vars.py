import yaml
import os

def load_config_vars(yaml_path=None):
  
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    rag_threshold  = cfg_dict.get("rag_threshold", 0.3)
    clip_threshold = cfg_dict.get("clip_threshold", 0.3)
    beta           = cfg_dict.get("beta", 3.0)

    USE_OCR        = cfg_dict.get("USE_OCR", False)
    USE_ASR        = cfg_dict.get("USE_ASR", False)
    USE_DET        = cfg_dict.get("USE_DET", False)

    max_frames_num = cfg_dict.get("max_frames_num", 64)
    device         = cfg_dict.get("device", "cuda")

    return (
        rag_threshold,
        clip_threshold,
        beta,
        USE_OCR,
        USE_ASR,
        USE_DET,
        max_frames_num,
        device
    )
