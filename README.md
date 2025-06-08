# 🧠 ReAgent-V
This repository provides the official implementation of  
**[ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding](https://arxiv.org/abs/2506.01300)**.

**ReAgent-V** is a **modular**, **extensible**, and **reward-aware** video reasoning framework designed to elevate video question answering and reasoning through:

- 🔧 **Flexible Tool Integration** — Plug-and-play support for OCR, ASR, object detection, scene graph generation, captioning, and more  
- 🧠 **Reward-Guided Inference** — Enables real-time self-correction via structured reward signals  
- 🎯 **Adaptive Model Alignment** — Aligns models dynamically based on inference-time feedback  
- 🗂️ **High-Quality Data Selection** — Facilitates sample-efficient learning using reflective evaluation  
- 📊 **Entropy-Calibrated Frame Selection** — Prioritizes key frames for focused reasoning  
- 🔁 **Multi-Perspective Reflection** — Refines answers through debate among conservative, neutral, and aggressive viewpoints  

---

## 📌 Overview
![Framework Overview](assets/framework.png)  
ReAgent-V consists of two major components:

### 🎥 Video Understanding Framework

- **Entropy-Calibrated Frame Selection**  
  Efficiently selects the most informative frames for video reasoning.

- **Tool-Augmented Inference**  
  Dynamically integrates multimodal tools including OCR, ASR, object detection, scene graph generation, and captioning.

- **Multi-Agent Reflection**  
  Iteratively refines outputs by encouraging disagreement and consensus among diverse agent personas (conservative / neutral / aggressive).

📄 **Technical Resources**  
- [⚙️ Prompt Templates and Tool Setup Guide](https://github.com/aiming-lab/ReAgent-V/blob/main/ReAgent-V/readme.md)

---

## 🚀 Applications

ReAgent-V supports a range of real-world tasks via dedicated application modules:

### 🧭 VLA Alignment  
Aligns **Vision-Language-Action (VLA)** models using **Trajectory-wise Preference Optimization (TPO)** guided by ReAgent-V’s reward feedback.

- 📁 Module: `Application/VLA-Alignment`  
- 📘 Instructions: [VLA Alignment README](https://github.com/aiming-lab/ReAgent-V/blob/main/Application/VLA-Alignment/README.md)

### 🔍 Video LLM Reasoning *(Coming Soon)*  
Supports reflection-based evaluation and reward-guided sample curation for long-form video LLM fine-tuning using **GRPO** strategies.

- 📁 Module: `Application/LLM-Reasoning`  

---

## 🧑‍💻 Getting Started

Each subfolder contains its own `README.md` with detailed installation, setup, and training instructions. To get started:

1. Clone the repository  
2. Follow the environment setup and requirements in each module  
3. Explore the demo scripts and customize as needed

💬 If you have questions or encounter any issues, feel free to open an [issue](https://github.com/aiming-lab/ReAgent-V/issues) or contact the maintainers.

---

## 📚 Citation

If you find ReAgent-V helpful in your research or projects, please consider citing:

```bibtex
@article{zhou2025reagent,
  title={ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding},
  author={Zhou, Yiyang and He, Yangfan and Su, Yaofeng and Han, Siwei and Jang, Joel and Bertasius, Gedas and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2506.01300},
  year={2025}
}
