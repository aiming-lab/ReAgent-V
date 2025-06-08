# 🧠 ReAgent-V

This repository contains the official implementation of  
**[ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding](https://arxiv.org/abs/2506.01300)**.

**ReAgent-V** is a **modular, extensible, and reward-aware video reasoning framework**. It empowers video question answering and reasoning with:

- 🔧 **Flexible tool integration** (e.g., OCR, ASR, object detection)
- 🧠 **Reward-guided inference** for real-time self-correction
- 🎯 **Adaptive model alignment** driven by structured feedback
- 🗂️ **High-quality data selection** for sample-efficient learning
- 📊 **Entropy-calibrated frame selection** to focus on key moments
- 🔁 **Multi-perspective reflection** (conservative, neutral, aggressive) for answer refinement

---

## 📌 Overview

ReAgent-V comprises two major components:

### 🎥 Video Understanding Framework

- **Entropy-Calibrated Frame Selection**: Identifies the most informative video segments for reasoning.
- **Tool-Augmented Inference**: Dynamically invokes OCR, ASR, object detection, scene graph generation, captioning, and more.
- **Multi-Agent Reflection**: Refines answers through iterative debate from multiple viewpoints.

📄 For technical details, see:  
- ![Framework Overview](./assets/framework.png)
- [⚙️ Prompt Templates and Tool Setup, etc.](https://github.com/aiming-lab/ReAgent-V/blob/main/ReAgent-V/readme.md)

---

### 🚀 Applications

ReAgent-V powers a variety of real-world tasks through specialized application modules:

#### 🧭 VLA Alignment

Aligns **Vision-Language-Action (VLA)** models using **Trajectory-wise Preference Optimization (TPO)** driven by ReAgent-V’s reward signals.

- 📂 Module: `Application/VLA-Alignment`  
- 📘 Instructions: [VLA Alignment README](https://github.com/aiming-lab/ReAgent-V/blob/main/Application/VLA-Alignment/README.md)

#### 🔍 Video LLM Reasoning

Leverages reflection-guided evaluation to curate training samples for **GRPO-based long-form video LLM fine-tuning**.

- 📂 Module: `Application/LLM-Reasoning` *(Coming Soon)*

---

## 🧑‍💻 Getting Started

To get started, please refer to the `README.md` in each subfolder for installation, setup, and training instructions.

If you have questions or encounter any issues, feel free to open an [issue](https://github.com/aiming-lab/ReAgent-V/issues) or contact the maintainers.


Citation
If you find our code or models useful in your work, please cite our paper: [ReAgent-V](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C24&q=ReAgent-V%3A+A+Reward-Driven+Multi-Agent+Framework+for+Video+Understanding&btnG=)

@article{zhou2025reagent,
  title={ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding},
  author={Zhou, Yiyang and He, Yangfan and Su, Yaofeng and Han, Siwei and Jang, Joel and Bertasius, Gedas and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2506.01300},
  year={2025}
}
