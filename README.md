# 🧠 ReAgent-V

This repository contains the official implementation of **[ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding](https://arxiv.org/abs/2506.01300)**.

ReAgent-V is a **modular, tool-augmented, and reward-aware video reasoning framework** designed for dynamic video question answering, adaptive model alignment, and high-quality data selection. It supports **flexible tool integration**, **entropy-calibrated frame selection**, and **multi-perspective reflection** during inference, enabling real-time self-correction and sample-efficient training.

---

## 📌 Overview

ReAgent-V consists of two main components:

### 🎥 Video Reasoning Framework

- **Entropy-Calibrated Frame Selection**: Efficiently identifies relevant video segments for reasoning.
- **Tool-Augmented Inference**: Dynamically invokes OCR, ASR, DET, Scene Graph, Captioning, and more.
- **Multi-Agent Reflection**: Iteratively refines answers from conservative, neutral, and aggressive viewpoints.
- **Inference-Time Reward Generation**: Produces structured feedback and scalar scores to guide answer updates.
- **Data Curation and Optimization**: Selects high-quality samples for SFT, DPO, and GRPO training.

For more details, refer to:
- [Framework Overview](./assets/framework.pdf)
- [Prompt Templates and Tool Setup](https://github.com/aiming-lab/ReAgent-V/blob/main/ReAgent-V/readme.md)

---

### 🚀 Applications

ReAgent-V supports diverse real-world tasks through specialized application modules.

#### 🧭 VLA Alignment

Leverages ReAgent-V's reward system to align **Vision-Language-Action (VLA)** models via **Trajectory-wise Preference Optimization (TPO)**.

📂 Directory: `Application/VLA-Alignment`  
📄 Detailed Instructions: [VLA Alignment README](https://github.com/aiming-lab/ReAgent-V/blob/main/Application/VLA-Alignment/README.md)

#### 🔍 Video LLM Reasoning

Uses reflection-guided evaluation to **select informative samples** for **GRPO-based reasoning fine-tuning** in long-form video LLMs.

📂 Directory: `Application/LLM-Reasoning` *(Coming Soon)*


## 🧑‍💻 Getting Started

Please refer to the `README.md` in each subfolder for installation, setup, and training instructions.

If you encounter issues or have questions, feel free to open an issue or contact us!

