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
- **Inference-Time Reward Generation**: Produces structured feedback and scalar scores to guide real-time answer improvement.
- **Data Curation & Optimization**: Selects valuable reasoning samples for fine-tuning via SFT, DPO, and GRPO.

📄 For technical details, see:  
- [🧩 Framework Overview (PDF)](./assets/framework.pdf)  
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
