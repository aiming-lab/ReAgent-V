# ReAgent-V VLA Alignment 

This application investigates whether ReAgent-V can enhance VLA alignment, moving beyond general video understanding improvements.
The following code outlines the pipeline presented in the paper, demonstrating how ReAgent-V's evaluation is utilized for Vision-Language Agent (VLA) alignment. You are welcome to use this pipeline in its entirety or adapt specific components for your research or further development. This part of code is modified from [GRAPE](https://github.com/aiming-lab/grape).


## Environment Setup

Use the setup commands below to get started.

To setup OpenVLA and training environment.
```bash
# Create and activate conda environment
conda create -n your_env python=3.10 -y
conda activate your_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  

# Install the modified openvla repo
cd Train
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

To setup Simpler-env and Maniskill environment.
Note: You might encounter issues with missing or conflicting dependencies. Manually resolving these usually proves effective.
```bash
pip install orbax-checkpoint==0.4.4
pip install scipy==1.12.0
pip install keras==2.15.0
pip install tensorflow==2.15.1
pip install --quiet tf_agents
pip install --quiet mediapy
pip install peft


# Install Simpler-Env
cd Env\Simpler-Env\ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
```

To setup RLDS builder environment.
Note: please create a new conda env.
Please refer to [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder).
## Data Preparation

To collect VLA rollouts for ReAgent-V to evaluate. You need to do the following things.

1.Navigate into the following directory: Env/Simpler-env/simpler_env/evaluation.
Once there, you're going to replace the file maniskill2_evaluator.py with maniskill2_evaluator_collect.py.
Please make sure to backup the original maniskill2_evaluator.py file before you replace it, as you will still need this original version for the evaluation phase.

2.Navigate into Env/Simpler-env.
Run the following 4 commands in sequence.
We will later reduce the number of these trajectories to a total of 80, to align with the description in the paper.
```bash
python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/sft_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/sft_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/sft_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/sft_model" \
  --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutEggplantInBasketScene-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
  --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

```
3.Please refer to [Main README](https://github.com/aiming-lab/ReAgent-V/blob/main/README.md) to use ReAgent-V for evaluating these rollouts.
After that, you will get an evaluation file like [example.jsonl](https://github.com/aiming-lab/ReAgent-V/blob/main/Application/VLA-Alignment/Dataset/example.jsonl).

4.Run our extraction script to extract positive-negative pairs for TPO dataset construction.
You will need to modify some directory paths in the script [extract.py](https://github.com/aiming-lab/ReAgent-V/blob/main/Application/VLA-Alignment/Dataset/extract_scores_with_prefix.py).
```bash
cd Dataset
python extract_scores_with_prefix.py
```

5. Now, you have the data for training. You still need to convert it to RLDS format.
Please use the RLDS environment you have already set up. Ensure you use our modified example_dataset files.
You will also need to modify some directory paths in the code.
You need to create datasets for positive rollouts and negative rollouts separately.
This means you may need to adjust your code and run the following command twice.
After creating each dataset, please move it to a suitable location and give it an appropriate name.
```bash
cd Dataset/rlds_dataset_builder/example_dataset/
tfds build
```

Data preparation is now complete.

## TPO Training

Below we show how you can use our data to train the main OpenVLA-SFT checkpoint via LoRA-TPO. Here we use a single A100 GPU with 80 GB VRAM. And our training stops near 6,800 steps. (We only support batchsize=1 and single-GPU training now, which means each batch has a pair of trajectories.)

You can get base SFT model from ([OpenVLA-7B-SFT-Simpler](https://huggingface.co/ZijianZhang/OpenVLA-7B-SFT-Simpler)).

Now, launch the TPO-LoRA script, as shown below. 

```bash
  torchrun --standalone --nnodes=1 --nproc-per-node 1 finetune.py \
  --vla_path <PATH TO REFERENCE MODEL> \
  --dataset_name "rlds_np_rollout" \
  --chosen_traj_dir <PATH TO CHOSEN TRAJECTORY DATASET> \
  --rejected_traj_dir <PATH TO REJECTED TRAJECTORY DATASET> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project <YOUR PROJECT NAME> \
  --wandb_entity <YOUR ENTITY> \
```

The chosen_traj_dir and rejected_traj_dir are the directories of your RLDS datasets.
You should rename the folders for both datasets to "rlds_np_rollout" and place these two datasets in different directories (i.e., two separate rlds_np_rollout directories).

After training, please merge the LoRA weights into the main model.
We have provided the merge script in Train/merge.py. You will also need to modify some directory paths in this script.

## Evaluation
The evaluation steps vary depending on the four generalization tasks.
First, make sure to use the original maniskill2_evaluator.py file.

For In-domain Tasks, 
simply run the following code:
```bash
python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutEggplantInBasketScene-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
  --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

```
For ​​Physical Generalization Tasks​​, 
the setup requires modifying Application/VLA-Alignment/Env/Simpler-env/ManiSkill2_real2sim/data/custom/info_bridge_custom_v0.json and re-running the ​​1st and 3rd commands​​ from the ​​In-domain tasks​​ procedure. The evaluation involves ​​8 test cases​​, where both "bridge_carrot_generated_modified" and "bridge_spoon_generated_modified" are tested at ​​1.1x and 0.5x scales​​, as well as at ​​1.0x scale with wider and longer bounding boxes​​.


For Subject Generalization Tasks, 
you need to replace Env/Simpler-env/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene.py with Env/Simpler-env/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene_generalization.py. (Please make backup.)
And simply run the following code:
```bash
python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name Coke --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name Sprite --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/merged_model" \
  --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name Pepsi --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
  --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

```

For Semantic Generalization Tasks, 
you also need to use Env/Simpler-env/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene_generalization.py.
And simply repeat the commands in In-domain tasks.
