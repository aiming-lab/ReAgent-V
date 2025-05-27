import os
import sys
from typing import Optional, Sequence, Dict, Any, Union, List, Tuple
from collections import deque

import numpy as np
import torch
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from transforms3d.euler import euler2axangle

# Add openvla-oft directory to path to import required modules
sys.path.append('/home/yaofeng/GRAPE/openvla-oft')

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_noisy_action_projector,
    get_proprio_projector,
    prepare_images_for_vla,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class OpenVLAOFTInference:
    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: str = "bridge_orig",
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: List[int] = [224, 224],
        action_scale: float = 1.0,
        use_diffusion: bool = True,
        num_diffusion_steps: int = 50,
        use_l1_regression: bool = False,
        use_proprio: bool = False,
        center_crop: bool = False,  # Changed default to False to avoid cropping
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        num_images_in_input: int = 1,
        use_film: bool = False,
    ) -> None:
        """
        Initialize OpenVLA-OFT inference model with diffusion head.
        
        Args:
            saved_model_path: Path to the saved model
            unnorm_key: Key for action un-normalization
            policy_setup: Robot policy setup type
            horizon: Future time horizon
            pred_action_horizon: Prediction action horizon
            exec_horizon: Execution horizon
            image_size: Input image size
            action_scale: Scale factor for actions
            use_diffusion: Whether to use diffusion head
            num_diffusion_steps: Number of diffusion steps for inference
            use_l1_regression: Whether to use L1 regression head
            use_proprio: Whether to use proprioception
            center_crop: Whether to center crop images (now defaults to False)
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            num_images_in_input: Number of images in input
            use_film: Whether to use FiLM
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Configure policy setup
        if policy_setup == "widowx_bridge":
            self.sticky_gripper_num_repeat = 1
            print(f"Using unnorm_key: {unnorm_key}")
        elif policy_setup == "google_robot":
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for OpenVLA-OFT models."
            )
        
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.use_diffusion = use_diffusion
        self.num_diffusion_steps = num_diffusion_steps
        self.use_l1_regression = use_l1_regression
        self.use_proprio = use_proprio
        self.center_crop = center_crop
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.num_images_in_input = num_images_in_input
        self.use_film = use_film

        # Create a config object for the VLA model
        class Config:
            def __init__(self):
                self.pretrained_checkpoint = saved_model_path
                self.unnorm_key = unnorm_key
                self.use_diffusion = use_diffusion
                self.num_diffusion_steps = num_diffusion_steps
                self.use_l1_regression = use_l1_regression
                self.use_proprio = use_proprio
                self.center_crop = center_crop
                self.load_in_8bit = load_in_8bit
                self.load_in_4bit = load_in_4bit
                self.num_images_in_input = num_images_in_input
                self.use_film = use_film
                self.model_family = "openvla"
                self.num_open_loop_steps = NUM_ACTIONS_CHUNK  # Important: Use same value as NUM_ACTIONS_CHUNK

        self.cfg = Config()
        
        # Load model and components
        print("Loading OpenVLA-OFT model and components...")
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        
        # Load action head
        if self.use_diffusion or self.use_l1_regression:
            self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        else:
            self.action_head = None
            
        # Load proprio projector
        if self.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, 
                self.vla.llm_dim, 
                proprio_dim=8  # 8D for robot state (position, rotation, gripper)
            )
        else:
            self.proprio_projector = None
            
        # Load noisy action projector for diffusion
        if self.use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(self.cfg, self.vla.llm_dim)
        else:
            self.noisy_action_projector = None

        # Initialize action queue and currently executing action list
        self.action_queue = []
        
        # Initialize policy state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0
        
        print("OpenVLA-OFT model initialized successfully.")
        print(f"Model will predict {NUM_ACTIONS_CHUNK} actions at once")

    def reset(self, task_description: str) -> None:
        """Reset policy state with a new task description."""
        self.task_description = task_description
        self.num_image_history = 0

        # Clear action queue
        self.action_queue = []

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _query_model_for_actions(self, image: np.ndarray, task_description: str) -> List[Dict[str, np.ndarray]]:
        """
        Query the model to get a chunk of actions.
        
        Args:
            image: Input image
            task_description: Task description
            
        Returns:
            List of action dictionaries
        """
        # Prepare observation
        obs = {
            "full_image": image,
            # Dummy state (position, rotation, gripper) - will be updated in real env
            "state": np.zeros(8, dtype=np.float32),
        }
        
        # For multi-image input (future extension)
        if self.num_images_in_input > 1:
            obs["wrist_image"] = image  # Use the same image as a placeholder

        # Following the same approach as in run_libero_eval.py
        # Use the robot_utils.get_action function via openvla_utils.get_vla_action
        from experiments.robot.robot_utils import get_action
        
        # Get a chunk of actions (this already returns a list of actions)
        actions = get_action(
            self.cfg,
            self.vla,
            obs,
            task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.use_film,
        )
        
        # Create a list of raw action dictionaries from the action chunk
        raw_action_list = []
        for action in actions:
            raw_action = {
                "world_vector": action[:3],
                "rotation_delta": action[3:6],
                "open_gripper": action[6:7],
            }
            raw_action_list.append(raw_action)
            
        return raw_action_list

    def _process_raw_action(self, raw_action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process a raw action into an action that can be executed by the environment.
        
        Args:
            raw_action: Raw action from the model
            
        Returns:
            Processed action
        """
        # Process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        
        # Convert rotation delta to axis-angle
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        # Handle gripper action based on policy setup
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        
        return action

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        
        # If action queue is empty, query model for more actions
        if len(self.action_queue) == 0:
            raw_action_list = self._query_model_for_actions(image, task_description)
            # Add all actions to the queue
            self.action_queue = raw_action_list
        
        # Get next action from queue
        raw_action = self.action_queue.pop(0)
        
        # Process raw action into executable action
        action = self._process_raw_action(raw_action)
        
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to the expected input size without cropping."""
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        """Visualize actions and images from an epoch."""
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
