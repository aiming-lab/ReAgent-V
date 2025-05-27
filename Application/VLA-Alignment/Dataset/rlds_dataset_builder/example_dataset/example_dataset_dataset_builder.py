from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os, re
from PIL import Image
import json

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=None),  # path parameter is not used anymore
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        # Load the JSON file
        with open('/home/yaofeng/GRAPE/bridge_rollout.json', 'r') as f:
            data = json.load(f)
        
        # Group entries by task and episode
        episodes = {}
        for entry in data:
            if not isinstance(entry, dict) or 'image' not in entry:
                continue
                
            image_path = entry.get('image', '')
            if not image_path:
                continue
                
            # Extract task and episode from image path
            match = re.search(r'bridge_rollout_([^/]+)/episode_(\d+)', image_path)
            if not match:
                continue
                
            task_name = match.group(1)
            episode_num = match.group(2)
            key = f"{task_name}_episode_{episode_num}"
            
            if key not in episodes:
                episodes[key] = []
            
            episodes[key].append(entry)
        
        # Process each episode
        episode_id = 0
        for key, entries in episodes.items():
            # Sort entries by step number
            entries.sort(key=lambda x: int(re.search(r'step_(\d+)', x['image']).group(1)))
            
            episode = []
            for i, entry in enumerate(entries):
                # Load image
                try:
                    image_path = entry['image']
                    image = Image.open(image_path)
                    
                    # Resize image to 224x224 if it's not already that size
                    if image.size != (224, 224):
                        image = image.resize((224, 224), Image.LANCZOS)
                    
                    # Convert to numpy array and ensure proper format
                    image_array = np.array(image)
                except Exception as e:
                    continue
                
                # Extract language instruction
                language_instruction = ""
                if 'conversations' in entry and entry['conversations']:
                    for conv in entry['conversations']:
                        if conv.get('from') == 'human' and 'value' in conv:
                            # Extract text between USER: and ASSISTANT:
                            match = re.search(r'USER:(.*?)ASSISTANT:', conv['value'])
                            if match:
                                language_instruction = match.group(1).strip()
                
                # Get raw actions
                raw_action = None
                if 'conversations' in entry and entry['conversations']:
                    for conv in entry['conversations']:
                        if conv.get('from') == 'gpt' and 'raw_actions' in conv:
                            raw_action = conv['raw_actions']
                
                # If no action is found, use a default
                if raw_action is None:
                    raw_action = [0.0] * 10  # Default action with 10 zeros
                else:
                    # Pad action to match expected shape of (10,)
                    if len(raw_action) == 7:
                        # Add 3 zeros at the end for the missing values
                        raw_action = raw_action + [0.0, 0.0, 0.0]
                
                # Create language embedding 
                if language_instruction:
                    language_embedding = self._embed([language_instruction])[0].numpy()
                else:
                    # Create a zero embedding if no instruction
                    language_embedding = np.zeros(512, dtype=np.float32)
                
                episode.append({
                    'observation': {
                        'image': image_array,
                        'wrist_image': image_array,  # Use the same image as a placeholder for wrist_image
                        'state': np.zeros(10, dtype=np.float32),  # Add state with zeros
                    },
                    'action': raw_action,
                    'discount': 1.0,
                    'reward': float(i == (len(entries) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(entries) - 1),
                    'is_terminal': i == (len(entries) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            
            # Skip empty episodes
            if not episode:
                continue
                
            yield str(episode_id), {
                'steps': episode,
                'episode_metadata': {
                    'file_path': key,
                }
            }
            episode_id += 1
