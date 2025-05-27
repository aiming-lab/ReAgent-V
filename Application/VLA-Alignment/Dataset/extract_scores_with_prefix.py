import json
import os
import shutil
import re
from collections import defaultdict

def extract_scores_from_json(json_str):
    """Extract scores from the JSON-formatted model answer."""
    try:
        # Check if the answer is wrapped in a code block
        if "```json" in json_str:
            # Extract the JSON content from within the code block
            match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if match:
                json_content = match.group(1)
            else:
                json_content = json_str
        else:
            json_content = json_str
            
        # Parse the JSON
        data = json.loads(json_content)
        
        # Extract the scores
        overall_score = data.get("overall_assessment", {}).get("score", 0)
        success_score = data.get("success_or_failure", {}).get("score", 0)
        
        return overall_score, success_score
    except Exception as e:
        print(f"Error extracting scores: {e}")
        return None, None  # Return None for both scores when parsing fails

def main():
    # Define paths
    jsonl_path = "example.jsonl"
    output_base = "extracted_results"
    positive_dir = os.path.join(output_base, "positive")
    negative_dir = os.path.join(output_base, "negative")
    
    # Create output directories if they don't exist
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    
    # Dictionary to store results by task and episode
    results_by_episode = defaultdict(list)
    
    # Track which episodes have success videos
    episodes_with_success = set()
    
    # Parse the JSONL file
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                video_path = entry["video_path"]
                task_description = entry["task_description"]
                model_answer = entry["model_answer_final"]
                
                # Extract task and episode from the path
                # New format: /home/yaofeng/GRAPE/Simpler-env/results/task_name/episode_X/success_episode_X_num_Y.mp4
                # or: /home/yaofeng/GRAPE/Simpler-env/results/task_name/episode_X/failure_episode_X_num_Y.mp4
                path_parts = video_path.split('/')
                task_name = path_parts[-3]
                episode = path_parts[-2]
                filename = path_parts[-1]
                
                # Check if this is a success or failure video
                is_success = filename.startswith("success_")
                
                # Track episodes with success videos
                if is_success:
                    episodes_with_success.add((task_name, episode))
                
                # Extract scores
                overall_score, success_score = extract_scores_from_json(model_answer)
                
                # Skip entries where parsing failed
                if overall_score is None or success_score is None:
                    print(f"Skipping entry with parsing issue: {video_path}")
                    continue
                
                # Store the result
                results_by_episode[(task_name, episode)].append({
                    "video_path": video_path,
                    "overall_score": overall_score,
                    "success_score": success_score,
                    "is_success": is_success
                })
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Dictionary to track saved counts per task
    saved_counts = defaultdict(int)
    max_pairs_per_task = 40

    # Calculate averages and find highest/lowest scores for each episode
    positive_counter = 1
    negative_counter = 1
    
    for (task, episode), results in results_by_episode.items():
        if not results:
            continue
            
        # Skip episodes that don't have any success videos
        if (task, episode) not in episodes_with_success:
            print(f"Skipping episode without success videos: {task}/{episode}")
            continue
            
        # Calculate average scores
        avg_overall = sum(r["overall_score"] for r in results) / len(results)
        avg_success = sum(r["success_score"] for r in results) / len(results)
        
        print(f"Task: {task}")
        print(f"Episode: {episode}")
        print(f"Average Overall Score: {avg_overall:.2f}")
        print(f"Average Success Score: {avg_success:.2f}")
        print("=" * 50)
        
        # Find videos with highest and lowest overall scores
        # If there are multiple with the same score, take the first one
        highest = max(results, key=lambda x: x["overall_score"])
        lowest = min(results, key=lambda x: x["overall_score"])
        
        # Create path for the .npy files
        highest_npy = highest["video_path"].replace(".mp4", ".npy")
        lowest_npy = lowest["video_path"].replace(".mp4", ".npy")
        
        # Check if the task has reached the maximum number of saved pairs
        if saved_counts[task] < max_pairs_per_task:
            if os.path.exists(highest_npy):
                positive_dest = os.path.join(positive_dir, f"{positive_counter:04d}.npy")
                shutil.copy(highest_npy, positive_dest)
                print(f"Copied {highest_npy} to {positive_dest}")
                positive_counter += 1
                saved_counts[task] += 1
            else:
                print(f"Warning: File not found: {highest_npy}")
            
            if os.path.exists(lowest_npy) and saved_counts[task] < max_pairs_per_task:
                negative_dest = os.path.join(negative_dir, f"{negative_counter:04d}.npy")
                shutil.copy(lowest_npy, negative_dest)
                print(f"Copied {lowest_npy} to {negative_dest}")
                negative_counter += 1
                saved_counts[task] += 1
            else:
                print(f"Warning: File not found: {lowest_npy}")
        else:
            print(f"Skipping additional pairs for task: {task}, limit reached.")

if __name__ == "__main__":
    main()
