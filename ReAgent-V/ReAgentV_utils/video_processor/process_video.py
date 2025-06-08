import numpy as np
from decord import VideoReader, cpu

def load_video_frames(video_path, fps=1, force_sample=False):
    max_frames_num = 64
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def save_frames(frames, file_name):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/{file_name}/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths
    