import os
import subprocess

import cv2
import librosa
import numpy as np
import torch
from scipy import signal

# Hyperparameters
num_mels = 80
sample_rate = 16000
hop_size = 200
win_size = 800
n_fft = 800
min_level_db = -100
ref_level_db = 20
preemphasize = True
preemphasis_val = 0.97
fmin = 55
fmax = 7600
signal_normalization = True
symmetric_mels = True
max_abs_value = 4.

video_T = 5
audio_T = 16
fps = 25
video_res = 96


def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def melspectrogram(wav):
    preemphasize_wav = preemphasis(wav, preemphasis_val, preemphasize)
    D = librosa.stft(y=preemphasize_wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db

    if signal_normalization:
        return _normalize(S)
    return S


def _linear_to_mel(spectrogram):
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    return np.dot(mel_basis, spectrogram)


def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    if symmetric_mels:
        return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                       -max_abs_value, max_abs_value)
    else:
        return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)


def resize_frames(frames, target_res=video_res):
    """Resize a list of frames to a target resolution.

    Args:
        frames (list): List of frames, each represented as a Numpy array of shape (h, w, 3).
        target_res (size): Target resolution as a tuple (size, size).

    Returns:
        list: Resized frames.
    """
    resized_frames = []

    for frame in frames:
        # OpenCV's resize function takes the target resolution in (width, height) format
        resized_frame = cv2.resize(frame, (target_res, target_res))
        resized_frames.append(resized_frame)

    return resized_frames


def crop_faces(rgb_frame, detection, expansion_factor=0):
    """Crop the face from the frame.

    Args:
        rgb_frame (np.array): RGB frame.
        detection (face_detection.FaceDetection): Face detection object.
        expansion_factor (float): Amount to expand the bounding box by.

    Returns:
        np.array: Cropped face.
        tuple: Bounding box coordinates.
    """
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = rgb_frame.shape

    # Calculate bounding box dimensions
    x = int(bboxC.xmin * iw)
    y = int(bboxC.ymin * ih)
    width = int(bboxC.width * iw)
    height = int(bboxC.height * ih)

    # Calculate the amount to expand the bounding box
    expand_w = int(width * expansion_factor)
    expand_h = int(height * expansion_factor)

    # Recalculate the bounding box with the expansion
    x = max(0, x - expand_w)
    y = max(0, y - expand_h)
    width += 2 * expand_w
    height += 2 * expand_h

    # Ensure that the bounding box is within the image boundaries
    x = min(iw - 1, x)
    y = min(ih - 1, y)
    width = min(iw - x, width)
    height = min(ih - y, height)

    # Crop the face from the frame
    face = rgb_frame[y:y + height, x:x + width]

    return face, (x, y, width, height)


def get_video_slices(video, video_T, padding=0):
    """Get slices of the video.

    Args:
        video (np.array): Video represented as a Numpy array of shape (T, H, W, 3).
        video_T (int): Number of frames to include in each slice.
        padding (int): Amount of padding to add to the video before slicing.

    Returns:
        list: List of video slices, each represented as a Numpy array of shape (3 * video_T, H/2, W).
    """
    video_size = video.shape[1]
    video_slices = []

    padded_video = np.pad(video, ((padding, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
    for start_idx in range(0, len(padded_video) - video_T + 1 - padding):
        window = padded_video[start_idx:start_idx + video_T]
        x = window.transpose(1, 2, 3, 0)  # H, W, C, T
        x = x.reshape(video_size, video_size, -1) / 255.  # H, W, C * T
        x = x.transpose(2, 0, 1)  # C * T, H, W
        x = x[:, x.shape[1] // 2:]  # C * T, H/2, W
        video_slices.append(torch.FloatTensor(x))
    return video_slices


def get_audio_slices(spec, audio_T, num_frames, padding=0):
    """Get slices of the audio spectrogram.

    Args:
        spec (np.array): Audio spectrogram represented as a Numpy array of shape (T, 80).
        audio_T (int): Number of frames to include in each slice.
        num_frames (int): Number of frames in the video.
        padding (int): Amount of padding to add to the spectrogram before slicing.

    Returns:
        list: List of audio slices, each represented as a Numpy array of shape (80, audio_T).
    """
    audio_slices = []

    padded_spec = np.pad(spec, ((padding, padding), (0, 0)), mode='constant')
    for start_frame in range(num_frames):
        start_idx = int(80. * (start_frame / float(fps)))
        end_idx = start_idx + audio_T
        if end_idx > len(padded_spec):
            break
        audio_slices.append(torch.FloatTensor(padded_spec[start_idx: end_idx, :]).T)
    return audio_slices


def process_slices(video, audio):
    """
    :param video: (T, H, W, C)
    :param audio: (T, )
    :return:
    """
    video_slices = get_video_slices(video, video_T, padding=video_T - 1)
    orig_mel = melspectrogram(audio).T
    audio_slices = get_audio_slices(orig_mel, audio_T, num_frames=len(video_slices), padding=audio_T - 1)
    return video_slices, audio_slices


def compute_iou(bbox1, bbox2):
    """Compute the intersection over union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersect_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersect_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def write_individual_video(faces_frames, audio_path):
    """Write the individual face videos to disk."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, face_frames in enumerate(faces_frames):
        output_filename = f'face_{i}.mp4'
        out = cv2.VideoWriter(output_filename, fourcc, fps, (video_res, video_res))

        face_frames = np.array(resize_frames(face_frames, video_res))

        for frame in face_frames:
            out.write(frame)

        out.release()


    # Iterate through each face sequence and add the audio
    for i in range(len(faces_frames)):
        output_filename = f'face_{i}.mp4'
        output_filename_audio = f'face_{i}_audio.mp4'

        # Add the audio to the video
        command = f"ffmpeg -y -i {output_filename} -i {audio_path} -c:a aac -strict experimental {output_filename_audio}"
        subprocess.call(command, shell=True)

        # wait for subprocess to finish before deleting
        os.remove(output_filename)


def save_bbox_video(original_frames, faces_bboxes, distances, save_path, audio_path):
    frames = []
    for idx, frame in enumerate(original_frames[:len(distances[0])]):
        target_bbox = np.argmax(np.array([distances[i][idx] for i in range(len(distances))]))
        x, y, w, h = faces_bboxes[target_bbox][idx]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, C = original_frames[0].shape
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    command = f"ffmpeg -y -i {save_path} -i {audio_path} -c:a aac -strict experimental {save_path[:-4]}_audio.mp4"
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    # Random input
    video_frames = np.random.rand(250, 96, 96, 3)
    audio_input = np.random.rand(16000 * 10)

    video_window_length = 5
    audio_window_length = 16

    video_windows, audio_windows = process_slices(video_frames, audio_input)
    print(len(video_windows))
    print(len(audio_windows))

    print(video_windows[0].shape)
    print(audio_windows[0].shape)
