import argparse
import subprocess
import warnings

import cv2
import librosa
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch

from .syncnet import SyncNet_color
from utils import process_slices, resize_frames, compute_iou, save_bbox_video, crop_faces, write_individual_video

warnings.filterwarnings("ignore")

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='input.mp4', help='path to the video file')
parser.add_argument('--audio', type=str, default='audio.aac', help='path to the audio file')
parser.add_argument('--output', type=str, default='output.mp4', help='path to the output file')
parser.add_argument('--model', type=str, default='lipsync_expert.pth', help='path to the model file')
parser.add_argument('--save_face', type=bool, default=False, help='save individual face video')
args = parser.parse_args()

model = SyncNet_color()
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Open the video file
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)

wav, sr = librosa.load(args.video, sr=16000)

# save original audio force rewrite
subprocess.call(f"ffmpeg -y -i {args.video} -vn -acodec copy {args.audio}", shell=True)

faces_frames = []
faces_bboxes = []

original_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    original_frames.append(frame)

    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find faces
    results = face_detection.process(rgb_frame)

    faces_frame = []
    faces_bbox = []

    # If faces are detected
    if results.detections:
        for face_idx, detection in enumerate(results.detections):
            expansion_factor = 0.2  # expand the bounding box by 20%
            face, bbox = crop_faces(rgb_frame, detection, expansion_factor)
            faces_frame.append(face)
            faces_bbox.append(bbox)

    # faces not necessarily match with previous frame, compute iou to find the best match
    # if faces_frames is empty, just append the current faces_frame
    if len(faces_frames) == 0:
        for i in range(len(faces_frame)):
            faces_frames.append([faces_frame[i]])
            faces_bboxes.append([faces_bbox[i]])
        continue

    # compute iou
    iou = np.zeros((len(faces_frames), len(faces_frame)))
    for i in range(len(faces_frames)):
        for j in range(len(faces_frame)):
            iou[i, j] = compute_iou(faces_bboxes[i][-1], faces_bbox[j])

    # find the best match
    max_iou = np.argmax(iou, axis=1)
    for i in range(len(max_iou)):
        # bgr frame
        bgr_frame = cv2.cvtColor(faces_frame[max_iou[i]], cv2.COLOR_RGB2BGR)
        faces_frames[i].append(bgr_frame)
        faces_bboxes[i].append(faces_bbox[max_iou[i]])

cap.release()

if args.save_face:
    write_individual_video(faces_frames, args.audio)

distances = [[] for _ in range(len(faces_frames))]
for idx, face_frames in enumerate(faces_frames):
    face_frames = np.array(resize_frames(face_frames))
    video_slices, audio_slices = process_slices(face_frames, wav)

    # Feed the inputs into the model
    for vid, aud in zip(video_slices, audio_slices):
        with torch.no_grad():
            audio_embedding, face_embedding = model(aud[None, None, :], vid[None, :])
            distance = torch.nn.functional.cosine_similarity(audio_embedding, face_embedding)
        distances[idx].append(distance.item())

# smoothing the cosine similarity distance as lip-sync activation signal
smoothed_distances = []
window_size = 75
for idx, series in enumerate(distances):
    smoothed_distance = np.convolve(series, np.ones(window_size) / window_size, mode='same')
    smoothed_distances.append(smoothed_distance)
    plt.plot(smoothed_distance, label=f'face {idx}')

plt.legend()
plt.title("Lip-sync activation signal\nargmax is the talking subject, clearly separated in the plot")
plt.savefig('sync_signals.png')

save_bbox_video(original_frames, faces_bboxes, smoothed_distances, args.output, args.audio)
