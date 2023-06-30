# SyncNet Speaker Diarization

## Introduction
Audio-visual speaker diarization is the process of detecting "who spoke when" using both auditory and visual signals. It aims to determine "who spoke when" in multi-speaker scenarios.

Speaker diarization is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker. It can enhance the readability of an automatic speech transcription by structuring the audio stream into speaker turns.

This project aims to perform lip sync detection using the modified Lip-Sync Expert SyncNet from Wav2Lip
## Getting Started

### Dependencies
- PyTorch
- OpenCV
- librosa
- MediaPipe
- matplotlib
- ffmpeg

### Project Structure
The project mainly consists of a Python script for lip sync detection and analysis. The script uses a pre-trained SyncNet model (not included in this repository).

- main.py: The main script for lip sync detection and analysis.
- syncnet.py: SyncNet model file.
- utils.py: Contains utility functions used in the script.
- input.mp4: Sample input video.


### Usage
To use the lip sync detection, follow these steps:

Download the pre-trained SyncNet model pth file from Wav2Lip repository [Here](https://github.com/Rudrabha/Wav2Lip) and [Here](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)
and place it in the same directory as the script


Run the script with the required arguments. You can change the default arguments in the script if needed:
```
python main.py --video input.mp4 --audio audio.aac --output output.mp4 --model lipsync_expert.pth --save_face True
```
Note: The `--save_face` argument is a boolean value that determines whether to save individual face videos. By default, it is set to True.

The script will output a plot of the lip sync activation signal for each face detected in the video and save it as 'sync_signals.png'. Note the argmax is the talking subject, which is clearly separated in the plot

### Features

1. Used IoU to track faces from consecutive frames
2. Added bounding box expansion
3. Padded video and audio frames to make segment length equal to original video length
4. smoothed sync signal using a moving average filter

