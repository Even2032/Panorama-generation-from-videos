### README.md

# Panorama Generator from Video

This project implements a software system to automatically generate panorama images from video input. Using computer vision techniques and a graphical user interface, the system extracts frames from videos, detects and matches features across these frames, and stitches them together to create a seamless panorama.

## Features

- Video Frame Extraction: Automatically extracts relevant frames from the input video.
- Feature Detection and Matching: Utilizes SIFT or ORB algorithms for robust feature detection and matching.
- Image Stitching: Generates panoramas by stitching together the processed frames.
- GUI: Offers an easy-to-use interface for uploading videos, monitoring processing, and viewing results.
- Image Adjustments: Includes functionality to adjust image properties such as brightness and contrast, apply Gaussian blur, and sharpen images.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- Pillow
- Tkinter (usually included with Python)


## Usage

Run the application by executing:
```
python panorama.py
```
