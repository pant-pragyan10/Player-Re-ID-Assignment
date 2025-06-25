# Player Re-Identification in Single Video Feed

## Overview
This project detects and tracks soccer players in a single video feed, assigning consistent IDs even when players leave and re-enter the frame. It uses YOLOv11 for detection and DeepSORT for tracking and re-identification.

## Project Structure
```
player_reid_single_feed/
├── assets/
│   ├── 15sec_input_720p.mp4
│   └── yolov11.pt
├── src/
│   ├── detector.py
│   ├── tracker.py
│   └── main.py
├── output/
│   └── output.mp4
├── requirements.txt
├── README.md
└── report.md
```

## Setup
1. **Clone the repository and navigate to the project directory.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place the input video and YOLOv11 weights in the `assets/` folder.**

## Usage
Run the main pipeline:
```bash
python src/main.py
```
The annotated output video will be saved to `output/output.mp4`.

## Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLOv11
- NumPy
- DeepSORT

## Notes
- Ensure the correct paths for the video and weights in `main.py`.
- For best results, use the provided YOLOv11 weights and a short soccer video as input. 

## Example Usage
```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt

# Download and place your model weights and input video in player_reid_single_feed/assets/
# e.g., best.pt and 15sec_input_720p.mp4

# Run the pipeline
python player_reid_single_feed/src/main.py

# Output will be in player_reid_single_feed/output/output1.mp4
``` 