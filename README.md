# Player Re-Identification in Single Video Feed
<img width="1335" alt="image" src="https://github.com/user-attachments/assets/64a36c77-e375-4a2c-8844-d04447522d35" />
<img width="1306" alt="image" src="https://github.com/user-attachments/assets/2673c410-a4a1-44be-94cd-c3044239f6ee" />
<img width="1306" alt="image" src="https://github.com/user-attachments/assets/3520bdca-ea06-4e2a-a75e-d416df2ac067" />
<img width="1306" alt="image" src="https://github.com/user-attachments/assets/72ba79d5-f71f-49e6-adbf-60ccde12617b" />


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

## Creative Extensions & Future Work

- **Appearance-based Re-ID:** Use color histograms or deep appearance features to improve ID consistency when players re-enter the frame.
- **Player Gallery:** Save cropped images of each unique player to visually inspect re-identification quality.
- **Trajectory Visualization:** Plot player movement heatmaps or trajectories on a 2D field.
- **Interactive Notebook:** Provide a Jupyter notebook for step-by-step demo and visualization.
- **Metrics & Analytics:** Output per-frame stats and ID assignment confidence scores.
- **Web Demo:** Build a simple web app (e.g., Streamlit) for interactive video upload and visualization.
- **Config File:** Move parameters to a YAML/JSON config for easy tuning.
- **Batch/Parallel Processing:** Speed up processing for longer videos.

These ideas can further improve accuracy, usability, and presentation of the project. 
