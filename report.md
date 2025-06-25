# Report: Player Re-Identification in Single Video Feed

## Approach
- Used YOLOv11 for player detection in each frame of the video.
- Applied DeepSORT for tracking and re-identification, ensuring consistent player IDs even when players leave and re-enter the frame.
- Tuned DeepSORT parameters for improved ID consistency.
- Combined detection and tracking to annotate the video with bounding boxes and player IDs.

## Challenges
- Maintaining consistent IDs when players are occluded or leave/re-enter the frame.
- Ensuring real-time performance with high-resolution video.
- Handling appearance changes and similar player uniforms.

## Results
- Successfully assigned consistent IDs to players throughout the 15-second video.
- Output video (`output/output1.mp4`) shows annotated bounding boxes and IDs.
- The pipeline is modular and can be extended for longer videos or more advanced re-ID features. 