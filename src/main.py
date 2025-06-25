import cv2
import os
import argparse
from detector import Detector
from tracker import Tracker

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '../assets')
INPUT_VIDEO = os.path.join(ASSETS_DIR, '15sec_input_720p.mp4')
WEIGHTS_PATH = os.path.join(ASSETS_DIR, 'best.pt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output')
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, 'output1.mp4')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_boxes(frame, tracks):
    for obj in tracks:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        track_id = obj['track_id']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for raw detections
        cv2.putText(frame, f'{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    return frame

def main():
    parser = argparse.ArgumentParser(description='Player Re-ID Pipeline')
    parser.add_argument('--max-frames', type=int, default=None, help='Process only the first N frames (for quick testing)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    detector = Detector(WEIGHTS_PATH)
    tracker = Tracker()
    frame_count = 0
    seen_ids = set()
    last_ids = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        frame = draw_boxes(frame, tracks)
        out.write(frame)
        frame_count += 1
        # Save frame if a new ID is detected
        current_ids = set(obj['track_id'] for obj in tracks)
        new_ids = current_ids - last_ids
        for new_id in new_ids:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'new_id_{new_id}_frame{frame_count}.jpg'), frame)
        seen_ids.update(current_ids)
        last_ids = current_ids
        if frame_count % 10 == 0:
            print(f'Processed {frame_count} frames')
        if args.max_frames is not None and frame_count >= args.max_frames:
            print(f'Reached max frames: {args.max_frames}')
            break
    cap.release()
    out.release()
    print(f'Total frames written: {frame_count}')
    print(f'Unique IDs seen: {sorted(seen_ids)}')
    print(f'Output saved to {OUTPUT_VIDEO}')

if __name__ == '__main__':
    main() 