from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        # Tuned parameters: longer max_age, quicker confirmation, limited embedding budget
        self.tracker = DeepSort(max_age=60, n_init=1, nms_max_overlap=1.0, nn_budget=100)

    def update(self, detections, frame):
        # detections: [N, 5] array (x1, y1, x2, y2, conf)
        # frame: current image (for appearance embedding)
        # DeepSort expects: ([x1, y1, x2, y2], confidence, class_id)
        dets = [([int(det[0]), int(det[1]), int(det[2]), int(det[3])], float(det[4]), 0) for det in detections]
        tracks = self.tracker.update_tracks(dets, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            results.append({'track_id': track_id, 'bbox': ltrb})
        return results 