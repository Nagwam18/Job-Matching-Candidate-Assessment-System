import cv2
from deepface import DeepFace
from collections import Counter

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frames_samples = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = gray.mean()
        if mean_intensity < 10:
            continue
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return {"confidence": 0, "anxiety": 0, "emotions_distribution": {}, "frames_sampled": 0}

    step = max(1, len(frames) // frames_samples)
    selected_frames = [frames[i] for i in range(0, len(frames), step)][:frames_samples]

    emotions = []
    for frame in selected_frames:
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            emotions.append(result.get("dominant_emotion", "unknown"))
        except Exception:
            emotions.append("unknown")

    confidence_emotions = {"happy", "neutral", "calm"}
    anxiety_emotions = {"fear", "sad", "angry", "disgust"}

    distribution = Counter(emotions)

    confidence_count = sum(distribution[e] for e in confidence_emotions)
    anxiety_count = sum(distribution[e] for e in anxiety_emotions)
    total = confidence_count + anxiety_count

    confidence_score = round((confidence_count / total) * 100, 1) if total > 0 else 0.0
    anxiety_score = round((anxiety_count / total) * 100, 1) if total > 0 else 0.0

    return {
        "confidence": confidence_score,
        "anxiety": anxiety_score,
        "emotions_distribution": dict(distribution),
        "frames_sampled": len(selected_frames)
    }
