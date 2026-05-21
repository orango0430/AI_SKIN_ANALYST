"""MediaPipe 단독 진단 — 어디서 hang하는지 확인."""
import sys
import time

print("1) Python 시작", flush=True)

import cv2
print(f"2) OpenCV: {cv2.__version__}", flush=True)

import numpy as np
print("3) NumPy OK", flush=True)

import mediapipe as mp
print(f"4) MediaPipe: {mp.__version__}", flush=True)

print("5) FaceMesh 초기화 중…", flush=True)
t = time.time()
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
)
print(f"6) FaceMesh 초기화 완료 ({time.time() - t:.2f}s)", flush=True)

print("7) 더미 이미지 (640x480) 생성", flush=True)
dummy = np.full((480, 640, 3), 128, dtype=np.uint8)

print("8) process() 시작…", flush=True)
t = time.time()
result = face_mesh.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
print(f"9) process() 완료 ({time.time() - t:.2f}s)", flush=True)
print(f"10) 결과: 얼굴 감지={result.multi_face_landmarks is not None}", flush=True)

face_mesh.close()
print("11) 끝", flush=True)
