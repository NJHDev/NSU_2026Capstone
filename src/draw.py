import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_pose(frame, pose_landmarks):
    """팔을 포함한 기본 포즈 시각화"""
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
    )
