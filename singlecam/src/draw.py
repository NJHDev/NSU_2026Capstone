import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def _to_xy(frame, lm):
    h, w = frame.shape[:2]
    return int(lm.x * w), int(lm.y * h)

def draw_pose(frame, pose_landmarks, left_angles=None, right_angles=None):
    """
    팔(어깨-팔꿈치-손목) 표시 + 각도 라벨(El, Sh, Wr).
    left_angles/right_angles: (elbow, shoulder, wrist, visibility)
    """
    idx = mp_pose.PoseLandmark

    # 좌표
    Ls = pose_landmarks.landmark[idx.LEFT_SHOULDER]
    Le = pose_landmarks.landmark[idx.LEFT_ELBOW]
    Lw = pose_landmarks.landmark[idx.LEFT_WRIST]
    Rs = pose_landmarks.landmark[idx.RIGHT_SHOULDER]
    Re = pose_landmarks.landmark[idx.RIGHT_ELBOW]
    Rw = pose_landmarks.landmark[idx.RIGHT_WRIST]

    ls, le, lw = _to_xy(frame, Ls), _to_xy(frame, Le), _to_xy(frame, Lw)
    rs, re, rw = _to_xy(frame, Rs), _to_xy(frame, Re), _to_xy(frame, Rw)

    # 팔 라인/조인트
    color = (0, 255, 0)
    for a, b in [(ls, le), (le, lw), (rs, re), (re, rw)]:
        cv2.line(frame, a, b, color, 2)
    for p in (ls, le, lw, rs, re, rw):
        cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)

    # 각도 라벨
    if left_angles is not None:
        L_el, L_sh, L_wr, _ = left_angles
        lx, ly = le[0] - 10, le[1] - 10
        cv2.putText(frame, f"El:{L_el:0.0f}", (lx, ly - 0),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sh:{L_sh:0.0f}", (lx, ly - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Wr:{L_wr:0.0f}", (lx, ly - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if right_angles is not None:
        R_el, R_sh, R_wr, _ = right_angles
        rx, ry = re[0] + 10, re[1] - 10
        cv2.putText(frame, f"El:{R_el:0.0f}", (rx, ry - 0),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sh:{R_sh:0.0f}", (rx, ry - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Wr:{R_wr:0.0f}", (rx, ry - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
