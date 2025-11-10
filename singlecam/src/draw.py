import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def _to_xy(frame, lm):
    h, w = frame.shape[:2]
    return int(lm.x * w), int(lm.y * h)

def draw_pose(frame, pose_landmarks, left_angles=None, right_angles=None):
    """
    팔(어깨-팔꿈치) 표시 + 각도 라벨(El, Sh). (손목 각도 삭제됨)
    left_angles/right_angles: (elbow, shoulder, visibility)
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
    # 어깨-팔꿈치, 팔꿈치-손목 라인 그리기 유지
    for a, b in [(ls, le), (le, lw), (rs, re), (re, rw)]:
        cv2.line(frame, a, b, color, 2)
    # 조인트 그리기 유지
    for p in (ls, le, lw, rs, re, rw):
        cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)

    # 🚨 상체 회전 선 그리기 (좌우 어깨 연결)
    cv2.line(frame, ls, rs, (255, 0, 255), 2) # 마젠타 색
    cv2.circle(frame, ls, 7, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, rs, 7, (255, 0, 255), -1, cv2.LINE_AA)

    # 각도 라벨
    if left_angles is not None:
        L_el, L_sh, _ = left_angles # L_wr 제거
        lx, ly = le[0] - 10, le[1] - 10
        cv2.putText(frame, f"El:{L_el:0.0f}", (lx, ly - 0),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sh:{L_sh:0.0f}", (lx, ly - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        # 손목 각도 라벨 삭제

    if right_angles is not None:
        R_el, R_sh, _ = right_angles # R_wr 제거
        rx, ry = re[0] + 10, re[1] - 10
        cv2.putText(frame, f"El:{R_el:0.0f}", (rx, ry - 0),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sh:{R_sh:0.0f}", (rx, ry - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        # 손목 각도 라벨 삭제