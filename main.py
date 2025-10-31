import os, sys, time, csv, platform
from datetime import datetime
import cv2
import mediapipe as mp

from src.camera_utils import select_camera_index, open_camera
from src.angles import compute_angles, EMA
from src.draw import draw_pose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def main():
    cam_index = select_camera_index()
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"카메라 {cam_index} 열기 실패.")
        sys.exit(1)

    mirror = False
    log_on = False
    os.makedirs("logs", exist_ok=True)
    csv_writer = None
    csv_file = None

    ema = {k: EMA(0.25) for k in ["L_el", "L_abd", "L_flex", "R_el", "R_abd", "R_flex"]}
    rom = {k: [999, -999] for k in ema.keys()}

    fps_t = time.time()
    fps = 0.0

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            if res.pose_landmarks:
                draw_pose(frame, res.pose_landmarks)

                L_el, L_abd, L_flex, L_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "left")
                R_el, R_abd, R_flex, R_vis = compute_angles(res.pose_landmarks, res.pose_world_landmarks, "right")

                L_el = ema["L_el"].update(L_el)
                L_abd = ema["L_abd"].update(L_abd)
                L_flex = ema["L_flex"].update(L_flex)
                R_el = ema["R_el"].update(R_el)
                R_abd = ema["R_abd"].update(R_abd)
                R_flex = ema["R_flex"].update(R_flex)

                vals = {"L_el": L_el, "L_abd": L_abd, "L_flex": L_flex,
                        "R_el": R_el, "R_abd": R_abd, "R_flex": R_flex}
                for k, v in vals.items():
                    if v is None:
                        continue
                    rom[k][0] = min(rom[k][0], v)
                    rom[k][1] = max(rom[k][1], v)

                def put(y, text):
                    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

                put(28, f"OS:{platform.system()}  Cam:{cam_index}  FPS:{fps:0.1f}   Mirror:[{'ON' if mirror else 'OFF'}]   Log:[{'ON' if log_on else 'OFF'}]")
                put(56, f"LEFT  Elbow:{'-' if L_el is None else f'{L_el:5.1f}'}  Abd:{'-' if L_abd is None else f'{L_abd:5.1f}'}  Flex:{'-' if L_flex is None else f'{L_flex:5.1f}'}  vis:{L_vis:0.2f}")
                put(84, f"RIGHT Elbow:{'-' if R_el is None else f'{R_el:5.1f}'}  Abd:{'-' if R_abd is None else f'{R_abd:5.1f}'}  Flex:{'-' if R_flex is None else f'{R_flex:5.1f}'}  vis:{R_vis:0.2f}")
                put(120, f"L ROM El:{rom['L_el'][0]:.0f}/{rom['L_el'][1]:.0f}  Abd:{rom['L_abd'][0]:.0f}/{rom['L_abd'][1]:.0f}  Flex:{rom['L_flex'][0]:.0f}/{rom['L_flex'][1]:.0f}")
                put(148, f"R ROM El:{rom['R_el'][0]:.0f}/{rom['R_el'][1]:.0f}  Abd:{rom['R_abd'][0]:.0f}/{rom['R_abd'][1]:.0f}  Flex:{rom['R_flex'][0]:.0f}/{rom['R_flex'][1]:.0f}")

                if log_on:
                    if csv_writer is None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = f"logs/angles_{ts}.csv"
                        csv_file = open(path, "w", newline="", encoding="utf-8")
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(
                            ["time", "L_el", "L_abd", "L_flex", "R_el", "R_abd", "R_flex", "L_vis", "R_vis"]
                        )
                    csv_writer.writerow([time.time(), L_el, L_abd, L_flex, R_el, R_abd, R_flex, L_vis, R_vis])

            now = time.time()
            dt = now - fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            fps_t = now

            cv2.imshow("Pose Angles", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                mirror = not mirror
            elif k == ord("l"):
                log_on = not log_on
                if not log_on and csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
            elif k == ord("r"):
                rom = {k: [999, -999] for k in rom.keys()}

    if csv_file:
        csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
