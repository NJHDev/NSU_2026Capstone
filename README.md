# 🧍‍♂️ NSU_2026CAPSTONE

---

## 📦 프로젝트 구조
```
NSU_2026CAPSTONE/
├─ main.py                # 실행 파일
├─ src/
│  ├─ camera_utils.py     # 카메라 검색 및 선택
│  ├─ angles.py           # 각도 계산, EMA 필터
│  └─ draw.py             # 포즈/팔 시각화
├─ logs/                  # CSV 로그 자동 저장 폴더
├─ requirements.txt       # 필요한 라이브러리 목록
└─ README.txt             # (이 파일)
```

---

## ⚙️ 1. 설치 방법

### Python 버전
- **Python 3.9.x 권장** (MediaPipe는 3.13 이상 미지원)

### pip 설치
pip3 install -r requirements.txt

---

## 🚀 2. 실행 방법
python3 main.py

실행하면 터미널에 아래처럼 출력됩니다:
```s
[INFO] Detected OS: ~~~
=== 연결된 카메라 목록 ===
0: ~~~
1: ~~~~
.
.
.
========================
사용할 카메라 번호를 입력하세요 :
```

**→ 번호 입력 후 엔터**

---

## 🕹️ 3. 단축키 명령어

| 키 (Mac 기준) | 기능 설명 | Windows/Linux 대응 |
|----------------|------------|--------------------|
| **⌘ + Q**      | 프로그램 종료 | Q 키 |
| **⌘ + S**      | 화면 좌우 반전 (거울 모드 ON/OFF) | S 키 |
| **⌘ + L**      | CSV 로그 저장 ON/OFF | L 키 |
| **⌘ + R**      | ROM(최대/최소 각도) 초기화 | R 키 |

> ⚠️ 주의: VSCode 내 실행 시 ⌘ 단축키가 IDE 단축키와 충돌할 수 있습니다.  
> 이 경우 터미널 창을 클릭한 뒤 **소문자 q, s, l, r** 로 입력하세요.

---

## 📈 4. 화면 표시 정보

| 구분 | 설명 |
|------|------|
| **FPS** | 초당 프레임 |
| **Mirror** | 좌우 반전 상태 |
| **Log** | CSV 로그 기록 상태 |
| **LEFT / RIGHT** | 팔 각도 (Elbow, Abduction, Flexion) 및 visibility |
| **ROM (Range of Motion)** | 관찰된 각도의 최소/최대 범위 |

---

## 📂 5. 로그 파일 (CSV)
- 저장 위치: logs/angles_YYYYMMDD_HHMMSS.csv
- 열 구성:
  time, L_el, L_abd, L_flex, R_el, R_abd, R_flex, L_vis, R_vis

---

## 🎨 6. 시각화 설명
현재는 **전신 포즈**가 표시되지만, 필요 시 src/draw.py에서 팔 부분만 시각화하도록 수정 가능:

# draw.py 내 draw_pose() 대신
from draw import draw_arms_only
draw_arms_only(frame, res.pose_landmarks)

---

## 🔧 7. 문제 해결

| 문제 | 해결 방법 |
|------|------------|
| 카메라가 켜지지 않음 | macOS → **시스템 설정 > 개인정보보호 > 카메라 > Terminal 허용** |
| Import 오류 (cv2, mediapipe) | python3 -m pip install opencv-python mediapipe |
| 프레임 지연 | 해상도 낮추기 또는 FPS 제한 (camera_utils.py 수정) |

---

## 🧠 8. 향후 확장 계획
- [ ] 멀티 카메라 동기화 지원 (삼각측량 기반)
- [ ] 3D 팔 궤적 시각화
- [ ] 실시간 각도 그래프 표시
- [ ] GUI 버전 (PyQt / WebUI)

---

## 👨‍💻 제작자
Author: *njhdev*  
Tech Stack: Python · MediaPipe · OpenCV  
Version: 0.0.1
