import cv2
import mediapipe as mp
import math
import time
from pathlib import Path
import winsound
import tkinter as tk
from tkinter import filedialog

# ---------------- 초기 설정 ----------------
# Mediapipe 얼굴 검출 모델 로드
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# 저장 경로 설정 (바탕화면)
desktop_path = Path.home() / "Desktop"
desktop_path.mkdir(exist_ok=True)

# ---------------- 함수 정의 ----------------
def get_face_roll_angle(detection, w, h, mode="abs"):
    """얼굴의 기울기(Roll)를 계산하여 반환"""
    kps = detection.location_data.relative_keypoints
    right_eye = kps[0]
    left_eye  = kps[1]

    x1, y1 = right_eye.x * w, right_eye.y * h
    x2, y2 = left_eye.x * w, left_eye.y * h

    dx = x2 - x1
    dy = y1 - y2

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # [-90, 90] 범위로 정규화
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return abs(angle_deg) if mode == "abs" else angle_deg

def select_reference_image():
    """파일 탐색기를 열어 기준 사진 선택 및 분석"""
    root = tk.Tk()
    root.withdraw()  # 빈 창 숨기기
    
    file_path = filedialog.askopenfilename(
        title="기준 사진 선택",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("기준 사진을 선택하지 않았습니다.")
        return None

    img = cv2.imread(file_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {file_path}")
        return None

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_detector.process(rgb)

    if not res.detections:
        print("기준 사진에서 얼굴을 찾지 못했습니다.")
        return None

    angle = get_face_roll_angle(res.detections[0], w, h, mode="abs")
    print(f"기준 사진 선택 완료: {file_path}")
    print(f"기준 각도: {angle:.2f}도")
    return angle

# ---------------- 메인 실행 로직 ----------------
def run_auto_capture_camera(camera_source=0):
    """
    camera_source: 0이면 기본 웹캠, 1이면 외부 카메라, 또는 URL 문자열(IP Webcam)
    """
    ref_angle = select_reference_image()
    if ref_angle is None:
        return

    # 카메라 시작
    cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW) 
    # 주의: IP Webcam(URL) 사용 시 cv2.CAP_DSHOW 제거 필요할 수 있음 -> cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 연결을 확인하세요.")
        return

    print("카메라 열기 성공! (종료하려면 'q'를 누르세요)")

    captured = False
    ANGLE_TOLERANCE = 8.0      # 각도 허용 오차 (이 범위 내에 들어오면 촬영)
    MAX_DIFF_FOR_SIM = 45.0    # 유사도 계산을 위한 최대 각도 차이

    while True:
        ret, img = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        img_h, img_w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_res = face_detector.process(rgb)
        
        face_detected = False
        face_angle = None
        similarity = 0
        angle_ok = False

        if face_res.detections:
            face_detected = True
            detection = face_res.detections[0]
            mp_draw.draw_detection(img, detection)

            # 현재 얼굴 각도 계산
            face_angle = get_face_roll_angle(detection, img_w, img_h, mode="abs")
            
            # 기준 각도와 비교
            diff = abs(face_angle - ref_angle)
            sim = 1.0 - min(diff / MAX_DIFF_FOR_SIM, 1.0)
            similarity = int(sim * 100)

            # 허용 오차 내에 들어오면 촬영 조건 충족
            if diff <= ANGLE_TOLERANCE:
                angle_ok = True

        # ------ 화면 표시 (UI) ------
        status_color = (0, 255, 0) if angle_ok else (0, 0, 255)
        
        cv2.putText(img, f"Ref Angle: {ref_angle:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if face_detected:
            cv2.putText(img, f"Cur Angle: {face_angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(img, f"Similarity: {similarity}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 유사도 게이지 바 그리기
            bar_x, bar_y, bar_w, bar_h = 10, 110, 200, 20
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            fill_w = int(bar_w * similarity / 100)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

        # ------ 자동 촬영 로직 ------
        if face_detected and angle_ok and not captured:
            # 캡처 시각을 파일명으로 사용
            filename = desktop_path / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), img)
            print(f"✔ 촬영 완료! 저장됨: {filename}")
            
            # 찰칵 소리 (Windows 전용)
            winsound.Beep(1000, 200) 
            
            # 연속 촬영 방지를 위한 플래그 설정 (한 번 찍으면 각도가 벗어났다가 다시 들어와야 찍힘)
            captured = True
            
            # 시각적 피드백 (화면 깜빡임 효과)
            cv2.rectangle(img, (0, 0), (img_w, img_h), (255, 255, 255), 10)

        # 각도가 벗어나면 다시 촬영 가능 상태로 변경
        if not angle_ok:
            captured = False

        cv2.imshow("Auto Capture Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    # 기본 웹캠 사용 시: 0
    # 외부 카메라 사용 시: 1 또는 URL 등 (아래 설명 참조)
    run_auto_capture_camera(0)
