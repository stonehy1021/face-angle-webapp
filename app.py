import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ========= 기본 설정 =========
st.set_page_config(page_title="얼굴 각도 분석", layout="wide")

# STUN 서버 설정 (배포 시 필수)
# 구글의 무료 STUN 서버를 사용하여 외부 접속 허용
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Mediapipe 초기화
mp_face = mp.solutions.face_detection

# ========= 유틸 함수 =========
def calc_roll_angle(detection, width, height):
    """얼굴의 기울기(Roll) 계산"""
    kp = detection.location_data.relative_keypoints
    left_eye = kp[0]  # 왼쪽 눈
    right_eye = kp[1] # 오른쪽 눈

    x1, y1 = left_eye.x * width, left_eye.y * height
    x2, y2 = right_eye.x * width, right_eye.y * height

    # 각도 계산 (dy, dx)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# ========= 기준 사진 분석 함수 =========
def analyze_static_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        res = detector.process(img_rgb)
        
        if res.detections:
            angle = calc_roll_angle(res.detections[0], w, h)
            return angle, img
    
    return None, img

# ========= 영상 처리 클래스 (WebRTC 핵심) =========
class FaceAngleProcessor(VideoProcessorBase):
    def __init__(self):
        self.ref_angle = None # 기준 각도
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 이미지 가져오기
        img = frame.to_ndarray(format="bgr24")
        
        # 2. 좌우 반전 (거울 모드)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 3. Mediapipe 분석을 위해 RGB 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)
        
        current_angle = 0.0
        diff = 0.0
        status_text = "No Face"
        color = (0, 0, 255) # 빨강

        if results.detections:
            detection = results.detections[0]
            current_angle = calc_roll_angle(detection, w, h)
            
            status_text = f"Angle: {current_angle:.1f}"
            color = (255, 0, 0) # 파랑

            # 기준 각도가 설정되어 있다면 차이 계산
            if self.ref_angle is not None:
                diff = abs(current_angle - self.ref_angle)
                status_text += f" | Diff: {diff:.1f}"
                
                # 차이가 5도 이내면 초록색
                if diff < 5.0:
                    color = (0, 255, 0)
                    status_text += " (MATCH!)"

            # 시각화 (박스 및 텍스트)
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        else:
            cv2.putText(img, "Face Not Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 4. 처리된 이미지를 다시 송출
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= 메인 UI =========
def main():
    st.title("📸 AI 얼굴 각도 분석기")
    st.info("왼쪽에서 기준 사진을 올리고, 아래에서 카메라를 켜세요.")

    col1, col2 = st.columns([1, 2])

    # [왼쪽] 기준 사진 업로드
    with col1:
        st.subheader("1. 기준 사진")
        uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
        
        ref_angle_val = None
        
        if uploaded_file:
            angle, processed_img = analyze_static_image(uploaded_file)
            if angle is not None:
                ref_angle_val = angle
                st.success(f"기준 각도: {angle:.1f}°")
                # OpenCV 이미지를 RGB로 변환해 표시
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("얼굴을 찾을 수 없습니다.")

    # [오른쪽] 실시간 카메라
    with col2:
        st.subheader("2. 실시간 분석")
        
        # WebRTC 스트리머
        ctx = webrtc_streamer(
            key="angle-analysis",
            video_processor_factory=FaceAngleProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"facingMode": "user"}, # 전면 카메라
                "audio": False
            },
            async_processing=True
        )

        # 프로세서에 기준 각도 전달
        if ctx.video_processor:
            ctx.video_processor.ref_angle = ref_angle_val

if __name__ == "__main__":
    main()
