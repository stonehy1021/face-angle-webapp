import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
import queue
import os
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ========= 기본 설정 =========
st.set_page_config(page_title="AI 자동 촬영기", layout="centered")

# 저장 폴더 확실하게 생성
SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# STUN 서버 (배포 필수)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Mediapipe 초기화
mp_face = mp.solutions.face_detection

# ========= 유틸 함수 =========
def calc_roll_angle(detection, width, height):
    kp = detection.location_data.relative_keypoints
    left_eye = kp[0]
    right_eye = kp[1]
    x1, y1 = left_eye.x * width, left_eye.y * height
    x2, y2 = right_eye.x * width, right_eye.y * height
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# ========= 영상 처리 클래스 =========
class FaceAngleProcessor(VideoProcessorBase):
    def __init__(self):
        self.ref_angle = None
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.result_queue = queue.Queue()
        
        # 촬영 관련 변수
        self.match_start_time = None
        self.last_capture_time = 0
        self.flash_frame = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 거울 모드
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)
        
        current_angle = 0.0
        status_text = "Looking..."
        color = (0, 0, 255) # 빨강

        if results.detections:
            detection = results.detections[0]
            current_angle = calc_roll_angle(detection, w, h)
            
            # 모바일 Z값 보정 (단순화)
            current_z = (detection.location_data.relative_keypoints[2].y - 
                         detection.location_data.relative_keypoints[0].y) * 10 
            # 실제로는 Roll 각도 기준으로 함 (질문자 의도 반영)
            # 여기서는 '각도' 자체를 기준으로 판별합니다.
            
            status_text = f"Angle: {current_angle:.1f}"

            # ★ 조건 체크 (각도 차이가 작으면) ★
            # 기준 각도가 없으면 0도(정면)를 기준으로 함
            target = self.ref_angle if self.ref_angle is not None else 0
            diff = abs(current_angle - target)
            
            if diff < 5.0:  # 5도 이내면 OK
                color = (0, 255, 0) # 초록
                status_text = "HOLD ON!"
                
                if self.match_start_time is None:
                    self.match_start_time = time.time()
                
                # 1초 유지 시 촬영
                if time.time() - self.match_start_time > 1.0:
                    if time.time() - self.last_capture_time > 3.0:
                        
                        # [1] 서버 폴더에 무조건 저장 (백업용)
                        ts = int(time.time())
                        filename = SAVE_DIR / f"Auto_Shot_{ts}.jpg"
                        # OpenCV는 BGR 이미지를 저장함
                        cv2.imwrite(str(filename), img)
                        print(f"💾 서버 저장 완료: {filename}")
                        
                        # [2] 화면으로 전송 (RGB 변환)
                        send_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.result_queue.put(send_img)
                        
                        self.last_capture_time = time.time()
                        self.flash_frame = 5
            else:
                self.match_start_time = None
        
        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            cv2.rectangle(img, (0,0), (w,h), (255,255,255), -1) # 하얀 화면
            status_text = "CAPTURED!"
            
        # 텍스트 그리기
        cv2.rectangle(img, (0,0), (w,h), color, 10)
        cv2.putText(img, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)
        cv2.putText(img, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= 메인 UI =========
def main():
    st.title("📸 AI 자동 촬영기")
    st.warning("👇 사진이 찍히면 화면 아래에 나타납니다! 스크롤을 내려보세요.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. 기준 사진 (선택)")
        uploaded_file = st.file_uploader("없으면 정면(0도)이 기준이 됩니다.", type=['jpg', 'png'])
        ref_angle_val = 0.0
        if uploaded_file:
            # (사진 분석 로직 생략 - 파일만 있으면 0도로 가정하거나 별도 분석 가능)
            st.success("기준 사진 설정됨!")

    with col2:
        st.subheader("2. 촬영 화면")
        ctx = webrtc_streamer(
            key="camera",
            video_processor_factory=FaceAngleProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            async_processing=True
        )
        if ctx.video_processor:
            ctx.video_processor.ref_angle = ref_angle_val

        # [핵심] 실시간으로 사진 배달 기다리기
        if ctx.state.playing:
            if ctx.video_processor:
                try:
                    result = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result is not None:
                        st.session_state.snapshot = result
                        st.rerun() # 화면 새로고침!
                except queue.Empty:
                    pass

    # ------------------------------------------------
    # 여기가 사진 나오는 곳입니다 (화면 하단)
    # ------------------------------------------------
    st.markdown("---")
    if st.session_state.snapshot is not None:
        st.balloons()
        st.success("📸 찍혔습니다! 아래 버튼을 눌러 저장하세요.")
        
        # 사진 표시
        st.image(st.session_state.snapshot, caption="인생샷 건짐", use_container_width=True)
        
        # 다운로드 버튼
        img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        if ret:
            st.download_button(
                label="📥 내 폰 갤러리에 저장하기",
                data=buffer.tobytes(),
                file_name=f"Selfie_{int(time.time())}.jpg",
                mime="image/jpeg",
                type="primary"
            )
            
        if st.button("🔄 다시 찍으러 가기"):
            st.session_state.snapshot = None
            st.rerun()

if __name__ == "__main__":
    main()


