import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ========= 1. 기본 설정 =========
st.set_page_config(page_title="AI 자동 촬영기", layout="wide")

# STUN 서버 (외부 접속 필수 설정)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 세션 상태 초기화 (찍은 사진 저장용)
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# 우체통(Queue) 초기화 (영상 처리기 -> 메인 화면 데이터 전송용)
if "img_queue" not in st.session_state:
    st.session_state.img_queue = queue.Queue()

# Mediapipe 설정
mp_face = mp.solutions.face_detection

# ========= 2. 유틸 함수 (각도 계산) =========
def calc_roll_angle_from_detection(detection, width, height):
    """눈 위치로 Roll 각도 계산 (주신 코드 로직 유지)"""
    keypoints = detection.location_data.relative_keypoints
    left_eye = keypoints[0]
    right_eye = keypoints[1]

    x1, y1 = left_eye.x * width, left_eye.y * height
    x2, y2 = right_eye.x * width, right_eye.y * height

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def analyze_reference_image(uploaded_file):
    """기준 사진 분석"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        res = detector.process(rgb)
        if res.detections:
            return calc_roll_angle_from_detection(res.detections[0], w, h)
    return None

# ========= 3. 영상 처리 클래스 (핵심) =========
class FaceAngleProcessor(VideoProcessorBase):
    def __init__(self):
        self.ref_angle = None  # 기준 각도 (외부에서 주입)
        self.img_queue = None  # 데이터 전송 통로
        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        
        # 자동 촬영 변수
        self.last_capture_time = 0
        self.flash_frame = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 거울 모드
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(img_rgb)
        
        current_angle = 0.0
        status_text = "Detecting..."
        color = (0, 0, 255) # 빨강

        # 플래시 효과 (촬영 직후)
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)
            status_text = "CAPTURED!"

        if res.detections:
            detection = res.detections[0]
            current_angle = calc_roll_angle_from_detection(detection, w, h)
            
            status_text = f"Cur: {current_angle:.1f}"
            
            # 기준 각도가 있으면 비교 시작
            if self.ref_angle is not None:
                diff = abs(current_angle - self.ref_angle)
                status_text += f" | Diff: {diff:.1f}"
                
                # ★ 오차 5도 이내면 촬영 ★
                if diff < 5.0:
                    color = (0, 255, 0) # 초록
                    
                    # 3초 쿨타임 체크
                    if time.time() - self.last_capture_time > 3.0:
                        # 사진 찍어서 큐에 넣기
                        if self.img_queue is not None:
                            # OpenCV(BGR) -> RGB 변환해서 전송
                            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.img_queue.put(save_img)
                            
                            self.last_capture_time = time.time()
                            self.flash_frame = 5
                            print("📸 자동 촬영됨!")
            
            # 그리기
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= 4. 메인 UI =========
def main():
    st.title("📸 AI 자동 촬영기 (WebRTC)")
    
    col1, col2 = st.columns([1, 1])

    # [왼쪽] 기준 사진 설정
    with col1:
        st.subheader("1️⃣ 기준 사진")
        uploaded_file = st.file_uploader("기준 사진 업로드", type=['jpg', 'png'])
        ref_angle_val = None
        
        if uploaded_file:
            angle = analyze_reference_image(uploaded_file)
            if angle is not None:
                ref_angle_val = angle
                st.success(f"기준 각도: {angle:.1f}°")
            else:
                st.error("얼굴 감지 실패")

    # [오른쪽] 카메라 실행
    with col2:
        st.subheader("2️⃣ 실시간 촬영")
        
        # Processor Factory 생성 (큐 주입)
        def processor_factory():
            proc = FaceAngleProcessor()
            proc.ref_angle = ref_angle_val  # 기준 각도 전달
            proc.img_queue = st.session_state.img_queue # 우체통 전달
            return proc

        ctx = webrtc_streamer(
            key="auto-capture",
            video_processor_factory=processor_factory,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            async_processing=True
        )

        # ★ 핵심: 실시간으로 우체통 확인하기 ★
        if ctx.state.playing:
            if not st.session_state.img_queue.empty():
                try:
                    # 사진 꺼내기
                    result_img = st.session_state.img_queue.get_nowait()
                    st.session_state.snapshot = result_img
                    st.rerun() # 화면 새로고침
                except queue.Empty:
                    pass

    # [하단] 결과물 & 다운로드
    st.markdown("---")
    if st.session_state.snapshot is not None:
        st.success("🎉 촬영 성공!")
        st.image(st.session_state.snapshot, caption="방금 찍은 사진", width=400)
        
        # 다운로드 버튼
        img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        if ret:
            st.download_button(
                label="📥 사진 다운로드",
                data=buffer.tobytes(),
                file_name=f"Auto_Shot_{int(time.time())}.jpg",
                mime="image/jpeg",
                type="primary"
            )
            
        if st.button("🔄 다시 찍기"):
            st.session_state.snapshot = None
            st.rerun()

if __name__ == "__main__":
    main()
