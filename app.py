import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ========= 기본 설정 =========
st.set_page_config(page_title="AI 자동 촬영기", layout="wide")

# STUN 서버 (배포 필수)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 세션 상태 초기화 (찍은 사진 저장용)
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

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

# ========= 영상 처리 클래스 (핵심 로직) =========
class FaceAngleProcessor(VideoProcessorBase):
    def __init__(self):
        self.ref_angle = None
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        # 사진 전송을 위한 우체통 (Queue)
        self.result_queue = queue.Queue()
        
        # 자동 촬영용 변수
        self.match_start_time = None
        self.capture_cooldown = 0

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
            status_text = f"Angle: {current_angle:.1f}"

            if self.ref_angle is not None:
                diff = abs(current_angle - self.ref_angle)
                status_text += f" | Diff: {diff:.1f}"
                
                # ★ 촬영 로직 ★
                # 1. 각도 차이가 5도 이내인지 확인
                if diff < 5.0:
                    color = (0, 255, 0) # 초록색
                    status_text = "HOLD ON!"
                    
                    # 타이머 시작
                    if self.match_start_time is None:
                        self.match_start_time = time.time()
                    
                    # 1초 동안 유지하면 촬영
                    if time.time() - self.match_start_time > 1.0:
                        # 쿨타임 체크 (연속 촬영 방지)
                        if time.time() - self.capture_cooldown > 3.0:
                            # ★ 사진 찍어서 우체통에 넣기 ★
                            # (OpenCV 이미지는 BGR이므로 RGB로 변환해서 보냄)
                            captured_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.result_queue.put(captured_img)
                            
                            self.capture_cooldown = time.time()
                            status_text = "CAPTURED!"
                else:
                    # 조건 안 맞으면 타이머 리셋
                    self.match_start_time = None
            
            # 시각화
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= 메인 UI =========
def main():
    st.title("📸 AI 자동 촬영기")
    st.info("왼쪽에서 사진을 올리고, 오른쪽에서 카메라를 켜세요. 각도가 맞으면 1초 뒤 찍힙니다!")

    col1, col2 = st.columns([1, 1])

    # [왼쪽] 기준 사진 업로드
    with col1:
        st.subheader("1. 기준 사진")
        uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
        ref_angle_val = None
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                res = detector.process(ref_img_rgb)
                if res.detections:
                    h, w, _ = ref_img.shape
                    ref_angle_val = calc_roll_angle(res.detections[0], w, h)
                    st.success(f"기준 각도: {ref_angle_val:.1f}°")
                    st.image(ref_img_rgb, use_container_width=True)
                else:
                    st.error("얼굴을 찾을 수 없습니다.")

    # [오른쪽] 실시간 카메라
    with col2:
        st.subheader("2. 실시간 촬영")
        
        # WebRTC 실행
        ctx = webrtc_streamer(
            key="auto-capture",
            video_processor_factory=FaceAngleProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            async_processing=True
        )

        # 기준 각도 전달
        if ctx.video_processor:
            ctx.video_processor.ref_angle = ref_angle_val

        # ★ 핵심: 우체통(Queue) 확인하여 사진 가져오기 ★
        if ctx.state.playing:
            if ctx.video_processor:
                try:
                    # 큐에서 사진이 왔나 확인 (블로킹 없이)
                    result_image = ctx.video_processor.result_queue.get(timeout=0.1)
                    
                    # 사진이 도착했다면 세션에 저장하고 앱 새로고침
                    if result_image is not None:
                        st.session_state.snapshot = result_image
                        st.rerun()
                except queue.Empty:
                    pass

    # [하단] 결과물 표시 및 다운로드
    st.markdown("---")
    if st.session_state.snapshot is not None:
        st.success("🎉 촬영 성공! 아래 버튼으로 저장하세요.")
        
        # 보기 좋게 표시
        st.image(st.session_state.snapshot, caption="방금 찍은 인생샷", width=400)
        
        # 다운로드 버튼 생성
        # 이미지를 바이트로 변환
        img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        
        if ret:
            st.download_button(
                label="📥 내 폰에 저장하기",
                data=buffer.tobytes(),
                file_name="AI_Capture.jpg",
                mime="image/jpeg",
                type="primary"
            )
            
        if st.button("다시 찍기"):
            st.session_state.snapshot = None
            st.rerun()

if __name__ == "__main__":
    main()