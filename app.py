import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="AI 자동 촬영기", layout="centered")

# [수정] winsound 제거 (서버 에러 원인)
# 서버에서는 st.audio로 소리를 재생해야 합니다.

# STUN 서버 (외부 접속용)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
    
# 우체통(Queue) 초기화
if "img_queue" not in st.session_state:
    st.session_state.img_queue = queue.Queue()

st.title("📸 AI 자동 촬영기")
st.info("각도가 맞으면 'CAPTURED' 메시지가 뜨고 사진이 저장됩니다.")

# ---------------- 2. 사이드바 설정 ----------------
st.sidebar.header("⚙️ 설정")
min_val = st.sidebar.slider("최소 각도", 0.0, 0.3, 0.02, 0.01)
max_val = st.sidebar.slider("최대 각도", 0.0, 0.3, 0.15, 0.01)

# ---------------- 3. 영상 처리 클래스 ----------------
class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self, img_queue):
        self.img_queue = img_queue
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.match_start_time = None
        self.last_capture_time = 0
        self.flash_frame = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        status_text = "Adjust Angle"
        color = (0, 0, 255) # 빨강

        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)
            status_text = "CAPTURED!"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Z-Diff 계산
            chin = landmarks[152].z
            forehead = landmarks[10].z
            current_z = (chin - forehead) * -1 
            
            # 범위 체크
            if 0.02 <= current_z <= 0.20:
                color = (0, 255, 0) # 초록
                status_text = "HOLD ON!"
                
                if self.match_start_time is None:
                    self.match_start_time = time.time()
                
                # 1초 유지 시 촬영
                if time.time() - self.match_start_time > 1.0:
                    if time.time() - self.last_capture_time > 3.0:
                        # ★ 촬영 및 전송 ★
                        send_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.img_queue.put(send_img)
                        
                        self.last_capture_time = time.time()
                        self.flash_frame = 5
                        # [수정] winsound.Beep 삭제 (서버에서 소리 못 냄)
            else:
                self.match_start_time = None
                
            # 시각화
            cv2.rectangle(img, (0,0), (w,h), color, 15)
            cv2.putText(img, f"Z: {current_z:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(img, status_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- 4. 메인 로직 ----------------

# 사진이 찍혔으면 결과 화면 보여주기
if st.session_state.snapshot is not None:
    st.success("🎉 촬영 성공!")
    
    # [추가] 브라우저에서 소리 재생 (이건 서버에서도 됨)
    # 찰칵 소리 파일이 없으므로 풍선 효과로 대체
    st.balloons()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.snapshot, caption="인생샷", use_container_width=True)
    with col2:
        img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        if ret:
            st.download_button(
                label="📥 갤러리에 저장",
                data=buffer.tobytes(),
                file_name=f"Selfie_{int(time.time())}.jpg",
                mime="image/jpeg",
                type="primary"
            )
    
    if st.button("🔄 다시 찍기", type="secondary"):
        st.session_state.snapshot = None
        st.rerun()

# 사진이 없으면 카메라 보여주기
else:
    def processor_factory():
        return FaceMeshProcessor(st.session_state.img_queue)

    ctx = webrtc_streamer(
        key="mobile-capture",
        video_processor_factory=processor_factory,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True
    )

    if ctx.state.playing:
        placeholder = st.empty()
        placeholder.write("📸 카메라 작동 중... (각도를 맞춰보세요)")
        
        while True:
            if ctx.video_processor:
                try:
                    if not st.session_state.img_queue.empty():
                        result_img = st.session_state.img_queue.get()
                        st.session_state.snapshot = result_img
                        st.rerun()
                except Exception as e:
                    print(e)
            time.sleep(0.1)
