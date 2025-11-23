import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue
import math
from PIL import Image

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="AI 각도 따라잡기", layout="centered")

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "target_angle" not in st.session_state:
    st.session_state.target_angle = None
if "target_image" not in st.session_state:
    st.session_state.target_image = None

# Mediapipe 초기화 (Face Mesh 사용 - 정밀도 높음)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- 2. 헬퍼 함수: 각도 계산 ----------------
def calculate_roll_angle(landmarks, img_w, img_h):
    """
    왼쪽 눈(33)과 오른쪽 눈(263)의 좌표를 이용해 얼굴의 기울기(Roll)를 계산
    """
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    x1, y1 = left_eye.x * img_w, left_eye.y * img_h
    x2, y2 = right_eye.x * img_w, right_eye.y * img_h

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

# ---------------- 3. 영상 처리 클래스 ----------------
class AngleProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.target_angle = None  # 외부에서 주입
        self.frame_count = 0
        self.capture_triggered = False
        self.enter_time = None
        self.flash_frame = 0
        
        # 설정값
        self.angle_tolerance = 5.0  # 허용 오차 (도)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 거울 모드
        h, w, _ = img.shape
        
        # Mediapipe 처리
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        current_angle = 0.0
        similarity = 0.0
        is_matched = False
        status_msg = "No Face"
        bar_color = (0, 0, 255) # 빨강
        
        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 현재 각도 계산
            current_angle = calculate_roll_angle(landmarks, w, h)
            
            # 타겟 각도가 설정되어 있다면 비교
            if self.target_angle is not None:
                diff = abs(current_angle - self.target_angle)
                
                # 유사도 계산 (단순화: 45도 차이면 0점, 0도 차이면 100점)
                max_diff = 45.0
                similarity = max(0, 100 - (diff / max_diff * 100))
                
                status_text = f"Cur: {current_angle:.1f} / Target: {self.target_angle:.1f}"
                cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 매칭 판단
                if diff <= self.angle_tolerance:
                    is_matched = True
                    bar_color = (0, 255, 0) # 초록
                    status_msg = "HOLD!"
                else:
                    status_msg = "Tilt Head"
                    bar_color = (0, 255, 255) if similarity > 70 else (0, 0, 255)

                # 유사도 게이지 바 그리기
                bar_x, bar_y, bar_w, bar_h = 20, 80, 200, 20
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                fill_w = int(bar_w * (similarity / 100))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.putText(img, f"{int(similarity)}%", (bar_x + bar_w + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

            # 촬영 카운트다운 로직
            if is_matched:
                if self.enter_time is None:
                    self.enter_time = time.time()
                
                elapsed = time.time() - self.enter_time
                countdown = 1.5 - elapsed
                
                if countdown > 0:
                    cx, cy = w//2, h//2
                    cv2.putText(img, f"{countdown:.1f}", (cx-50, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
                else:
                    # 촬영!
                    if not self.capture_triggered:
                        save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB로 저장
                        self.result_queue.put(save_img)
                        self.capture_triggered = True
                        self.flash_frame = 5
            else:
                self.enter_time = None
                self.capture_triggered = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- 4. UI 구성 ----------------
st.title("📸 AI 각도 따라잡기")
st.markdown("따라하고 싶은 **'기준 사진'**을 올리면, 같은 각도가 되었을 때 자동으로 찍어줍니다!")

# 4-1. 기준 사진 업로드 섹션 (촬영 전)
if st.session_state.snapshot is None:
    with st.expander("1. 기준 사진 업로드 (Click to Open)", expanded=(st.session_state.target_angle is None)):
        uploaded_file = st.file_uploader("따라할 사진을 올려주세요 (정면/기울인 얼굴)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # 파일 읽기 및 분석
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Mediapipe 분석을 위해 RGB 변환
            if img_array.shape[2] == 4: # PNG alpha channel 처리
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 2: # Grayscale 처리
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
            results = face_mesh.process(img_array)
            
            if results.multi_face_landmarks:
                h, w, _ = img_array.shape
                landmarks = results.multi_face_landmarks[0].landmark
                angle = calculate_roll_angle(landmarks, w, h)
                
                st.session_state.target_angle = angle
                st.session_state.target_image = image
                st.success(f"✅ 기준 사진 분석 완료! 목표 각도: {angle:.1f}도")
                st.image(image, caption=f"기준 사진 (각도: {angle:.1f})", width=200)
            else:
                st.error("사진에서 얼굴을 찾을 수 없습니다. 다른 사진을 시도해주세요.")

# 4-2. 결과 화면 (촬영 후)
if st.session_state.snapshot is not None:
    st.markdown("---")
    st.success("📸 찰칵! 촬영에 성공했습니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.target_image, caption="기준 사진", use_container_width=True)
    with col2:
        st.image(st.session_state.snapshot, caption="내 사진", use_container_width=True)
        
    # 다운로드 버튼
    img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    
    if is_success:
        st.download_button(
            label="📥 결과 사진 저장하기",
            data=buffer.tobytes(),
            file_name=f"AI_Shot_{int(time.time())}.jpg",
            mime="image/jpeg",
            type="primary",
            use_container_width=True
        )
    
    # [수정됨] 다시 찍기 버튼 삭제 후 안내 문구 추가
    st.warning("🔄 다시 촬영하시려면 웹페이지를 새로고침 해주세요.")

# 4-3. 촬영 화면 (WebRTC)
elif st.session_state.target_angle is not None:
    st.markdown("---")
    st.header("2. 카메라를 보고 각도를 맞춰보세요!")
    
    ctx = webrtc_streamer(
        key="angle-shooter",
        video_processor_factory=AngleProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    )
    
    # 실시간으로 타겟 각도 정보를 프로세서에 전달
    if ctx.video_processor:
        ctx.video_processor.target_angle = st.session_state.target_angle
        
    # 결과 수신 대기
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result_img = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result_img is not None:
                        st.session_state.snapshot = result_img
                        st.rerun()
                except queue.Empty:
                    pass
            time.sleep(0.1)

else:
    st.info("👆 먼저 위에서 '기준 사진'을 업로드해주세요.")
