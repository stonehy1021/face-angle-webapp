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

# 세션 상태 초기화 (찍힌 사진 저장용)
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# 우체통(Queue) 초기화 (영상 처리 스레드 → 메인 스레드로 사진 전달)
if "img_queue" not in st.session_state:
    st.session_state.img_queue = queue.Queue()

# Mediapipe 얼굴 검출
mp_face = mp.solutions.face_detection


# ========= 2. 유틸 함수 =========
def calc_roll_angle_from_detection(detection, width, height):
    """
    Mediapipe FaceDetection 결과에서 왼/오른쪽 눈 위치를 이용해
    얼굴 roll(기울기) 각도를 구하는 함수
    """
    keypoints = detection.location_data.relative_keypoints
    left_eye = keypoints[0]
    right_eye = keypoints[1]
    x1, y1 = left_eye.x * width, left_eye.y * height
    x2, y2 = right_eye.x * width, right_eye.y * height
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))


def analyze_reference_image(uploaded_file):
    """
    업로드된 기준(타겟) 사진에서 얼굴 각도(roll)를 분석해서 기준 각도 리턴
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with mp_face.FaceDetection(model_selection=1,
                               min_detection_confidence=0.5) as detector:
        res = detector.process(rgb)
        if res.detections:
            return calc_roll_angle_from_detection(res.detections[0], w, h)

    return None


# ========= 3. 영상 처리 클래스 =========
class FaceAngleProcessor(VideoProcessorBase):
    """
    WebRTC 영상 프레임을 받아서
    - 얼굴 각도 계산
    - 기준 각도(ref_angle)와의 차이가 작으면 자동 촬영
    - 찍힌 사진은 img_queue에 넣어서 메인 스레드로 전달
    """
    def __init__(self):
        self.ref_angle = None      # 기준(타겟) 사진 각도
        self.img_queue = None      # 메인 스레드로 보낼 큐
        self.detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        self.last_capture_time = 0
        self.flash_frame = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 거울 모드
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(img_rgb)

        current_angle = 0.0
        status_text = "Detecting..."
        color = (0, 0, 255)  # 기본 빨강

        # 플래시 효과 (사진 찍힌 후 하얀 번쩍)
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)
            status_text = "CAPTURED!"

        if res.detections:
            detection = res.detections[0]
            current_angle = calc_roll_angle_from_detection(detection, w, h)
            status_text = f"Cur: {current_angle:.1f}°"

            if self.ref_angle is not None:
                diff = abs(current_angle - self.ref_angle)
                status_text += f" | Diff: {diff:.1f}°"

                # ★ 기준 각도와의 차이가 5도 이내면 자동 촬영
                if diff < 5.0:
                    color = (0, 255, 0)  # 초록색 박스
                    # 최소 3초 간격으로만 촬영
                    if time.time() - self.last_capture_time > 3.0:
                        if self.img_queue is not None:
                            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.img_queue.put(save_img)  # 메인으로 전달
                            self.last_capture_time = time.time()
                            self.flash_frame = 5
                            print("📸 자동 촬영됨!")

            # 얼굴 박스 + 텍스트 그리기
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(img, status_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ========= 4. 메인 UI =========
def main():
    st.title("📸 타겟 구도 맞추는 AI 자동 촬영기")

    # 이미 한 번 찍혔으면 → 사진 + 저장/다시찍기 UI만 보여주기 (z_shooter1 스타일)
    if st.session_state.snapshot is not None:
        st.success("🎉 타겟 구도에 맞게 촬영 완료!")
        col1, col2 = st.columns(2)

        with col1:
            st.image(st.session_state.snapshot,
                     caption="방금 찍은 사진",
                     use_container_width=True)

        with col2:
            img_bgr = cv2.cvtColor(st.session_state.snapshot,
                                   cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode(".jpg", img_bgr)
            if ret:
                st.download_button(
                    label="📥 사진 저장하기",
                    data=buffer.tobytes(),
                    file_name=f"Auto_Shot_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    type="primary"
                )

        if st.button("🔄 다시 찍기"):
            st.session_state.snapshot = None
            st.rerun()
        return  # 밑에 카메라 영역은 안 보여줌

    # 아직 사진이 없으면 → 기준 사진 + 카메라 UI 보여주기
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1️⃣ 타겟(기준) 사진 업로드")
        uploaded_file = st.file_uploader("기준 사진 업로드", type=['jpg', 'png'])

        ref_angle_val = None
        if uploaded_file is not None:
            angle = analyze_reference_image(uploaded_file)
            if angle is not None:
                ref_angle_val = angle
                st.success(f"기준 각도: {angle:.1f}°")
                st.caption("실시간 카메라에서 이 각도에 맞으면 자동으로 촬영됩니다.")
            else:
                st.error("얼굴 감지 실패. 다른 사진으로 다시 시도해 주세요.")

    with col2:
        st.subheader("2️⃣ 실시간 촬영")

        # 메인 스레드에서 큐를 꺼내서 전달 (작업자 스레드는 st.session_state 직접 접근 X)
        queue_ref = st.session_state.img_queue

        def processor_factory():
            proc = FaceAngleProcessor()
            proc.ref_angle = ref_angle_val  # ★ 타겟 사진에서 구한 기준 각도 전달
            proc.img_queue = queue_ref      # ★ 촬영된 사진을 보낼 큐 연결
            return proc

        ctx = webrtc_streamer(
            key="auto-capture",
            video_processor_factory=processor_factory,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"facingMode": "user"},
                "audio": False
            },
            async_processing=True
        )

        # WebRTC가 재생 중일 때, 우체통에 사진이 들어왔는지 확인
        if ctx.state.playing:
            if not st.session_state.img_queue.empty():
                try:
                    result_img = st.session_state.img_queue.get_nowait()
                    st.session_state.snapshot = result_img
                    st.rerun()  # 사진 찍히면 화면을 "저장/다시찍기" 모드로 전환
                except queue.Empty:
                    pass


if __name__ == "__main__":
    main()
