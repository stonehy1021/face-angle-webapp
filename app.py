import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================
# 0. 기본 세팅
# =========================
st.set_page_config(page_title="AI 타겟 구도 자동 촬영기", layout="wide")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_face = mp.solutions.face_detection

# 세션 기본값 초기화
for key, default in [
    ("snapshot", None),   # 찍힌 최종 사진
    ("ref_angle", None),  # 기준 사진에서 나온 각도
    ("angle_tol", 12.0),  # 허용 오차 (기본 12도 정도로 널널하게)
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "img_queue" not in st.session_state:
    st.session_state.img_queue = queue.Queue()


# =========================
# 1. 유틸 함수
# =========================
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

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


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

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        res = detector.process(rgb)
        if res.detections:
            angle = calc_roll_angle_from_detection(res.detections[0], w, h)
            return angle

    return None


# =========================
# 2. WebRTC 영상 처리
# =========================
class FaceAngleProcessor(VideoProcessorBase):
    """
    - 각 프레임에서 얼굴 각도 계산
    - 최근 여러 프레임의 '평균 각도'를 구해서 안정화
    - 평균 각도가 ref_angle과 tolerance 이내면 자동 촬영
    - 찍힌 사진은 img_queue로 메인 스레드에 전달
    """
    def __init__(self):
        self.ref_angle = None          # 기준 각도
        self.tolerance = 12.0          # 허용 오차
        self.img_queue = None          # 메인 스레드로 보낼 큐

        self.detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )
        self.last_capture_time = 0
        self.flash_frame = 0

        # 각도 안정화를 위한 히스토리
        self.angle_history = []
        self.max_history = 10  # 최근 10프레임까지만 사용

    def _update_angle_history(self, angle):
        self.angle_history.append(angle)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)

    def _get_smoothed_angle(self):
        """
        최근 angle_history를 이용해 '평균 각도'를 리턴
        (노이즈 줄이기용)
        """
        if not self.angle_history:
            return None
        return float(sum(self.angle_history) / len(self.angle_history))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 거울 모드
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(img_rgb)

        status_text = "Detecting..."
        color = (0, 0, 255)  # 기본 빨강

        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)
            status_text = "CAPTURED!"

        if res.detections:
            detection = res.detections[0]
            current_angle = calc_roll_angle_from_detection(detection, w, h)

            # 히스토리에 추가하고, 평균 각도 계산
            self._update_angle_history(current_angle)
            smoothed_angle = self._get_smoothed_angle()

            if smoothed_angle is not None:
                status_text = f"Cur: {current_angle:.1f}° / Avg: {smoothed_angle:.1f}°"
            else:
                status_text = f"Cur: {current_angle:.1f}°"

            # 기준 각도가 있을 때만 자동 촬영 로직
            if (self.ref_angle is not None) and (smoothed_angle is not None):
                diff = abs(smoothed_angle - self.ref_angle)
                status_text += f" | Diff: {diff:.1f}° (Tol: {self.tolerance:.0f}°)"

                # '평균 각도'가 기준 각도와 충분히 가까워졌을 때 촬영
                if diff < self.tolerance:
                    color = (0, 255, 0)
                    if time.time() - self.last_capture_time > 3.0:
                        if self.img_queue is not None:
                            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.img_queue.put(save_img)
                            self.last_capture_time = time.time()
                            self.flash_frame = 5
                            print("📸 자동 촬영됨!")

            # 얼굴 박스 + 텍스트
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                img,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# 3. 메인 UI
# =========================
def main():
    st.title("📸 타겟 구도 맞추는 AI 자동 촬영기")

    # 이미 한 번 찍혔으면 → 저장 화면
    if st.session_state.get("snapshot") is not None:
        st.success("타겟 구도에 맞게 촬영 완료!")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                st.session_state.snapshot,
                caption="방금 찍은 사진",
                use_container_width=True,
            )

        with col2:
            img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode(".jpg", img_bgr)
            if ret:
                st.download_button(
                    label="📥 사진 저장하기",
                    data=buffer.tobytes(),
                    file_name=f"Auto_Shot_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    type="primary",
                )

        if st.button("🔄 다시 찍기"):
            st.session_state.snapshot = None
            st.rerun()
        return

    col1, col2 = st.columns([1, 1])

    # -------- 왼쪽: 기준 사진 업로드 --------
    with col1:
        st.subheader("1️⃣ 타겟(기준) 사진 업로드")

        uploaded_file = st.file_uploader(
            "기준 사진 업로드 (얼굴이 정면/측면이든 한 번에 보이게)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            angle = analyze_reference_image(uploaded_file)
            if angle is not None:
                st.session_state.ref_angle = angle
                st.success(f"기준 각도: {angle:.1f}° 로 설정되었습니다.")
            else:
                st.error("얼굴 감지 실패. 다른 사진으로 다시 시도해 주세요.")

        ref_angle = st.session_state.get("ref_angle", None)
        if ref_angle is not None:
            st.info(f"현재 기준 각도: {ref_angle:.1f}°")
        else:
            st.warning("기준 사진을 업로드하면 각도를 분석합니다.")

        # 허용 오차 슬라이더 (기본 12도, 노트북이면 15~20도까지도 추천)
        st.session_state.angle_tol = st.slider(
            "허용 각도 오차(도)",
            min_value=5.0,
            max_value=25.0,
            value=float(st.session_state.get("angle_tol", 12.0)),
            step=1.0,
            help="얼마나 비슷해야 자동 촬영할지 정하는 값입니다.",
        )

    # -------- 오른쪽: WebRTC 카메라 --------
    with col2:
        st.subheader("2️⃣ 실시간 촬영")

        queue_ref = st.session_state.img_queue

        def processor_factory():
            proc = FaceAngleProcessor()
            proc.ref_angle = st.session_state.get("ref_angle", None)
            proc.tolerance = float(st.session_state.get("angle_tol", 12.0))
            proc.img_queue = queue_ref
            return proc

        ctx = webrtc_streamer(
            key="auto-capture",
            video_processor_factory=processor_factory,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "facingMode": "user",
                },
                "audio": False,
            },
            async_processing=True,
        )

        # 디버그용: 강제 캡쳐 버튼 (파이프라인 확인용)
        if ctx.state.playing:
            if st.button("💥 강제 캡쳐 (디버그용)"):
                # 강제로 한 프레임을 캡쳐하는 건 어렵지만,
                # 이미 Processor에서 queue로 넣어준 게 있으면 우선 가져옴
                if not st.session_state.img_queue.empty():
                    try:
                        result_img = st.session_state.img_queue.get_nowait()
                        st.session_state.snapshot = result_img
                        st.rerun()
                    except queue.Empty:
                        pass

        # 자동 촬영된 사진 수신
        if ctx.state.playing:
            if not st.session_state.img_queue.empty():
                try:
                    result_img = st.session_state.img_queue.get_nowait()
                    st.session_state.snapshot = result_img
                    st.rerun()
                except queue.Empty:
                    pass


if __name__ == "__main__":
    main()
