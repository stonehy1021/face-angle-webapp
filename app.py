import time
import math
from io import StringIO

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoTransformerBase,
)

# ========= Mediapipe 설정 =========
mp_face = mp.solutions.face_detection

# WebRTC STUN 서버 설정 (Cloud 환경에서 필수)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)


# ========= 유틸 함수들 =========
def calc_roll_angle_from_detection(detection, width, height):
    """
    Mediapipe FaceDetection 결과에서 얼굴 기울기(roll angle)를 계산.
    두 눈 위치를 이용해서 각도 구함.
    """
    keypoints = detection.location_data.relative_keypoints

    # LEFT_EYE = 0, RIGHT_EYE = 1
    left_eye = keypoints[0]
    right_eye = keypoints[1]

    x1, y1 = left_eye.x * width, left_eye.y * height
    x2, y2 = right_eye.x * width, right_eye.y * height

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def analyze_reference_image(file):
    """
    업로드 기준 사진에서 얼굴을 찾고 각도를 계산해서 반환.
    실패 시 None.
    """
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("기준 이미지를 불러올 수 없습니다.")
        return None

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        res = detector.process(rgb)

    if not res.detections:
        st.error("기준 이미지에서 얼굴을 찾지 못했습니다.")
        return None

    detection = res.detections[0]
    angle = calc_roll_angle_from_detection(detection, w, h)
    return angle


def encode_image_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    """BGR 이미지를 PNG 바이트로 인코딩."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("이미지 인코딩 실패")
    return buf.tobytes()


# ========= WebRTC용 VideoTransformer =========
class FaceAngleTransformer(VideoTransformerBase):
    def __init__(self):
        # Mediapipe detector
        self.detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )

        # 상태 값들
        self.ref_angle = None      # 기준 사진 각도
        self.last_angle = None     # 최근 프레임 각도
        self.last_diff = None      # 기준과의 차이
        self.last_frame = None     # 최근 프레임 (BGR)

        # 로그 기록 (시간, 각도, 차이)
        self.log = []

    def set_reference_angle(self, angle):
        self.ref_angle = angle

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()

        img_h, img_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = self.detector.process(img_rgb)

        angle = None
        diff = None

        if res.detections:
            detection = res.detections[0]
            angle = calc_roll_angle_from_detection(detection, img_w, img_h)
            self.last_angle = angle

            # 기준 각도와 차이
            if self.ref_angle is not None:
                diff = angle - self.ref_angle
                self.last_diff = diff

            # 화면에 그리기 (박스 + 텍스트)
            relative_bbox = detection.location_data.relative_bounding_box
            x1 = int(relative_bbox.xmin * img_w)
            y1 = int(relative_bbox.ymin * img_h)
            w = int(relative_bbox.width * img_w)
            h = int(relative_bbox.height * img_h)

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            text = f"angle: {angle:.1f} deg"
            if diff is not None:
                text += f" | diff: {diff:+.1f} deg"

            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # 로그 저장
            self.log.append(
                {
                    "time": time.time(),
                    "angle": float(angle),
                    "diff": float(diff) if diff is not None else None,
                }
            )

        else:
            self.last_angle = None
            self.last_diff = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="얼굴 각도 분석 데모", layout="wide")
    st.title("📷 얼굴 각도 분석 · 기준 사진과의 유사도 체크")

    st.write(
        """
        - 왼쪽에서 **기준 사진**을 업로드해서 기준 얼굴 각도를 설정합니다.  
        - 오른쪽에 카메라를 켜면 실시간으로 각도와 기준 대비 차이를 보여줍니다.  
        - 아래에서 **스냅샷 저장 / CSV 다운로드**도 할 수 있습니다.
        """
    )

    # 세션 상태 초기화
    if "snapshot_counter" not in st.session_state:
        st.session_state["snapshot_counter"] = 0
    if "last_snapshot_png" not in st.session_state:
        st.session_state["last_snapshot_png"] = None

    col_left, col_right = st.columns(2)

    # ---- 1️⃣ 기준 사진 업로드 & 분석 ----
    with col_left:
        st.subheader("1️⃣ 기준 사진 설정")

        uploaded_file = st.file_uploader(
            "얼굴이 잘 나온 기준 사진을 업로드하세요 (jpg, png 등)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None and st.button("기준 사진 각도 분석하기"):
            angle = analyze_reference_image(uploaded_file)
            if angle is not None:
                st.session_state["ref_angle_value"] = angle
                st.success(f"기준 얼굴 각도: {angle:.2f}°")

        if "ref_angle_value" in st.session_state:
            st.info(f"현재 저장된 기준 각도: {st.session_state['ref_angle_value']:.2f}°")

    # ---- 2️⃣ 카메라 WebRTC ----
    with col_right:
        st.subheader("2️⃣ 카메라로 실시간 분석")

        webrtc_ctx = webrtc_streamer(
            key="face-angle-demo",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_transformer_factory=FaceAngleTransformer,
            async_processing=True,
        )

        angle_placeholder = st.empty()
        diff_placeholder = st.empty()

        if webrtc_ctx and webrtc_ctx.video_transformer:
            transformer: FaceAngleTransformer = webrtc_ctx.video_transformer  # type: ignore

            # 기준 각도 주입
            if "ref_angle_value" in st.session_state:
                transformer.set_reference_angle(st.session_state["ref_angle_value"])

            if webrtc_ctx.state.playing:
                current_angle = transformer.last_angle
                current_diff = transformer.last_diff

                if current_angle is not None:
                    angle_placeholder.metric("현재 얼굴 각도", f"{current_angle:.2f}°")
                else:
                    angle_placeholder.write("얼굴을 찾지 못했습니다.")

                if current_diff is not None and transformer.ref_angle is not None:
                    diff_placeholder.metric(
                        "기준 대비 차이",
                        f"{current_diff:+.2f}°",
                    )
                elif transformer.ref_angle is None:
                    diff_placeholder.write("기준 각도가 아직 설정되지 않았습니다.")
                else:
                    diff_placeholder.write("차이를 계산할 수 없습니다.")

    st.markdown("---")

    # ---- 3️⃣ 스냅샷 & CSV 다운로드 ----
    st.subheader("3️⃣ 스냅샷 및 기록 저장")

    if "ref_angle_value" in st.session_state:
        st.write(f"사용 중인 기준 각도: **{st.session_state['ref_angle_value']:.2f}°**")

    # 위에서 만든 webrtc_ctx 그대로 재사용
    transformer = None
    if webrtc_ctx and webrtc_ctx.video_transformer:
        transformer = webrtc_ctx.video_transformer  # type: ignore

    col1, col2 = st.columns(2)

    # 🔹 왼쪽: 스냅샷 저장 + 다운로드
    with col1:
        if transformer is None:
            st.info("위의 카메라를 먼저 켜고, 얼굴이 보이도록 해 주세요.")
        else:
            if st.button("현재 화면 스냅샷 저장"):
                if transformer.last_frame is not None:
                    img_png = encode_image_to_png_bytes(transformer.last_frame)
                    st.session_state["last_snapshot_png"] = img_png
                    st.session_state["snapshot_counter"] += 1
                    st.success("스냅샷을 임시로 저장했습니다. 아래에서 다운로드할 수 있습니다.")
                else:
                    st.warning("프레임이 아직 없습니다. 카메라가 켜져 있는지 확인해 주세요.")

        # 저장된 스냅샷이 있으면 다운로드 버튼 표시
        if st.session_state.get("last_snapshot_png") is not None:
            st.download_button(
                label=f"마지막 스냅샷 PNG 다운로드 (#{st.session_state['snapshot_counter']})",
                data=st.session_state["last_snapshot_png"],
                file_name=f"snapshot_{st.session_state['snapshot_counter']}.png",
                mime="image/png",
            )

    # 🔹 오른쪽: CSV 로그 다운로드
    with col2:
        if transformer is None:
            st.info("카메라가 켜진 이후에 각도 기록이 쌓입니다.")
        else:
            st.write("실시간 각도 기록을 CSV로 저장할 수 있습니다.")

            if transformer.log:
                csv_buffer = StringIO()
                csv_buffer.write("time,angle,diff\n")
                for row in transformer.log:
                    csv_buffer.write(
                        f"{row['time']},{row['angle']},{'' if row['diff'] is None else row['diff']}\n"
                    )

                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="측정 기록 CSV 다운로드",
                    data=csv_data,
                    file_name="face_angle_log.csv",
                    mime="text/csv",
                )
            else:
                st.info("아직 기록된 각도 로그가 없습니다. 카메라를 켜고 얼굴을 비춰보세요.")


if __name__ == "__main__":
    main()