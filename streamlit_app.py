import os
# cv2/ultralytics가 GUI 관련 라이브러리를 로드하지 않도록 사전 차단
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import time
import base64
import pandas as pd
from colorthief import ColorThief
import io


# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

# ============================================================
# CSS 스타일
# ============================================================
st.markdown("""
    <style>
    @font-face {
        font-family: 'OngleipParkDahyeon';
        src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2411-3@1.0/Ownglyph_ParkDaHyun.woff2') format('woff2');
        font-weight: normal;
        font-display: swap;
    }
    html, body, [class*="st-"] {
        font-family: 'OngleipParkDahyeon', sans-serif !important;
        font-size: 18px;
    }
    h1, h2, h3 {
        font-family: 'OngleipParkDahyeon', sans-serif !important;
        color: #5D5D5D;
        margin-bottom: 1rem;
    }
    .stApp {
        background-color: #f3ede9;
    }
    div.stButton > button:first-child {
        background-color: #EAB0C9;
        color: white;
        border-radius: 30px;
        border: none;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #EBC76A;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 2. 유틸리티 함수들
# ============================================================

YOLO_CLASS_NAMES = {
    0:  "나무전체", 1:  "기둥",   2:  "수관",   3:  "가지",
    4:  "뿌리",     5:  "나뭇잎", 6:  "꽃",     7:  "열매",
    8:  "그네",     9:  "새",     10: "다람쥐", 11: "구름",
    12: "달",       13: "별",     14: "사람전체",15: "머리",
    16: "얼굴",     17: "눈",     18: "코",     19: "입",
    20: "귀",       21: "남자머리카락", 22: "목", 23: "상체",
    24: "팔",       25: "손",     26: "다리",   27: "발",
    28: "단추",     29: "주머니", 30: "운동화", 31: "남자구두",
    32: "여자머리카락", 33: "여자구두", 34: "집전체", 35: "지붕",
    36: "집벽",     37: "문",     38: "창문",   39: "굴뚝",
    40: "연기",     41: "울타리", 42: "길",     43: "연못",
    44: "산",       45: "잔디",   46: "태양",
}


def rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def analyze_colors(image_file) -> dict:
    """ColorThief로 색채 분석 후 LLM에 넘길 수 있는 dict 반환."""
    img_bytes = io.BytesIO(image_file.getvalue())
    ct = ColorThief(img_bytes)

    dominant = ct.get_color(quality=1)
    palette  = ct.get_palette(color_count=5, quality=1)

    dominant_hex = rgb_to_hex(dominant)
    palette_info = [{"hex": rgb_to_hex(c), "rgb": list(c)} for c in palette]
    palette_hex_list = ", ".join(p["hex"] for p in palette_info)

    llm_summary = (
        f"그림의 주요(지배적) 색상은 {dominant_hex} "
        f"(RGB: {dominant[0]}, {dominant[1]}, {dominant[2]})입니다. "
        f"전체 색상 팔레트는 {palette_hex_list} 로 구성되어 있습니다."
    )
    return {
        "dominant_color": {"hex": dominant_hex, "rgb": list(dominant)},
        "palette": palette_info,
        "llm_summary": llm_summary,
    }


def run_yolo(image_file) -> dict:
    """
    YOLO 분석 후 LLM에 넘길 수 있는 dict 반환.
    ※ YOLO import를 함수 안에서 지연 로드해 앱 시작 시 cv2 충돌 방지.
    """
    # ── 지연 import: 앱 시작 시가 아닌 실제 사용 시점에만 로드 ──
    from ultralytics import YOLO  # noqa: PLC0415

    image = Image.open(image_file)
    model = YOLO("best.pt")
    results = model.predict(source=image, save=False, conf=0.25)

    objects = []
    for r in results:
        for box in r.boxes:
            x_c, y_c, w, h = box.xywhn[0].tolist()
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            cls_name   = YOLO_CLASS_NAMES.get(cls_id, r.names.get(cls_id, f"class_{cls_id}"))
            area_pct   = round(w * h * 100, 2)

            col = "왼쪽" if x_c < 0.33 else ("오른쪽" if x_c > 0.66 else "중앙")
            row = "상단" if y_c < 0.33 else ("하단" if y_c > 0.66 else "중간")
            position_label = f"{row} {col}"

            objects.append({
                "name":           cls_name,
                "x_center_pct":   round(x_c * 100),
                "y_center_pct":   round(y_c * 100),
                "area_pct":       area_pct,
                "position_label": position_label,
                "confidence":     round(confidence, 2),
            })

    if objects:
        items_str = ", ".join(
            f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%)"
            for o in objects
        )
        llm_summary = f"그림에서 감지된 요소: {items_str}."
    else:
        llm_summary = "그림에서 감지된 요소가 없습니다."

    plotted = results[0].plot() if results else None

    return {
        "objects":       objects,
        "llm_summary":   llm_summary,
        "plotted_image": plotted,
    }


def build_llm_prompt(color_result: dict, yolo_result: dict,
                     child_age: int, child_sex: str) -> str:
    """색채 + YOLO 분석 결과를 하나의 LLM 프롬프트 문자열로 조합."""
    return f"""
[그림 분석 결과 - 심리 해석 요청]

대상 아동 정보:
- 나이: {child_age}세
- 성별: {child_sex}

[색채 분석]
{color_result['llm_summary']}

[객체 감지 결과]
{yolo_result['llm_summary']}

위 정보를 바탕으로 아동의 심리 상태를 따뜻하고 희망적인 어조로 해석해 주세요.
분석 항목: 정서 상태, 자아상, 가족/환경 인식, 전반적인 심리적 특성.
""".strip()


# ============================================================
# 3. 팝업 다이얼로그
# ============================================================
@st.dialog("🎨 마음친구가 분석 중이에요!")
def show_analysis_popup(img_file):

    # ── 아직 분석 전: 진행바 표시하며 분석 수행 ──
    if not st.session_state.get("analysis_ready", False):

        status_text  = st.empty()
        progress_bar = st.progress(0)
        detail_text  = st.empty()

        try:
            # STEP 1: 색채 분석 (0 → 40%)
            status_text.markdown("**🎨 1단계: 그림 속 색깔을 읽고 있어요...**")
            detail_text.caption("그림에서 가장 많이 쓰인 색을 찾는 중")
            for pct in range(0, 41, 5):
                progress_bar.progress(pct)
                time.sleep(0.04)

            color_result = analyze_colors(img_file)
            st.session_state["color_result"] = color_result
            progress_bar.progress(40)
            detail_text.caption(f"✅ 주요 색상 {color_result['dominant_color']['hex']} 발견!")

            # STEP 2: YOLO 추론 (40 → 80%)
            status_text.markdown("**🤖 2단계: AI 눈을 켜는 중이에요...**")
            detail_text.caption("그림 분석 모델을 불러오는 중 (시간이 좀 걸려요 ⏳)")
            progress_bar.progress(45)

            yolo_result = run_yolo(img_file)
            st.session_state["yolo_result"] = yolo_result
            progress_bar.progress(80)

            found_count = len(yolo_result["objects"])
            detail_text.caption(f"✅ {found_count}개의 마음 조각을 찾았어요!")

            # STEP 3: 프롬프트 조합 (80 → 100%)
            status_text.markdown("**📝 3단계: 마음 이야기를 정리하는 중이에요...**")
            detail_text.caption("분석 결과를 정리하는 중")
            for pct in range(80, 101, 5):
                progress_bar.progress(pct)
                time.sleep(0.04)

            llm_prompt = build_llm_prompt(
                color_result,
                yolo_result,
                st.session_state.get("child_age", 7),
                st.session_state.get("child_sex", "남자"),
            )
            st.session_state["llm_prompt"]     = llm_prompt
            st.session_state["analysis_ready"] = True

            # 완료 메시지로 교체 (rerun 없이 그 자리에서 업데이트)
            status_text.markdown("**✨ 분석 완료! 아래 버튼을 눌러줘 💎**")
            detail_text.empty()
            progress_bar.progress(100)
            st.balloons()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            st.error(f"분석 중 에러가 발생했어요: {e}")
            st.exception(e)
            return

    # ── 분석 완료 후 버튼만 표시 ──
    if st.session_state.get("analysis_ready", False):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("보물 상자 열어보기 💎", use_container_width=True):
                st.session_state["analysis_done"] = True
                st.rerun()   # 팝업 닫고 메인 페이지 결과 섹션 렌더


# ============================================================
# 4. 배너 이미지
# ============================================================
def get_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_base64 = get_image_base64("main_concept_image.webp")
    st.markdown(
        f'<img src="data:image/webp;base64,{img_base64}" '
        f'alt="배너 이미지" style="width:100%; border-radius:15px;">',
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.info("✨ 예쁜 그림 친구가 곧 도착할 거예요!")


# ============================================================
# 5. 타이틀 & 초기화
# ============================================================
st.title("🌟 안녕? 나는 네 마음을 읽어주는 '마음친구'야!")
st.write("네가 그린 멋진 그림을 보여주면, 내가 네 마음을 토닥토닥해줄게. 🥰")

if "models_loaded" not in st.session_state:
    with st.spinner("AI 친구가 기지개를 켜며 준비하고 있어요... 💤"):
        time.sleep(1.0)
    st.session_state["models_loaded"]  = True
    st.session_state["analysis_done"]  = False
    st.session_state["analysis_ready"] = False
    st.session_state["chat_history"]   = []


# ============================================================
# 6. 입력 섹션
# ============================================================
st.header("1. 너에 대해 알려줘! 🎈")
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("🖼️ 네가 그린 멋진 그림을 보여줘")
    img_file = st.file_uploader(
        "그림 파일을 여기에 쏙 넣어줘!",
        type=["jpg", "png", "jpeg"],
        key="img",
    )
    if img_file:
        st.image(Image.open(img_file), use_container_width=True, caption="와! 정말 멋진 그림이야! ✨")

with col_input2:
    st.subheader("🧒 너에 대해 알려줘!")
    child_age = st.slider("내 나이는 이만큼이야!", 5, 13, 7)
    child_sex = st.radio("너는 남자니, 여자니?", ["남자", "여자"], horizontal=True)

    st.session_state["child_age"] = child_age
    st.session_state["child_sex"] = child_sex

    st.write("---")
    analyze_btn = st.button("마음친구야, 내 그림 좀 봐줄래? 🚀", use_container_width=True)


# ============================================================
# 7. 분석 버튼 → 팝업
# ============================================================
if analyze_btn:
    if not img_file:
        st.warning("그림 파일을 먼저 올려줘! 🖼️")
    else:
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_done"]  = False
        show_analysis_popup(img_file)


# ============================================================
# 8. 결과 섹션
# ============================================================
if st.session_state.get("analysis_done"):
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")

    color_result = st.session_state.get("color_result", {})
    yolo_result  = st.session_state.get("yolo_result", {})
    llm_prompt   = st.session_state.get("llm_prompt", "")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎨 마음 지도로 그려본 너의 그림")
        plotted = yolo_result.get("plotted_image")
        if plotted is not None:
            st.image(plotted, caption="마음친구가 찾아낸 보물들이야!", use_container_width=True)

        st.subheader("🖌️ 그림 속 색깔들")
        palette = color_result.get("palette", [])
        if palette:
            dominant_hex = color_result.get("dominant_color", {}).get("hex", "")
            st.markdown(
                f"**주요 색상:** "
                f'<span style="display:inline-block;width:20px;height:20px;'
                f'background:{dominant_hex};border-radius:4px;vertical-align:middle;'
                f'border:1px solid #ccc;"></span> {dominant_hex}',
                unsafe_allow_html=True,
            )
            swatch_html = "".join(
                f'<div style="display:inline-block;width:40px;height:40px;'
                f'background:{p["hex"]};border-radius:8px;margin:4px;'
                f'border:1px solid #ccc;" title="{p["hex"]}"></div>'
                for p in palette
            )
            st.markdown(swatch_html, unsafe_allow_html=True)

    with col2:
        st.subheader("🔍 찾아낸 마음 조각들")
        objects = yolo_result.get("objects", [])
        if objects:
            df = pd.DataFrame([
                {
                    "찾은 것": o["name"],
                    "위치":    o["position_label"],
                    "가로":    f"{o['x_center_pct']}%",
                    "세로":    f"{o['y_center_pct']}%",
                    "크기":    f"{o['area_pct']}%",
                    "신뢰도":  f"{o['confidence']:.0%}",
                }
                for o in objects
            ])
            st.table(df)

            with st.expander("💡 위치 정보로 보는 아이의 마음"):
                st.write("- **가로 위치**: 왼쪽은 과거/내향성, 오른쪽은 미래/외향성을 의미하기도 해요.")
                st.write("- **세로 위치**: 상단은 이상/상상력, 하단은 현실 감각을 나타내기도 해요.")
                st.write("- **크기**: 면적이 클수록 아이에게 중요한 대상일 수 있어요.")
        else:
            st.info("감지된 객체가 없어요. 그림을 다시 확인해줘!")

    with st.expander("🤖 LLM에 전달되는 분석 데이터 확인하기 (개발자용)"):
        st.code(llm_prompt, language="markdown")
        st.write("**색채 분석 (JSON):**")
        st.json({k: v for k, v in color_result.items() if k != "plotted_image"})
        st.write("**YOLO 분석 (JSON):**")
        st.json({k: v for k, v in yolo_result.items() if k != "plotted_image"})


# ============================================================
# 9. 대화 섹션
# ============================================================
if st.session_state.get("analysis_done"):
    st.divider()
    st.header("3. 우리 같이 도란도란 이야기하자 💬")

    st.subheader("🎙️ 네 목소리를 들려줘!")
    audio_file = st.audio_input("여기를 눌러서 목소리를 들려줘! 🎤")

    if audio_file:
        with st.spinner("네 목소리를 귀 기울여 듣고 있어... 👂"):
            time.sleep(1)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": "와! 목소리가 정말 예쁘구나. 방금 말해준 이야기, 나랑 같이 차근차근 생각해보자. 너는 정말 용기 있는 아이야! ❤️",
        })

    st.subheader("💬 마음친구랑 채팅하기")
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("하고 싶은 말을 여기에 적어줘!"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("생각 중... 💭"):
            time.sleep(1)
            response = (
                f"'{prompt}'라고 말해줘서 고마워! "
                "네 마음이 어떤지 조금 더 알 것 같아. "
                "우리 다음엔 또 어떤 즐거운 이야기를 해볼까? 😊"
            )
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
