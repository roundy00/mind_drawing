import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import json
import base64
import io
import time

import openai
import streamlit as st
from PIL import Image
import pandas as pd
from colorthief import ColorThief

# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

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
    .stApp { background-color: #f3ede9; }
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
# 2. 상수 & 유틸리티 (기존 코드 유지)
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

# HTP 주요 객체 그룹 (대화 우선순위 — 상위 그룹부터 질문)
HTP_MAIN_GROUPS = [
    {"group": "집", "names": {"집전체", "지붕", "집벽", "문", "창문", "굴뚝", "연기", "울타리"}},
    {"group": "나무", "names": {"나무전체", "기둥", "수관", "가지", "뿌리", "나뭇잎", "꽃", "열매"}},
    {"group": "사람", "names": {"사람전체", "머리", "얼굴", "눈", "코", "입", "귀",
                                 "남자머리카락", "여자머리카락", "목", "상체",
                                 "팔", "손", "다리", "발", "단추", "주머니",
                                 "운동화", "남자구두", "여자구두"}},
]


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def analyze_colors(image_file):
    img_bytes = io.BytesIO(image_file.getvalue())
    ct = ColorThief(img_bytes)
    dominant = ct.get_color(quality=1)
    palette  = ct.get_palette(color_count=5, quality=1)
    dominant_hex = rgb_to_hex(dominant)
    palette_info = [{"hex": rgb_to_hex(c), "rgb": list(c)} for c in palette]
    palette_hex_list = ", ".join(p["hex"] for p in palette_info)
    return {
        "dominant_color": {"hex": dominant_hex, "rgb": list(dominant)},
        "palette": palette_info,
        "llm_summary": (
            f"그림의 주요 색상은 {dominant_hex} "
            f"(RGB: {dominant[0]}, {dominant[1]}, {dominant[2]})입니다. "
            f"전체 색상 팔레트는 {palette_hex_list} 로 구성되어 있습니다."
        ),
    }


def run_yolo(image_file):
    from ultralytics import YOLO
    image  = Image.open(image_file)
    model  = YOLO("best.pt")
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
            objects.append({
                "name":           cls_name,
                "x_center_pct":   round(x_c * 100),
                "y_center_pct":   round(y_c * 100),
                "width_pct":      round(w * 100),
                "height_pct":     round(h * 100),
                "area_pct":       area_pct,
                "position_label": f"{row} {col}",
                "confidence":     round(confidence, 2),
            })
    items_str = ", ".join(
        f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%)" for o in objects
    ) if objects else "없음"
    return {
        "objects":       objects,
        "llm_summary":   f"그림에서 감지된 요소: {items_str}.",
        "plotted_image": results[0].plot() if results else None,
    }


# ============================================================
# 3. ★ 4단계: 대화 큐 구성
# ============================================================
def build_conversation_queue(yolo_result: dict) -> list[dict]:
    """
    YOLO 결과에서 HTP 주요 그룹(집/나무/사람) 중 탐지된 것만 추려
    대화 큐를 만든다.
    각 항목: {"group": "집", "objects": [...], "questions": [], "answers": []}
    """
    detected_names = {o["name"] for o in yolo_result["objects"]}
    queue = []
    for grp in HTP_MAIN_GROUPS:
        found = [o for o in yolo_result["objects"] if o["name"] in grp["names"]]
        if found:
            queue.append({
                "group":     grp["group"],
                "objects":   found,
                "questions": [],   # LLM이 채울 예정
                "answers":   [],
                "summary":   "",
            })
    # 탐지됐지만 어느 그룹에도 속하지 않는 기타 객체
    grouped_names = set()
    for grp in HTP_MAIN_GROUPS:
        grouped_names |= grp["names"]
    others = [o for o in yolo_result["objects"] if o["name"] not in grouped_names]
    if others:
        queue.append({"group": "기타", "objects": others,
                      "questions": [], "answers": [], "summary": ""})
    return queue


# ============================================================
# 4. ★ LLM 연동 (OpenAI)
# ============================================================
@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def call_llm(system: str, user: str, max_tokens: int = 800,
             image_base64: str = None) -> str:
    """단순 LLM 호출 래퍼. image_base64가 있으면 Vision으로 호출."""
    client = get_openai_client()
    if image_base64:
        content = [
            {"type": "text", "text": user},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "low"}},
        ]
    else:
        content = user
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": content},
        ],
    )
    return resp.choices[0].message.content.strip()


def transcribe_audio(audio_bytes: bytes) -> str:
    """Whisper STT."""
    client = get_openai_client()
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.webm", audio_bytes, "audio/webm"),
        language="ko",
    )
    return transcript.text


# ============================================================
# 5. ★ 4단계: 그룹별 시작 질문 생성
# ============================================================
QUESTION_GEN_SYSTEM = """
너는 아동과 따뜻하게 대화하는 미술 심리 선생님이야.
아동이 그린 HTP 그림의 분석 결과를 보고, 그 그림 요소에 대해
아동이 자유롭게 이야기할 수 있도록 개방형 질문을 만들어줘.

규칙:
- 쉽고 친근한 말투 (예: ~했어?, ~야?, ~줄래?)
- 진단·해석 표현 절대 금지
- 그림에서 보이는 구체적 특징(위치, 크기, 색)을 언급
- 질문은 짧고 명확하게 1~2문장
- 반드시 JSON만 반환: {"questions": ["질문1", "질문2", "질문3"]}
"""

def generate_questions_for_group(group_item: dict, color_summary: str,
                                  child_age: int, image_base64: str = None) -> list[str]:
    obj_desc = ", ".join(
        f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%)"
        for o in group_item["objects"]
    )
    user_msg = (
        f"아동 나이: {child_age}세\n"
        f"그림 요소 그룹: {group_item['group']}\n"
        f"감지된 세부 요소: {obj_desc}\n"
        f"색상 정보: {color_summary}\n\n"
        f"이 그룹에 대해 아동에게 물어볼 질문 3개를 만들어줘."
    )
    raw = call_llm(QUESTION_GEN_SYSTEM, user_msg,
                   max_tokens=400, image_base64=image_base64)
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip()).get("questions", [])[:3]
    except Exception:
        return [
            f"이 {group_item['group']}에 대해 이야기해줄 수 있어?",
            f"{group_item['group']}을 그릴 때 어떤 기분이었어?",
            f"{group_item['group']}에서 제일 마음에 드는 부분은 어디야?",
        ]


# ============================================================
# 6. ★ 5단계: 아동 답변 → 후속 반응 + 요약
# ============================================================
REACTION_SYSTEM = """
너는 아동과 대화하는 따뜻한 미술 심리 선생님이야.
아동의 답변에 공감하고 격려하는 짧은 반응과,
더 탐색할 여지가 있으면 자연스러운 후속 질문을 만들어줘.

규칙:
- 아동 눈높이 말투, 진단·해석 금지
- 반응: 1문장, 따뜻하고 짧게
- 후속 질문: 필요할 때만 1문장, 없으면 null
- 반드시 JSON만 반환: {"reaction": "...", "followup": "..." or null}
"""

SUMMARY_SYSTEM = """
미술 심리 전문가로서 아동과의 대화 기록을 바탕으로
이 그림 요소에 대한 객관적 관찰 요약을 3~4문장으로 작성해.
- 관찰 사실만, 해석/진단 없음
- 아동의 표현을 최대한 반영
"""

def get_reaction_and_followup(group_name: str, question: str, answer: str) -> tuple[str, str | None]:
    user_msg = f"그림 요소: {group_name}\n질문: {question}\n아동 답변: {answer}"
    raw = call_llm(REACTION_SYSTEM, user_msg, max_tokens=200)
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        return parsed.get("reaction", "잘 말해줬어!"), parsed.get("followup")
    except Exception:
        return "잘 이야기해줬어! 😊", None


def summarize_group_dialogue(group_item: dict) -> str:
    qa_text = "\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in zip(group_item["questions"], group_item["answers"])
        if a
    )
    if not qa_text:
        return ""
    return call_llm(
        SUMMARY_SYSTEM,
        f"그림 요소: {group_item['group']}\n\n대화 기록:\n{qa_text}",
        max_tokens=300,
    )


# ============================================================
# 7. ★ 6단계: RAG 기반 최종 레포트
# ============================================================
def load_vector_db():
    """FAISS 벡터 DB 로드 (캐싱)."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    DB_PATH = st.secrets.get("FAISS_DB_PATH", "./rag_results/final_index")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def retrieve_papers(query: str, k: int = 4) -> str:
    """FAISS에서 관련 논문 청크를 검색해 텍스트로 반환."""
    try:
        db   = load_vector_db()
        docs = db.similarity_search(query, k=k)
        return "\n\n".join(
            f"[출처: {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )
    except Exception as e:
        return f"(논문 검색 실패: {e})"


REPORT_SYSTEM = """
당신은 미술 심리 전문가입니다. 아동의 HTP 그림 분석 결과와
관련 논문 내용을 바탕으로 전문적이고 객관적인 분석 레포트를 작성하세요.

반드시 지킬 원칙:
- 교육적·객관적 톤 유지
- 의료적 진단이나 치료적 해석 절대 금지
- 관찰 사실 + 논문 근거 중심
- 부모·교사가 참고할 수 있는 실용적 내용

출력: 아래 JSON 구조로만 반환하세요.
{
  "executive_summary": "전체 요약 (2~3문단)",
  "key_findings": ["발견1", "발견2", "발견3"],
  "color_analysis": "색채 관찰 요약",
  "object_analysis": "객체 구성 관찰 요약",
  "dialogue_insights": "아동 대화에서 드러난 표현 특징",
  "recommendations": ["제안1", "제안2", "제안3"],
  "references_used": "참고한 논문 정보 요약"
}
"""

def generate_report(
    color_result: dict,
    yolo_result: dict,
    conversation_queue: list[dict],
    child_age: int,
    child_sex: str,
    image_base64: str = None,
) -> dict:
    # 대화 기록 텍스트화
    dialogue_text = ""
    for item in conversation_queue:
        qa = "\n".join(
            f"  Q: {q}\n  A: {a}"
            for q, a in zip(item["questions"], item["answers"]) if a
        )
        dialogue_text += f"\n[{item['group']}]\n{qa}\n요약: {item.get('summary','')}\n"

    # RAG 검색 쿼리: 대화 내용 + 주요 객체명
    rag_query = (
        f"HTP 그림 {', '.join(i['group'] for i in conversation_queue)} "
        f"아동 {child_age}세 {child_sex} 색채 심리"
    )
    paper_context = retrieve_papers(rag_query)

    user_msg = (
        f"아동 정보: {child_age}세 / {child_sex}\n\n"
        f"[색채 분석]\n{color_result['llm_summary']}\n\n"
        f"[객체 감지]\n{yolo_result['llm_summary']}\n\n"
        f"[아동 대화 기록]\n{dialogue_text}\n\n"
        f"[관련 논문 내용]\n{paper_context}"
    )

    raw = call_llm(REPORT_SYSTEM, user_msg,
                   max_tokens=1500, image_base64=image_base64)
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return {"executive_summary": raw, "key_findings": [],
                "color_analysis": "", "object_analysis": "",
                "dialogue_insights": "", "recommendations": [],
                "references_used": ""}


# ============================================================
# 8. session_state 초기화
# ============================================================
def init_state():
    defaults = {
        "models_loaded":       False,
        # 1~3단계
        "analysis_done":       False,
        "analysis_ready":      False,
        "color_result":        {},
        "yolo_result":         {},
        "image_base64":        None,
        "child_age":           7,
        "child_sex":           "남자",
        # 4~5단계
        "app_stage":           "upload",   # upload | chatting | reporting | done
        "conv_queue":          [],         # 대화 큐
        "current_group_idx":   0,          # 현재 그룹 인덱스
        "current_q_idx":       0,          # 현재 질문 인덱스
        "current_question":    "",
        # 6단계
        "report":              None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

if not st.session_state["models_loaded"]:
    with st.spinner("AI 친구가 기지개를 켜며 준비하고 있어요... 💤"):
        time.sleep(0.5)
    st.session_state["models_loaded"] = True


# ============================================================
# 9. 배너
# ============================================================
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_b64 = get_image_base64("main_concept_image.webp")
    st.markdown(
        f'<img src="data:image/webp;base64,{img_b64}" '
        f'alt="배너 이미지" style="width:100%; border-radius:15px;">',
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.info("✨ 예쁜 그림 친구가 곧 도착할 거예요!")


# ============================================================
# 10. 타이틀
# ============================================================
st.title("🌟 안녕? 나는 네 마음을 읽어주는 '마음친구'야!")
st.write("네가 그린 멋진 그림을 보여주면, 내가 네 마음을 토닥토닥해줄게. 🥰")


# ============================================================
# 11. 팝업 다이얼로그 (1~3단계) — 기존 코드 유지
# ============================================================
@st.dialog("🎨 마음친구가 분석 중이에요!")
def show_analysis_popup(img_file):
    if not st.session_state.get("analysis_ready", False):
        status_text  = st.empty()
        progress_bar = st.progress(0)
        detail_text  = st.empty()
        try:
            status_text.markdown("**🎨 1단계: 그림 속 색깔을 읽고 있어요...**")
            detail_text.caption("그림에서 가장 많이 쓰인 색을 찾는 중")
            for pct in range(0, 41, 5):
                progress_bar.progress(pct)
                time.sleep(0.04)
            color_result = analyze_colors(img_file)
            st.session_state["color_result"] = color_result
            progress_bar.progress(40)
            detail_text.caption(f"✅ 주요 색상 {color_result['dominant_color']['hex']} 발견!")

            status_text.markdown("**🤖 2단계: AI 눈을 켜는 중이에요...**")
            detail_text.caption("그림 분석 모델을 불러오는 중 (시간이 좀 걸려요 ⏳)")
            progress_bar.progress(45)
            yolo_result = run_yolo(img_file)
            st.session_state["yolo_result"] = yolo_result
            progress_bar.progress(80)
            detail_text.caption(f"✅ {len(yolo_result['objects'])}개의 마음 조각을 찾았어요!")

            # ★ 이미지를 base64로 저장 (이후 LLM Vision 호출용)
            img_file.seek(0)
            st.session_state["image_base64"] = base64.b64encode(img_file.read()).decode()

            status_text.markdown("**📝 3단계: 마음 이야기를 정리하는 중이에요...**")
            for pct in range(80, 101, 5):
                progress_bar.progress(pct)
                time.sleep(0.04)

            st.session_state["analysis_ready"] = True
            status_text.markdown("**✨ 분석 완료! 아래 버튼을 눌러줘 💎**")
            detail_text.empty()
            progress_bar.progress(100)
            st.balloons()
        except Exception as e:
            progress_bar.empty(); status_text.empty(); detail_text.empty()
            st.error(f"분석 중 에러가 발생했어요: {e}")
            st.exception(e)
            return

    if st.session_state.get("analysis_ready", False):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("보물 상자 열어보기 💎", use_container_width=True):
                st.session_state["analysis_done"] = True
                st.session_state["app_stage"]     = "result"
                st.rerun()


# ============================================================
# 12. 입력 섹션 (기존 코드 유지)
# ============================================================
st.header("1. 너에 대해 알려줘! 🎈")
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("🖼️ 네가 그린 멋진 그림을 보여줘")
    img_file = st.file_uploader(
        "그림 파일을 여기에 쏙 넣어줘!", type=["jpg", "png", "jpeg"], key="img"
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

if analyze_btn:
    if not img_file:
        st.warning("그림 파일을 먼저 올려줘! 🖼️")
    else:
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_done"]  = False
        show_analysis_popup(img_file)


# ============================================================
# 13. 결과 섹션 (기존 코드 유지 + 대화 버튼 추가)
# ============================================================
if st.session_state.get("analysis_done") and st.session_state["app_stage"] == "result":
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")

    color_result = st.session_state["color_result"]
    yolo_result  = st.session_state["yolo_result"]

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
            df = pd.DataFrame([{
                "찾은 것": o["name"], "위치": o["position_label"],
                "가로": f"{o['x_center_pct']}%", "세로": f"{o['y_center_pct']}%",
                "크기": f"{o['area_pct']}%", "신뢰도": f"{o['confidence']:.0%}",
            } for o in objects])
            st.table(df)
            with st.expander("💡 위치 정보로 보는 아이의 마음"):
                st.write("- **가로 위치**: 왼쪽은 과거/내향성, 오른쪽은 미래/외향성을 의미하기도 해요.")
                st.write("- **세로 위치**: 상단은 이상/상상력, 하단은 현실 감각을 나타내기도 해요.")
                st.write("- **크기**: 면적이 클수록 아이에게 중요한 대상일 수 있어요.")
        else:
            st.info("감지된 객체가 없어요. 그림을 다시 확인해줘!")

    st.divider()
    # ★ 4단계 시작 버튼
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🗣️ 이제 그림에 대해 이야기해보자!", use_container_width=True):
            with st.spinner("마음친구가 질문을 준비하고 있어요... 💭"):
                queue = build_conversation_queue(yolo_result)
                # 각 그룹 질문 미리 생성
                for item in queue:
                    item["questions"] = generate_questions_for_group(
                        item,
                        color_result["llm_summary"],
                        st.session_state["child_age"],
                        image_base64=st.session_state.get("image_base64"),
                    )
                st.session_state["conv_queue"]        = queue
                st.session_state["current_group_idx"] = 0
                st.session_state["current_q_idx"]     = 0
                st.session_state["app_stage"]         = "chatting"
            st.rerun()


# ============================================================
# 14. ★ 대화 섹션 (5단계)
# ============================================================
if st.session_state["app_stage"] == "chatting":
    st.header("3. 우리 같이 도란도란 이야기하자 💬")

    queue     = st.session_state["conv_queue"]
    g_idx     = st.session_state["current_group_idx"]
    q_idx     = st.session_state["current_q_idx"]

    # 모든 그룹 완료 체크
    if g_idx >= len(queue):
        st.success("모든 이야기를 다 들었어! 정말 잘했어 😊")
        st.session_state["app_stage"] = "reporting"
        st.rerun()

    current_group = queue[g_idx]
    total_groups  = len(queue)

    # 진행 상황
    st.progress(
        g_idx / total_groups,
        text=f"{current_group['group']} 이야기 중 ({g_idx + 1}/{total_groups})",
    )

    # 사이드바: 그림 + 이전 대화 요약
    with st.sidebar:
        st.subheader("🖼️ 내 그림")
        if img_file:
            st.image(Image.open(img_file), use_container_width=True)
        if g_idx > 0:
            st.subheader("📖 지금까지 이야기")
            for past in queue[:g_idx]:
                with st.expander(past["group"], expanded=False):
                    st.caption(past.get("summary", "요약 없음"))

    # 현재 질문 표시
    questions  = current_group["questions"]
    current_q  = questions[q_idx] if q_idx < len(questions) else None

    st.markdown(f"### 🎨 {current_group['group']}에 대해 이야기해요")
    if current_q:
        st.info(f"💬 {current_q}")

    # 입력 탭: 채팅 / 음성
    tab_chat, tab_voice = st.tabs(["💬 채팅으로 답하기", "🎤 음성으로 답하기"])

    def process_answer(answer_text: str):
        """답변 처리 → 반응 생성 → 상태 전환."""
        if not answer_text.strip():
            return
        reaction, followup = get_reaction_and_followup(
            current_group["group"], current_q, answer_text.strip()
        )
        # 답변 저장
        current_group["answers"].append(answer_text.strip())

        # 후속 질문 삽입
        if followup:
            questions.insert(q_idx + 1, followup)

        # 반응 저장 (다음 rerun 때 표시용)
        st.session_state["last_reaction"] = reaction

        # 다음 질문으로
        next_q_idx = q_idx + 1
        if next_q_idx < len(questions):
            st.session_state["current_q_idx"] = next_q_idx
        else:
            # 현재 그룹 완료 → 요약 생성 후 다음 그룹
            with st.spinner(f"{current_group['group']} 이야기를 정리하는 중..."):
                current_group["summary"] = summarize_group_dialogue(current_group)
            st.session_state["current_group_idx"] += 1
            st.session_state["current_q_idx"]     = 0
            st.session_state.pop("last_reaction",  None)

        st.rerun()

    # 마지막 반응 표시
    if reaction_msg := st.session_state.get("last_reaction"):
        st.success(f"마음친구: {reaction_msg}")

    with tab_chat:
        with st.form(key=f"chat_form_{g_idx}_{q_idx}"):
            answer_input = st.text_area(
                "답변을 입력하세요",
                placeholder="자유롭게 이야기해줘! ✏️",
                height=100,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("보내기 ➤")
        if submitted and answer_input.strip():
            with st.spinner("생각 중... 💭"):
                process_answer(answer_input)

    with tab_voice:
        st.caption("🎤 음성으로 답하고 싶으면 아래에서 녹음하거나 파일을 올려줘!")
        audio_input = st.audio_input(
            "여기를 눌러서 목소리를 들려줘!",
            key=f"audio_{g_idx}_{q_idx}",
        )
        if audio_input:
            with st.spinner("네 목소리를 귀 기울여 듣고 있어... 👂"):
                transcribed = transcribe_audio(audio_input.getvalue())
            if transcribed:
                st.caption(f"인식된 내용: *{transcribed}*")
                if st.button("이 내용으로 답하기 ✅", key=f"stt_btn_{g_idx}_{q_idx}"):
                    with st.spinner("생각 중... 💭"):
                        process_answer(transcribed)

    # 이전 문답 기록
    past_qa = list(zip(current_group["questions"][:q_idx],
                       current_group["answers"]))
    if past_qa:
        with st.expander("📜 이 주제에서 나눈 이야기", expanded=False):
            for q, a in past_qa:
                st.markdown(f"**마음친구:** {q}")
                st.markdown(f"**나:** {a}")
                st.divider()


# ============================================================
# 15. ★ 레포트 생성 섹션 (6단계)
# ============================================================
if st.session_state["app_stage"] == "reporting":
    st.header("4. 네 마음 보고서를 만들고 있어! 📋")

    if st.session_state["report"] is None:
        with st.spinner("논문을 찾아보고 보고서를 열심히 쓰고 있어요... ✍️ (1~2분 소요)"):
            report = generate_report(
                color_result   = st.session_state["color_result"],
                yolo_result    = st.session_state["yolo_result"],
                conversation_queue = st.session_state["conv_queue"],
                child_age      = st.session_state["child_age"],
                child_sex      = st.session_state["child_sex"],
                image_base64   = st.session_state.get("image_base64"),
            )
            st.session_state["report"]    = report
            st.session_state["app_stage"] = "done"
        st.rerun()


# ============================================================
# 16. ★ 레포트 표시 섹션
# ============================================================
if st.session_state["app_stage"] == "done" and st.session_state["report"]:
    report = st.session_state["report"]

    st.header("🌈 마음친구가 써준 보고서야!")
    st.caption("⚠️ 이 보고서는 교육 목적의 참고 자료이며, 의료적 진단이나 치료를 대체하지 않습니다.")

    # 핵심 요약
    if summary := report.get("executive_summary"):
        st.subheader("📌 전체 요약")
        st.write(summary)

    # 주요 발견
    if findings := report.get("key_findings"):
        st.subheader("🔍 주요 발견")
        for f in findings:
            st.markdown(f"- {f}")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if ca := report.get("color_analysis"):
            st.subheader("🎨 색채 관찰")
            st.write(ca)
        if di := report.get("dialogue_insights"):
            st.subheader("💬 대화에서 보인 표현 특징")
            st.write(di)
    with col_r2:
        if oa := report.get("object_analysis"):
            st.subheader("🏠🌳👤 그림 구성 관찰")
            st.write(oa)
        if recs := report.get("recommendations"):
            st.subheader("💡 선생님·부모님께 드리는 제안")
            for r in recs:
                st.markdown(f"- {r}")

    if refs := report.get("references_used"):
        with st.expander("📚 참고한 논문 정보"):
            st.write(refs)

    # 대화 기록 전체
    with st.expander("📖 전체 대화 기록 보기"):
        for item in st.session_state["conv_queue"]:
            st.markdown(f"**{item['group']}**")
            for q, a in zip(item["questions"], item["answers"]):
                st.markdown(f"- **Q:** {q}")
                st.markdown(f"  **A:** {a}")
            st.divider()

    # JSON 다운로드
    st.download_button(
        label="📄 보고서 JSON 다운로드",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="htp_report.json",
        mime="application/json",
    )

    st.divider()
    if st.button("🔄 새 그림 분석하기"):
        keep = {"models_loaded"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        st.rerun()
