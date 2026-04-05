import os
import warnings

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from",
)

import json
import re
import base64
import io
import time

import openai
import streamlit as st
from PIL import Image
import pandas as pd
from colorthief import ColorThief

from utils.stt import transcribe_audio

# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

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
    
    .main .block-container {
        padding-bottom: 15rem !important;
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

    .stApp span[data-testid="stIconMaterial"],
    .stApp div[data-testid="stExpanderHeader"] span[data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
        font-size: 1.4em;
        line-height: 1;
        font-style: normal;
        font-weight: normal;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        -webkit-font-feature-settings: 'liga';
        font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }

    .stApp div[data-testid="stExpander"] div[data-testid="stExpanderHeader"] {
        background-color: #F7E6F0;
        border-radius: 12px;
        padding: 0.6em 1em;
        font-weight: bold;
        color: #5D5D5D;
        cursor: pointer;
        min-height: 3em;
        display: flex;
        align-items: center;
        gap: 0.5em;
        transition: background-color 0.3s;
    }
    .stApp div[data-testid="stExpander"] div[data-testid="stExpanderHeader"]:hover {
        background-color: #EAB0C9;
        color: white;
    }
    .stApp div[data-testid="stExpander"][data-expanded="true"] div[data-testid="stExpanderHeader"] {
        background-color: #EAB0C9;
        color: white;
        border-radius: 12px 12px 0 0;
    }
    .stApp div[data-testid="stExpander"] div[data-testid="stExpanderHeader"] span[data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
        display: flex;
        align-items: center;
        flex-shrink: 0;
        margin-left: auto;
    }

    .voice-guide-box {
        background: linear-gradient(135deg, #f0e6f6 0%, #e8f4fd 100%);
        border: 1.5px solid #D4A8E0;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 15px;
        color: #5D5D5D;
        line-height: 1.7;
    }
    .voice-guide-box .step {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 4px;
    }
    .voice-guide-box .step-icon {
        font-size: 18px;
        flex-shrink: 0;
        margin-top: 1px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 음성 안내 문구 헬퍼
# ============================================================
VOICE_GUIDE_HTML = """
<div class="voice-guide-box">
  <div class="step"><span class="step-icon"></span><span>아래 <b>🎙️ 버튼</b>을 눌러 녹음을 시작해주세요.</span></div>
  <div class="step"><span class="step-icon"></span><span>말이 끝나면 <b>빨간 🔴 버튼</b>을 눌러 멈추면 인식된 텍스트를 확인할 수 있어요.</span></div>
  <div class="step"><span class="step-icon"></span><span>다시 녹음하려면 <b>🎙️ 버튼을 다시 눌러</b> 시작한 후 빨간 🔴 버튼으로 멈추면 돼요.</span></div>
</div>
"""

def show_voice_guide():
    st.markdown(VOICE_GUIDE_HTML, unsafe_allow_html=True)


# ============================================================
# 2. 상수 & 유틸리티
# ============================================================
YOLO_CLASS_NAMES = {
    0:  "나무전체", 1:  "기둥",   2:  "수관",   3:  "가지",
    4:  "뿌리",     5:  "나뭇잎", 6:  "꽃",     7:  "열매",
    8:  "그네",     9:  "새",     10: "다람쥐", 11: "구름",
    12: "달",       13: "별",     14: "사람전체", 15: "머리",
    16: "얼굴",     17: "눈",     18: "코",     19: "입",
    20: "귀",       21: "남자머리카락", 22: "목", 23: "상체",
    24: "팔",       25: "손",     26: "다리",   27: "발",
    28: "단추",     29: "주머니", 30: "운동화", 31: "남자구두",
    32: "여자머리카락", 33: "여자구두", 34: "집전체", 35: "지붕",
    36: "집벽",     37: "문",     38: "창문",   39: "굴뚝",
    40: "연기",     41: "울타리", 42: "길",     43: "연못",
    44: "산",       45: "잔디",   46: "태양",
}

HTP_MAIN_GROUPS = [
    {"group": "집",   "names": {"집전체", "지붕", "집벽", "문", "창문", "굴뚝", "연기", "울타리"}},
    {"group": "나무", "names": {"나무전체", "기둥", "수관", "가지", "뿌리", "나뭇잎", "꽃", "열매"}},
    {"group": "사람", "names": {"사람전체", "머리", "얼굴", "눈", "코", "입", "귀",
                                 "남자머리카락", "여자머리카락", "목", "상체",
                                 "팔", "손", "다리", "발", "단추", "주머니",
                                 "운동화", "남자구두", "여자구두"}},
]

BASE_QUESTIONS  = 3
MAX_FOLLOWUPS   = 1
MAX_Q_PER_GROUP = BASE_QUESTIONS + MAX_FOLLOWUPS

# app_stage 순서 정의 (단계 비교용)
STAGE_ORDER = ["upload", "describe", "result", "chatting", "reporting", "done"]

def stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return 0

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
    image   = Image.open(image_file)
    model   = YOLO("best.pt")
    results = model.predict(source=image, save=False, conf=0.25)
    objects = []
    for r in results:
        for box in r.boxes:
            x_c, y_c, w, h = box.xywhn[0].tolist()
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])

            # threshold
            if confidence < 0.6: continue

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
# 2-1. YOLO 오탐 검증 및 그림 설명 기반 보정
# ============================================================

YOLO_VERIFY_SYSTEM = """
너는 미술 심리 전문가이자 이미지 분석 보조자야.
아동이 직접 설명한 그림 내용과 AI 객체 탐지(YOLO) 결과를 비교해서,
탐지 결과 중 설명과 명확히 모순되는 항목만 오탐으로 표시해줘.

규칙:
- 아동의 설명을 최우선 기준으로 삼을 것
- 설명이 짧거나 불완전해도, 설명에 없다는 이유만으로 오탐 처리 금지
- 팔, 다리, 손, 발, 눈, 코, 입, 귀, 머리카락, 목 등 신체 세부 부위는
  설명에 사람 관련 내용이 조금이라도 있으면 오탐으로 처리하지 말 것.
  '사람전체'가 탐지되지 않아도 신체 부위만으로 사람이 존재한다고 판단할 수 있음.
- 집벽, 지붕, 문, 창문, 굴뚝 등 집 구성 요소는
  설명에 집과 관련된 내용(예: 집, 집에서, 집으로, 지붕, 문 등)이
  조금이라도 언급되었다면 오탐으로 처리하지 말 것.
  '집전체'가 탐지되지 않아도 구성 요소만으로 집이 존재한다고 판단할 수 있음.
- 나무기둥, 수관, 가지, 뿌리, 나뭇잎 등 나무 구성 요소는
  설명에 나무 관련 내용이 조금이라도 있으면 오탐으로 처리하지 말 것.
  '나무전체'가 탐지되지 않아도 구성 요소만으로 나무가 존재한다고 판단할 수 있음.
- 오탐으로 처리하는 기준은 오직 하나:
  설명이 해당 요소의 존재를 명확히 부정할 때만
  예) "집이 없어요", "나무를 그리지 않았어요", "사람은 없어요"
- 반드시 JSON만 반환:
{
  "verified_objects": [유효하다고 판단된 객체 name 목록],
  "suspicious_objects": [설명과 명확히 모순된 오탐 name 목록],
  "description_extras": [설명에는 있지만 YOLO가 탐지 못한 요소 목록],
  "notes": "간단한 검증 메모 (1~2문장)"
}
"""


def verify_yolo_with_description(
    yolo_result: dict,
    drawing_description: str,
    image_base64: str = None,
) -> dict:
    if not drawing_description.strip():
        return {
            "verified_objects":   [o["name"] for o in yolo_result["objects"]],
            "suspicious_objects": [],
            "description_extras": [],
            "notes": "설명이 입력되지 않아 검증을 생략했습니다.",
        }

    obj_list = ", ".join(
        f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%, 신뢰도 {o['confidence']})"
        for o in yolo_result["objects"]
    ) if yolo_result["objects"] else "없음"

    user_msg = (
        f"[그림 제작자가 직접 설명한 그림 내용]\n{drawing_description}\n\n"
        f"[AI(YOLO)가 탐지한 객체 목록]\n{obj_list}\n\n"
        "위 두 정보를 비교해서 검증 결과를 JSON으로 반환해줘."
    )

    try:
        raw = call_llm(YOLO_VERIFY_SYSTEM, user_msg,
                       max_tokens=500, image_base64=image_base64)
        result = extract_json(raw)
        return result
    except Exception as e:
        return {
            "verified_objects":   [o["name"] for o in yolo_result["objects"]],
            "suspicious_objects": [],
            "description_extras": [],
            "notes": f"검증 중 오류 발생: {e}",
        }


def apply_verification_to_yolo(yolo_result: dict, verification: dict) -> dict:
    suspicious = set(verification.get("suspicious_objects", []))
    extras     = verification.get("description_extras", [])

    updated_objects = []
    for obj in yolo_result["objects"]:
        obj_copy = dict(obj)
        obj_copy["is_suspicious"] = obj["name"] in suspicious
        updated_objects.append(obj_copy)

    for extra_name in extras:
        updated_objects.append({
            "name":           extra_name,
            "x_center_pct":   50,
            "y_center_pct":   50,
            "width_pct":      0,
            "height_pct":     0,
            "area_pct":       0.0,
            "position_label": "설명에서 확인",
            "confidence":     1.0,
            "is_suspicious":  False,
            "from_description": True,
        })

    valid_items = [o for o in updated_objects if not o.get("is_suspicious")]
    items_str = ", ".join(
        f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%)"
        + (" [설명기반]" if o.get("from_description") else "")
        for o in valid_items
    ) if valid_items else "없음"

    suspicious_str = ", ".join(suspicious) if suspicious else "없음"

    new_result = dict(yolo_result)
    new_result["objects"] = updated_objects
    new_result["llm_summary"] = (
        f"그림에서 감지된 요소: {items_str}. "
        f"(오탐 의심으로 제외된 요소: {suspicious_str}. "
        f"검증 메모: {verification.get('notes', '')})"
    )
    new_result["verification"] = verification
    return new_result


# ============================================================
# 3. 대화 큐 구성
# ============================================================
def build_conversation_queue(yolo_result: dict) -> list[dict]:
    grouped_names = set()
    for grp in HTP_MAIN_GROUPS:
        grouped_names |= grp["names"]

    other_objects = [o for o in yolo_result["objects"] if o["name"] not in grouped_names]
    other_summary = ", ".join(o["name"] for o in other_objects) if other_objects else ""

    queue = []
    for grp in HTP_MAIN_GROUPS:
        found = [
            o for o in yolo_result["objects"]
            if o["name"] in grp["names"] and not o.get("is_suspicious", False)
        ]
        if found:
            queue.append({
                "group":         grp["group"],
                "objects":       found,
                "other_context": other_summary,
                "questions":     [],
                "answers":       [],
                "summary":       "",
            })
    return queue


# ============================================================
# 4. LLM 연동 (OpenAI)
# ============================================================
@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def call_llm(system: str, user: str, max_tokens: int = 800,
             image_base64: str = None) -> str:
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

    choice = resp.choices[0]
    if choice.message.content is None:
        raise ValueError(f"LLM 응답 content가 None입니다. finish_reason: {choice.finish_reason}")

    return choice.message.content.strip()


# ============================================================
# 5. 그룹별 시작 질문 생성
# ============================================================
QUESTION_GEN_SYSTEM = """
너는 그림 제작자와 따뜻하게 대화하는 미술 심리 선생님이야.
그림 제작자가 그린 HTP 그림의 분석 결과와 그림 제작자(또는 보호자)가 직접 설명한 그림 내용을 보고,
그 그림 요소에 대해 그림 제작자가 자유롭게 이야기할 수 있도록 개방형 질문을 만들어줘.

규칙:
- 이 이미지는 실제 사람이 아니라 그림 제작자가 그린 그림(캐릭터/상상 표현)임을 항상 전제로 할 것
- 사람처럼 보이는 요소가 있어도 실제 인물로 해석하지 말 것
- 절대 인물 식별, 나이 추정, 정체 추측을 하지 말 것
- 그림 제작자의 연령이 쉽게 이해할 수 있는 질문을 할 것
- 쉽고 친근한 말투 (예: ~했어?, ~야?, ~줄래?), 말투에 대해서는 해당 세션이 종료될 때까지 유지할 것
- 진단·해석 표현 절대 금지
- 그림에서 보이는 구체적 특징(위치, 크기, 색)을 언급
- 질문은 짧고 명확하게 1~2문장
- [중복 금지] 이미 한 질문 목록이 제공되면,
  같은 의도·주제·표현의 질문은 절대 반복하지 마.
  반드시 완전히 다른 주제나 관점으로 질문해야 해.
- [중요] 그림 설명(drawing_description)에서 이미 충분히 답변된 내용은 다시 묻지 말 것.
  설명을 통해 알게 된 내용을 바탕으로, 아직 이야기되지 않은 부분을 탐색하는 질문을 만들 것.
- 모든 질문이 그림 설명만으로 이미 답변되었다면 questions 배열을 빈 배열([])로 반환할 것.
- 반드시 JSON만 반환: {"questions": ["질문1", "질문2", "질문3"]} 또는 {"questions": []}
"""


def _is_duplicate(new_q: str, existing: list[str]) -> bool:
    STOPWORDS = {
        "이", "가", "을", "를", "은", "는", "에", "의", "도", "로", "으로",
        "에서", "와", "과", "하고", "한", "해", "했", "어", "야", "줄래",
        "있어", "어때", "어땠어", "있니", "했어", "어떤", "이야", "말해줄",
        "수", "것", "때", "좀", "더", "가장", "제일", "정말", "혹시",
    }
    def tokenize(s: str) -> set[str]:
        tokens = set()
        for tok in s.replace("?", "").replace("!", "").replace(",", "").split():
            if len(tok) >= 2 and tok not in STOPWORDS:
                tokens.add(tok)
        return tokens

    new_tokens = tokenize(new_q)
    if not new_tokens:
        return False
    for ex in existing:
        if len(new_tokens & tokenize(ex)) >= 2:
            return True
    return False


def _is_llm_refusal(text: str) -> bool:
    REFUSAL_HINTS = [
        "i'm sorry", "i am sorry", "i cannot", "i can't",
        "sorry, but", "unable to", "can't help", "cannot help",
    ]
    return any(hint in text.lower() for hint in REFUSAL_HINTS)


def extract_json(raw: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        return json.loads(match.group(1).strip())
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group(0).strip())
    raise ValueError(f"JSON을 찾을 수 없음: {raw[:200]}")


def generate_questions_for_group(
    group_item,
    color_summary,
    child_age,
    drawing_description: str = "",
    image_base64=None,
    existing_questions: list[str] | None = None,
) -> list[str] | None:
    FALLBACK = [
        f"이 {group_item['group']}에 대해 이야기해줄 수 있어?",
        f"{group_item['group']}을 그릴 때 어떤 기분이었어?",
        f"{group_item['group']}에서 제일 마음에 드는 부분은 어디야?",
    ]

    obj_desc = ", ".join(
        f"{o['name']}({o['position_label']}, 면적 {o['area_pct']}%)"
        for o in group_item["objects"]
        if not o.get("is_suspicious", False)
    )

    existing_block = ""
    if existing_questions:
        existing_block = (
            "\n\n[이미 한 질문 — 이것과 비슷한 질문은 만들지 마]\n"
            + "\n".join(f"- {q}" for q in existing_questions)
        )

    description_block = ""
    if drawing_description.strip():
        description_block = (
            f"\n\n[그림 제작자(또는 보호자)가 미리 설명한 그림 내용 — 이미 답된 내용은 다시 묻지 마]\n"
            f"{drawing_description.strip()}"
        )

    user_msg = (
        "이 이미지는 그림입니다. 실제 사람이 아닌 캐릭터와 상상 표현입니다.\n"
        "이미지 속 인물처럼 보이는 요소도 실제 인물이 아닙니다.\n\n"
        f"그림 제작자 연령: {child_age}세\n"
        f"그림 요소 그룹: {group_item['group']}\n"
        f"감지된 세부 요소: {obj_desc}\n"
        f"색상 정보: {color_summary}"
        f"{description_block}"
        f"{existing_block}\n\n"
        f"이 그룹에 대해 그림 제작자에게 물어볼 질문을 만들어줘. "
        f"그림 설명에서 이미 충분히 답변되었다면 빈 배열([])을 반환해."
    )

    try:
        raw = call_llm(QUESTION_GEN_SYSTEM, user_msg,
                       max_tokens=400, image_base64=image_base64)
    except Exception as e:
        st.session_state["q_gen_error"] = f"LLM 호출 실패: {type(e).__name__} — {e}"
        return None

    if _is_llm_refusal(raw):
        st.session_state.pop("q_gen_error", None)
        return FALLBACK

    try:
        new_qs = extract_json(raw).get("questions", [])[:3]
        if new_qs == []:
            st.session_state.pop("q_gen_error", None)
            return []
    except Exception as e:
        st.session_state["q_gen_error"] = (
            f"JSON 파싱 실패: {type(e).__name__} — {e}\n\nLLM 원본 응답:\n{raw[:500]}"
        )
        return None

    if existing_questions:
        existing_set = set(existing_questions)
        seen = list(existing_questions)
        filtered = []
        for q in new_qs:
            if q in existing_set or _is_duplicate(q, seen):
                continue
            filtered.append(q)
            seen.append(q)
        new_qs = filtered

    if not new_qs:
        st.session_state.pop("q_gen_error", None)
        return []

    st.session_state.pop("q_gen_error", None)
    return new_qs


# ============================================================
# 6. 아동 답변 → 후속 반응 + 요약
# ============================================================
REACTION_SYSTEM = """
너는 그림 제작자와 대화하는 따뜻한 미술 심리 선생님이야.
그림 제작자의 답변에 공감하고 격려하는 짧은 반응과,
더 탐색할 여지가 있으면 자연스러운 후속 질문을 만들어줘.

규칙:
- 사람과 관련된 대화는 그림 제작자가 그린 그림 속 캐릭터에 대한 이야기임을 유지할 것
- 그림 제작자 눈높이 말투, 진단·해석 금지
- 반응: 1문장, 따뜻하고 짧게
- 후속 질문: 필요할 때만 1문장, 없으면 null
  단, 이미 한 질문 목록과 의미·주제가 겹치면 반드시 null로 반환해.
- 반드시 JSON만 반환: {"reaction": "...", "followup": "..." or null}
"""

SUMMARY_SYSTEM = """
미술 심리 전문가로서 그림 제작자와의 대화 기록을 바탕으로
이 그림 요소에 대한 객관적 관찰 요약을 3~4문장으로 작성해.
- 관찰 사실만, 해석/진단 없음
- 그림 제작자의 표현을 최대한 반영
"""


def summarize_group_dialogue(group_item: dict) -> str:
    qa_text = "\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in zip(group_item["questions"], group_item["answers"]) if a
    )
    if not qa_text:
        return ""
    return call_llm(
        SUMMARY_SYSTEM,
        f"그림 요소: {group_item['group']}\n\n대화 기록:\n{qa_text}",
        max_tokens=300,
    )


def get_reaction_and_followup(
    group_name: str,
    question: str,
    answer: str,
    existing_questions: list[str] | None = None,
) -> tuple[str, str | None]:
    asked_block = ""
    if existing_questions:
        asked_block = (
            "\n\n[이미 한 질문 목록]\n"
            + "\n".join(f"- {q}" for q in existing_questions)
        )
    user_msg = (
        f"그림 요소: {group_name}\n"
        f"질문: {question}\n"
        f"그림 제작자 답변: {answer}"
        f"{asked_block}"
    )
    raw = call_llm(REACTION_SYSTEM, user_msg, max_tokens=200)
    try:
        parsed   = extract_json(raw)
        reaction = parsed.get("reaction", "잘 말해줬어!")
        followup = parsed.get("followup")
        if followup and existing_questions and _is_duplicate(followup, existing_questions):
            followup = None
        return reaction, followup
    except Exception:
        return "잘 이야기해줬어! 😊", None


# ============================================================
# 7. RAG 기반 최종 레포트
# ============================================================

FAISS_OPENAI_DB_PATH = "./final_index"
FAISS_ST_DB_PATH     = "./st_index"


@st.cache_resource
def load_openai_vector_db():
    import glob
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    faiss_files = glob.glob(f"{FAISS_OPENAI_DB_PATH}/*.faiss")
    if not faiss_files:
        raise FileNotFoundError(
            f"'{FAISS_OPENAI_DB_PATH}' 안에 .faiss 파일이 없어요."
        )
    index_name = os.path.splitext(os.path.basename(faiss_files[0]))[0]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return FAISS.load_local(
        FAISS_OPENAI_DB_PATH, embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_st_vector_db():
    import glob
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    faiss_files = glob.glob(f"{FAISS_ST_DB_PATH}/*.faiss")
    if not faiss_files:
        raise FileNotFoundError(
            f"'{FAISS_ST_DB_PATH}' 안에 .faiss 파일이 없어요."
        )
    index_name = os.path.splitext(os.path.basename(faiss_files[0]))[0]
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(
        FAISS_ST_DB_PATH, embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def retrieve_papers(query: str, k: int = 4) -> tuple[str, list[dict]]:
    all_docs  = []
    seen_keys = set()

    try:
        db_openai   = load_openai_vector_db()
        docs_openai = db_openai.similarity_search(query, k=k)
        for d in docs_openai:
            key = (d.metadata.get("source", ""), d.metadata.get("page", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(("OpenAI", d))
    except Exception as e:
        print(f"[FAISS DEBUG] OpenAI DB 검색 실패: {e}")

    try:
        db_st   = load_st_vector_db()
        docs_st = db_st.similarity_search(query, k=k)
        for d in docs_st:
            key = (d.metadata.get("source", ""), d.metadata.get("page", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(("ST", d))
    except Exception as e:
        print(f"[FAISS DEBUG] SentenceTransformer DB 검색 실패: {e}")

    if not all_docs:
        return "(논문 검색 실패: 두 DB 모두 검색 결과 없음)", []

    meta_list = []
    chunks    = []
    for db_name, d in all_docs:
        source = d.metadata.get("source", "?")
        page   = d.metadata.get("page",   "?")
        title  = d.metadata.get("title",  None)
        label  = title if title else source
        meta_list.append({"title": title, "source": source, "page": page, "db": db_name})
        chunks.append(f"[출처: {label} p.{page}]\n{d.page_content}")

    return "\n\n".join(chunks), meta_list


REFUSAL_HINTS = [
    "i'm sorry", "i am sorry", "i cannot", "i can't",
    "sorry, but", "unable to", "can't help", "cannot help",
]


def _has_refusal(text: str) -> bool:
    return any(h in text.lower() for h in REFUSAL_HINTS)


EMOTION_ANALYSIS_SYSTEM = """
당신은 미술 심리 전문가입니다.
다음 그림 관찰 결과를 바탕으로 감정적 언어를 분석하세요.

분석 항목:
1. 주요 감정 (색상, 형태, 구성에서 나타나는 감정적 표현)
2. 감정적 톤 (전체적인 감정적 분위기)
3. 상징적 요소 (반복되는 패턴, 특별한 기호나 상징)
4. 강도 수준 (감정 표현의 강도: 약함/보통/강함)

중요 원칙:
- 이 이미지는 그림 제작자가 그린 그림이며 실제 인물이 아님. 그림 속 인물도 상상 표현으로 간주할 것
- 관찰된 사실을 바탕으로 감정적 표현을 식별하세요
- 진단이나 치료적 해석은 절대 하지 마세요
- 색채, 형태에 대한 일반적 지식을 참고하되 객관적으로 기록하세요
- "이 색상은 슬픔을 의미한다"가 아니라 "이 색상은 일반적으로 슬픔과 연관된다"로 기록하세요
- [그림 제작자의 그림 설명]이 제공된 경우, 이를 중요한 1차 정보로 반영할 것
- JSON 외 문장을 앞뒤에 절대 붙이지 말 것

출력: 아래 JSON 구조로만 반환하세요.
{
  "dominant_emotions": ["감정1", "감정2", "감정3"],
  "emotional_tone": "감정적 톤에 대한 상세한 설명 (2~3문장)",
  "symbolic_elements": ["시각적 특징 요소1", "요소2", "요소3"],
  "intensity_level": "매우 약함 | 약함 | 보통 | 강함 | 매우 강함 중 하나"
}

⚠️ 본 분석은 교육 목적의 참고 자료이며, 진단이나 치료를 목적으로 하지 않습니다.
"""

EXPLORATION_QUESTIONS_SYSTEM = """
당신은 미술 심리 전문가입니다.
다음 그림 분석 결과를 바탕으로 그림 제작자와 보호자(부모·교사)가 자기 탐색에 활용할 수 있는
개방형 질문을 생성하세요.

질문 생성 원칙:
1. 그림의 구체적인 요소에 대한 탐색 질문 (색상, 형태, 구성 등)
2. 감정적 표현에 대한 탐색 질문
3. 시각적 특징에 대한 탐색 질문
4. 그림 제작 경험에 대한 탐색 질문

질문 특성:
- 이 이미지는 그림 제작자가 그린 그림이며 실제 인물이 아님. 그림 속 인물도 상상 표현으로 간주할 것
- 개방적이고 탐색적인 질문
- 그림 제작자가 자신의 그림을 더 깊이 이해하고 탐색할 수 있도록 구성
- 각 카테고리별로 2~3개의 질문 생성
- 진단이나 치료 목적의 질문이 아닌, 교육적 자기 탐색 질문만 생성
- [그림 제작자의 그림 설명]이 제공된 경우, 이미 언급된 내용은 되묻지 말 것
- JSON 외 문장을 앞뒤에 절대 붙이지 말 것

출력: 아래 JSON 구조로만 반환하세요.
{
  "questions": ["질문1", "질문2", "질문3", "질문4", "질문5"],
  "categories": ["색상/형태", "감정", "상징", "경험"],
  "purpose": "질문 생성의 목적 설명 (1~2문장)"
}

⚠️ 진단이나 치료 목적의 질문이 아닌, 교육적 자기 탐색 질문만 생성하세요.
"""

REPORT_SYSTEM = """
미술 심리 전문가로서 다른 전문가들의 보고서(감정 언어 분석, 탐색 질문)를 포함한
전체 관찰 결과를 종합적으로 심리 분석하여 전문적이고 객관적인 결론을 작성하세요.

분석 구조:
1. 요약 및 핵심 인사이트 (Executive Summary): 전체 분석의 핵심 요약 (2~3문단), 가장 중요한 발견 3~5개
2. 주요 발견 사항 (Key Findings): 관찰·감정 분석·대화에서 도출된 주요 발견, 객관적 사실 중심
3. 활용 방향성 제시 (Usage Direction): 이 그림을 바탕으로 한 교육적 활용의 주요 방향성
4. 집중 탐색 영역 (Focus Areas): 활용 시 주목할 구체적 영역과 간단한 설명
5. AI 시스템 종합 분석 (Comprehensive Analysis): 관찰 기반의 종합 분석 (2~3문단)
6. 참고 사항 (References): 다음 단계 제안, 실용적 참고 사항, 논문 인용

반드시 지킬 원칙:
- 이 서비스는 한국어 서비스입니다. 반드시 한국어로 작성하시기 바랍니다.
- 그림 속 인물 표현은 모두 상상 또는 상징으로 간주할 것
- 특정 개인 식별, 나이/성별/인종 추정 금지
- 객관적이고 교육적인 톤 유지
- 의료적 진단이나 치료적 해석 절대 금지
- 해석이나 진단은 포함하지 않고, 관찰된 사실을 바탕으로 한 분석만 제시
- 부모·교사가 교육 목적으로 활용할 수 있는 실용적 내용
- [그림 제작자의 그림 설명]이 제공된 경우, 이를 중요한 1차 정보로 반영할 것
- [관련 논문 내용]에 제공된 논문만 references_used에 인용할 것
- references_used 형식: 논문명 | 저자 | 연도 | 저널명 | URL 또는 DOI (항목마다 줄바꿈)
- JSON 외 문장을 앞뒤에 절대 붙이지 말 것

출력: 아래 JSON 구조로만 반환하세요.
{
  "executive_summary": "전체 요약 및 핵심 인사이트 (2~3문단)",
  "key_findings": ["발견1", "발견2", "발견3", "발견4", "발견5"],
  "usage_direction": "활용 방향성 설명 (2~3문장)",
  "focus_areas": ["집중 탐색 영역1: 설명", "영역2: 설명", "영역3: 설명"],
  "comprehensive_analysis": "AI 시스템 종합 분석 (2~3문단)",
  "color_analysis": "색채 관찰 요약",
  "object_analysis": "객체 구성 관찰 요약",
  "dialogue_insights": "그림 제작자와의 대화 및 탐색 질문에서 드러난 표현 특징",
  "recommendations": ["참고 사항1", "참고 사항2", "참고 사항3"],
  "references_used": "논문명 | 저자 | 연도 | 저널명 | DOI 또는 URL\n논문명 | ..."
}

⚠️ 본 분석은 교육 목적의 참고 자료이며, 의료적 진단이나 치료를 대체하지 않습니다.
"""


def _build_analysis_context(
    color_result: dict,
    yolo_result: dict,
    conversation_queue: list[dict],
    child_age: int,
    child_sex: str,
    drawing_description: str = "",
) -> str:
    dialogue_text = ""
    for item in conversation_queue:
        qa = "\n".join(
            f"  Q: {q}\n  A: {a}"
            for q, a in zip(item["questions"], item["answers"]) if a
        )
        if item.get("covered_by_description"):
            dialogue_text += f"\n[{item['group']}]\n  ※ 그림 설명으로 충분히 파악됨 — 별도 대화 없음\n"
        else:
            dialogue_text += f"\n[{item['group']}]\n{qa}\n요약: {item.get('summary','')}\n"

    description_block = ""
    if drawing_description.strip():
        description_block = (
            f"\n\n[그림 제작자(또는 보호자)가 미리 설명한 그림 내용]\n{drawing_description.strip()}"
        )

    verification = yolo_result.get("verification", {})
    verification_block = ""
    if verification:
        suspicious = verification.get("suspicious_objects", [])
        extras     = verification.get("description_extras", [])
        notes      = verification.get("notes", "")
        if suspicious or extras:
            verification_block = (
                f"\n\n[YOLO 검증 결과]\n"
                f"오탐 의심 항목: {', '.join(suspicious) if suspicious else '없음'}\n"
                f"설명에서 추가 확인된 항목: {', '.join(extras) if extras else '없음'}\n"
                f"검증 메모: {notes}"
            )

    return (
        f"그림 제작자 정보: {child_age}세 / {child_sex}\n\n"
        f"[색채 분석]\n{color_result['llm_summary']}\n\n"
        f"[객체 감지]\n{yolo_result['llm_summary']}"
        f"{verification_block}"
        f"{description_block}\n\n"
        f"[그림 제작자와의 대화 기록]\n{dialogue_text}"
    )


def _safe_llm_json(
    system: str,
    user: str,
    step_name: str,
    errors: list[dict],
    max_tokens: int = 800,
    image_base64: str = None,
) -> dict | None:
    try:
        raw = call_llm(system, user, max_tokens=max_tokens, image_base64=image_base64)
    except Exception as e:
        errors.append({"step": step_name, "reason": f"LLM 호출 실패: {type(e).__name__} — {e}"})
        return None

    if _has_refusal(raw):
        errors.append({"step": step_name, "reason": "AI가 응답을 거부했습니다."})
        return None

    try:
        return extract_json(raw)
    except Exception as e:
        errors.append({
            "step":   step_name,
            "reason": f"JSON 파싱 실패: {type(e).__name__} — {e}",
            "raw":    raw[:500],
        })
        return None


def generate_report(
    color_result: dict,
    yolo_result: dict,
    conversation_queue: list[dict],
    child_age: int,
    child_sex: str,
    drawing_description: str = "",
    image_base64: str = None,
) -> dict:
    errors: list[dict] = []

    base_ctx = _build_analysis_context(
        color_result, yolo_result, conversation_queue,
        child_age, child_sex, drawing_description,
    )

    emotion_result = _safe_llm_json(
        EMOTION_ANALYSIS_SYSTEM,
        base_ctx,
        step_name="감정 언어 분석",
        errors=errors,
        max_tokens=600,
    ) or {
        "dominant_emotions": [],
        "emotional_tone":    "(분석 실패)",
        "symbolic_elements": [],
        "intensity_level":   "(분석 실패)",
    }

    emotion_block = (
        f"\n\n[감정 언어 분석 결과]\n"
        f"주요 감정: {', '.join(emotion_result.get('dominant_emotions', []))}\n"
        f"감정적 톤: {emotion_result.get('emotional_tone', '')}\n"
        f"상징적 요소: {', '.join(emotion_result.get('symbolic_elements', []))}\n"
        f"감정 강도: {emotion_result.get('intensity_level', '')}"
    )
    exploration_result = _safe_llm_json(
        EXPLORATION_QUESTIONS_SYSTEM,
        base_ctx + emotion_block,
        step_name="탐색 질문 생성",
        errors=errors,
        max_tokens=600,
    ) or {
        "questions":  [],
        "categories": [],
        "purpose":    "(생성 실패)",
    }

    rag_query = (
        f"HTP 그림 {', '.join(i['group'] for i in conversation_queue)} 색채 심리"
    )
    paper_context = ""
    faiss_meta    = []
    try:
        paper_context, faiss_meta = retrieve_papers(rag_query)
        if paper_context.startswith("(논문 검색 실패"):
            errors.append({"step": "FAISS 논문 검색", "reason": paper_context})
            paper_context = ""
    except Exception as e:
        errors.append({
            "step":   "FAISS 논문 검색",
            "reason": f"예외 발생: {type(e).__name__} — {e}",
        })

    if not paper_context:
        errors.append({
            "step":   "FAISS 논문 없음",
            "reason": "두 FAISS DB 모두에서 관련 논문을 찾지 못했어요.",
        })
        return {
            "executive_summary": "", "key_findings": [],
            "usage_direction": "",   "focus_areas": [],
            "comprehensive_analysis": "",
            "color_analysis": "",    "object_analysis": "",
            "dialogue_insights": "", "recommendations": [],
            "references_used": "",
            "emotion_analysis":      emotion_result,
            "exploration_questions": exploration_result,
            "report_errors": errors,
            "faiss_meta":    faiss_meta,
        }

    exploration_block = (
        f"\n\n[자기 탐색 질문 목록]\n"
        + "\n".join(f"- {q}" for q in exploration_result.get("questions", []))
        + f"\n탐색 목적: {exploration_result.get('purpose', '')}"
    )

    report_user_msg = (
        base_ctx
        + emotion_block
        + exploration_block
        + f"\n\n[관련 논문 내용 — 반드시 아래 논문만 인용할 것]\n{paper_context}"
    )

    report_parsed = _safe_llm_json(
        REPORT_SYSTEM,
        report_user_msg,
        step_name="종합 보고서 생성",
        errors=errors,
        max_tokens=2000,
    )

    if report_parsed is None:
        report_parsed = {
            "executive_summary": "", "key_findings": [],
            "usage_direction": "",   "focus_areas": [],
            "comprehensive_analysis": "",
            "color_analysis": "",    "object_analysis": "",
            "dialogue_insights": "", "recommendations": [],
            "references_used": "",
        }

    report_parsed["emotion_analysis"]      = emotion_result
    report_parsed["exploration_questions"] = exploration_result
    report_parsed["report_errors"]         = errors
    report_parsed["faiss_meta"]            = faiss_meta
    return report_parsed


# ============================================================
# 7-1. 전체 재생성
# ============================================================

def regen_full_report() -> None:
    errors: list[dict] = []
    try:
        report = generate_report(
            color_result        = st.session_state["color_result"],
            yolo_result         = st.session_state["yolo_result"],
            conversation_queue  = st.session_state["conv_queue"],
            child_age           = st.session_state["child_age"],
            child_sex           = st.session_state["child_sex"],
            drawing_description = st.session_state.get("drawing_description", ""),
        )
        st.session_state["report"]        = report
        st.session_state["report_errors"] = report.get("report_errors", [])
        st.toast("✅ 보고서 재생성 완료!")
    except Exception as e:
        errors.append({"step": "보고서 재생성", "reason": f"{type(e).__name__}: {e}"})
        st.session_state["report_errors"] = errors
        st.toast(f"❌ 재생성 실패: {e}", icon="⚠️")


# ============================================================
# 8. 대화 답변 처리 로직
# ============================================================
def process_answer(
    answer_text: str,
    current_group: dict,
    questions: list,
    q_idx: int,
    reaction: str,
    followup: str | None,
) -> dict:
    if not answer_text.strip():
        return {"next_q_idx": q_idx, "group_done": False, "followup_inserted": False}

    current_group["answers"].append(answer_text.strip())

    followup_clean = (
        followup if followup and not _is_duplicate(followup, questions) else None
    )
    can_insert = (
        followup_clean
        and current_group["followup_count"] < MAX_FOLLOWUPS
        and len(questions) < MAX_Q_PER_GROUP
    )
    if can_insert:
        questions.insert(q_idx + 1, followup_clean)
        current_group["followup_count"] += 1

    next_q_idx = q_idx + 1
    group_done = next_q_idx >= len(questions)

    return {
        "next_q_idx":        next_q_idx,
        "group_done":        group_done,
        "followup_inserted": bool(can_insert),
    }


# ============================================================
# 9. session_state 초기화
# ============================================================
def init_state():
    defaults = {
        "models_loaded":           False,
        "analysis_done":           False,
        "analysis_ready":          False,
        "color_result":            {},
        "yolo_result":             {},
        "image_base64":            None,
        "uploaded_img":            None,
        "child_age":               7,
        "child_sex":               "남자",
        "app_stage":               "upload",
        "conv_queue":              [],
        "current_group_idx":       0,
        "current_q_idx":           0,
        "current_question":        "",
        "report":                  None,
        "drawing_description":     "",
        "yolo_verification":       {},
        "description_submitted":   False,
        "stt_description_draft":   "",
        "desc_voice_confirm_pending": False,
        # ✅ 추가: JSON 불러오기 여부
        "report_loaded_from_json": False,
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
# 10. 배너
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
# 11. 타이틀 + 처음부터 하기 버튼
# ============================================================
col_title_main, col_reset_main = st.columns([5, 1])
with col_title_main:
    st.title("🌟 안녕? 나는 네 마음을 읽어주는 '마음친구'야!")
    st.write("네가 그린 멋진 그림을 보여주면, 내가 네 마음을 토닥토닥해줄게. 🥰")
with col_reset_main:
    st.write("")
    st.write("")
    if st.session_state["app_stage"] != "upload":
        if st.button("🔄 처음부터 하기", key="reset_top", width='stretch'):
            keep = {"models_loaded"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    del st.session_state[k]
            st.rerun()


# ============================================================
# 12. 팝업 다이얼로그
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
            if st.button("보물 상자 열어보기 💎", width='stretch'):
                st.session_state["analysis_done"] = True
                st.session_state["app_stage"]     = "describe"
                st.rerun()


# ============================================================
# 인식 텍스트 없이 진행 확인 다이얼로그 (그림 설명용)
# ============================================================
@st.dialog("🎤 음성 인식 결과가 없어요")
def show_desc_voice_confirm_dialog():
    st.warning(
        "아직 인식된 텍스트가 없어요!\n\n"
        "이대로 진행하면 **음성 설명 없이** 다음 단계로 넘어가요.\n"
        "정말 이대로 진행할까요?",
        icon="⚠️",
    )
    st.caption("💡 다시 녹음하려면 '취소'를 누른 뒤 🎙️ 버튼을 다시 눌러 시작해주세요.")
    col_cancel, col_ok = st.columns([1, 1])
    with col_cancel:
        if st.button("취소 — 다시 녹음할게요", width='stretch', key="desc_voice_confirm_cancel"):
            st.session_state["desc_voice_confirm_pending"] = False
            st.rerun()
    with col_ok:
        if st.button("이대로 진행할게요 ➤", width='stretch', key="desc_voice_confirm_ok"):
            st.session_state["desc_voice_confirm_pending"] = False
            st.session_state["drawing_description"]        = ""
            st.session_state["description_submitted"]      = True
            st.session_state["stt_description_draft"]      = ""
            st.session_state["app_stage"]                  = "result"
            st.rerun()

# ============================================================
# 현재 단계 인덱스
# ============================================================
current_stage_idx = stage_index(st.session_state["app_stage"])

# img_file 복원
img_file = st.session_state.get("uploaded_img", None)

# JSON 불러오기 여부
from_json = st.session_state.get("report_loaded_from_json", False)


# ============================================================
# STEP 1. 그림 업로드 & 정보 입력
# ============================================================
step1_expanded = (st.session_state["app_stage"] == "upload")
step1_label = (
    "1. 너에 대해 알려줘! 🎈"
    if st.session_state["app_stage"] == "upload"
    else "✅ 1단계 완료 — 그림 업로드"
)

with st.expander(step1_label, expanded=step1_expanded):

    # ── ✅ JSON 불러오기 UI (항상 STEP 1 안에 노출) ──────────────────
    with st.expander("📂 이전에 저장한 보고서 불러오기", expanded=False):
        st.caption("이전에 저장한 JSON 보고서 파일을 올리면 바로 보고서 화면으로 이동해요.")
        uploaded_json = st.file_uploader(
            "저장된 JSON 보고서 파일을 올려주세요",
            type=["json"],
            key="report_json_uploader",
        )
        if uploaded_json:
            try:
                loaded = json.loads(uploaded_json.read().decode("utf-8"))
                if "executive_summary" not in loaded:
                    st.error("올바른 보고서 JSON 파일이 아니에요.")
                else:
                    st.success("보고서를 읽었어요! 아래 버튼을 눌러 불러오세요.")
                    if st.button("이 보고서 불러오기 ✅", key="load_json_btn"):
                        st.session_state["report"]                  = loaded
                        st.session_state["report_errors"]           = loaded.get("report_errors", [])
                        st.session_state["report_loaded_from_json"] = True
                        st.session_state["analysis_done"]           = True
                        if desc := loaded.get("drawing_description"):
                            st.session_state["drawing_description"] = desc
                        if age := loaded.get("child_age"):
                            st.session_state["child_age"] = age
                        if sex := loaded.get("child_sex"):
                            st.session_state["child_sex"] = sex
                        st.session_state["app_stage"] = "done"
                        st.toast("✅ 보고서를 불러왔어요!")
                        st.rerun()
            except Exception as e:
                st.error(f"JSON 파싱 실패: {e}")

    st.divider()

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        st.subheader("🖼️ 네가 그린 멋진 그림을 보여줘")
        uploaded = st.file_uploader(
            "그림 파일을 여기에 쏙 넣어줘!", type=["jpg", "png", "jpeg"], key="img"
        )
        if uploaded:
            st.session_state["uploaded_img"] = uploaded
            img_file = uploaded
            st.image(Image.open(uploaded), width='stretch', caption="와! 정말 멋진 그림이야! ✨")

    with col_input2:
        st.subheader("🧒 너에 대해 알려줘!")
        child_age = st.slider("내 나이는 이만큼이야!", 5, 13,
                              st.session_state.get("child_age", 7))
        child_sex = st.radio("너는 남자니, 여자니?", ["남자", "여자"], horizontal=True,
                             index=0 if st.session_state.get("child_sex", "남자") == "남자" else 1)
        st.session_state["child_age"] = child_age
        st.session_state["child_sex"] = child_sex
        st.write("---")
        analyze_btn = st.button("마음친구야, 내 그림 좀 봐줄래? 🚀", width='stretch')

    if analyze_btn:
        if not img_file:
            st.warning("그림 파일을 먼저 올려줘! 🖼️")
        else:
            st.session_state["analysis_ready"]        = False
            st.session_state["analysis_done"]         = False
            st.session_state["description_submitted"] = False
            st.session_state["drawing_description"]   = ""
            st.session_state["stt_description_draft"] = ""
            show_analysis_popup(img_file)


# ============================================================
# STEP 2. 그림 설명 입력 (텍스트 + STT)
# ── JSON 불러오기 시에는 표시하지 않음
# ============================================================
if current_stage_idx >= stage_index("describe") and not from_json:
    step2_expanded = (st.session_state["app_stage"] == "describe")
    step2_label = (
        "2. 그림에 대해 조금 더 알려줘! 🖌️"
        if st.session_state["app_stage"] == "describe"
        else "✅ 2단계 완료 — 그림 설명"
    )

    with st.expander(step2_label, expanded=step2_expanded):
        st.write(
            "AI가 그림을 살펴봤어! "
            "그림에 대해 간단히 설명해주면 더 잘 이해할 수 있어. "
            "설명하기 어려우면 건너뛰어도 괜찮아! 😊"
        )

        col_desc1, col_desc2 = st.columns([1.2, 1])

        with col_desc1:
            if img_file:
                st.image(Image.open(img_file), width='stretch', caption="네가 그린 그림이야!")
            yolo_result = st.session_state.get("yolo_result", {})
            objects     = yolo_result.get("objects", [])
            if objects:
                st.caption("🤖 AI가 찾은 것들:")
                obj_names = ", ".join(set(o["name"] for o in objects))
                st.info(obj_names)

        with col_desc2:
            st.subheader("📝 그림 설명")

            if st.session_state["app_stage"] != "describe":
                saved_desc = st.session_state.get("drawing_description", "")
                if saved_desc:
                    st.info(saved_desc)
                else:
                    st.caption("(설명 없이 건너뛰었어요)")

            else:
                show_voice_guide()

                audio_desc = st.audio_input(
                    "아래 🎙️ 버튼을 누르면 녹음이 시작돼요! 말이 끝나면 빨간 🔴 버튼을 눌러 멈춰주세요.",
                    key="audio_description",
                )

                if audio_desc:
                    audio_bytes = audio_desc.getvalue()
                    current_audio_hash = hash(audio_bytes)

                    if st.session_state.get("last_audio_desc_hash") != current_audio_hash:
                        with st.spinner("네 목소리를 귀 기울여 듣고 있어... 👂"):
                            transcribed_desc = transcribe_audio(
                                audio_bytes,
                                language_code="ko-KR",
                                enable_auto_punctuation=True,
                                use_enhanced=True,
                            )
                        if transcribed_desc:
                            st.session_state["stt_desc_editable"]    = transcribed_desc
                            st.session_state["stt_description_draft"] = transcribed_desc
                        else:
                            st.session_state["stt_description_draft"] = ""
                            st.warning(
                                "목소리를 잘 못 들었어 😢\n"
                                "🎙️ 버튼을 다시 눌러서, 더 크고 천천히 말해볼래?",
                                icon=None,
                            )
                        st.session_state["last_audio_desc_hash"] = current_audio_hash

                st.caption("✏️ 녹음된 내용을 확인하거나, **직접 글로 설명을 적어주세요!**")

                edited_draft = st.text_area(
                    "🗣️ 그림 설명 (음성 인식 및 직접 입력)",
                    placeholder="녹음하면 여기에 인식된 내용이 나타나요. 마이크가 없다면 직접 타이핑해서 적어주셔도 돼요! ✏️",
                    height=120,
                    key="stt_desc_editable",
                    label_visibility="collapsed",
                )

                if st.session_state.get("stt_desc_editable"):
                    st.caption("다시 녹음하려면 🎙️ 버튼을 다시 눌러 시작한 후 빨간 🔴 버튼으로 멈추면 돼요.")

                st.write("")
                col_voice_skip, col_voice_next = st.columns([1, 1])

                with col_voice_next:
                    if st.button("설명 완료! ✅", key="voice_desc_next", width='stretch'):
                        current_draft = edited_draft.strip() if edited_draft else ""
                        if not current_draft:
                            st.session_state["desc_voice_confirm_pending"] = True
                            show_desc_voice_confirm_dialog()
                        else:
                            st.session_state["drawing_description"]   = current_draft
                            st.session_state["description_submitted"] = True
                            st.session_state["stt_description_draft"] = ""
                            with st.spinner("설명을 바탕으로 그림을 다시 확인하고 있어요... 🔍"):
                                verification = verify_yolo_with_description(
                                    yolo_result=st.session_state["yolo_result"],
                                    drawing_description=current_draft,
                                    image_base64=st.session_state.get("image_base64"),
                                )
                                st.session_state["yolo_verification"] = verification
                                updated_yolo = apply_verification_to_yolo(
                                    st.session_state["yolo_result"], verification
                                )
                                st.session_state["yolo_result"] = updated_yolo
                            st.session_state["app_stage"] = "result"
                            st.rerun()

                with col_voice_skip:
                    if st.button("건너뛰기 ⏭️", key="voice_desc_skip", width='stretch'):
                        st.session_state["drawing_description"]   = ""
                        st.session_state["description_submitted"] = True
                        st.session_state["stt_description_draft"] = ""
                        st.session_state["app_stage"]             = "result"
                        st.rerun()

                with st.expander("💡 어떻게 설명하면 좋을까요?", expanded=False):
                    st.write("**그림 속 인물에 대해:** 누구를 그렸나요? 어떤 표정인가요?")
                    st.write("**집/나무에 대해:** 어떤 집/나무인가요? 계절이나 날씨는요?")
                    st.write("**전체적인 이야기:** 이 그림에서 어떤 일이 일어나고 있나요?")
                    st.write("**AI가 잘못 읽은 것:** AI가 엉뚱하게 인식한 것이 있다면 알려주세요!")


# ============================================================
# STEP 3. 분석 결과
# ── JSON 불러오기 시에는 표시하지 않음
# ============================================================
if current_stage_idx >= stage_index("result") and st.session_state.get("analysis_done") and not from_json:
    step3_expanded = (st.session_state["app_stage"] == "result")
    step3_label = (
        "3. 네 마음속에 이런 보물이 들어있구나! 💎"
        if st.session_state["app_stage"] == "result"
        else "✅ 3단계 완료 — 분석 결과"
    )

    with st.expander(step3_label, expanded=step3_expanded):
        color_result = st.session_state["color_result"]
        yolo_result  = st.session_state["yolo_result"]
        verification = st.session_state.get("yolo_verification", {})

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("🎨 마음 지도로 그려본 너의 그림")
            plotted = yolo_result.get("plotted_image")
            if plotted is not None:
                st.image(plotted, caption="마음친구가 찾아낸 보물들이야!", width='stretch')
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
                df_rows = []
                for o in objects:
                    row = {
                        "찾은 것": o["name"],
                        "위치":    o["position_label"],
                        "크기":    f"{o['area_pct']}%",
                        "신뢰도":  f"{o['confidence']:.0%}",
                        "상태":    "⚠️ 오탐 의심" if o.get("is_suspicious") else (
                                   "📝 설명 기반" if o.get("from_description") else "✅ 정상"),
                    }
                    df_rows.append(row)
                df = pd.DataFrame(df_rows)
                st.dataframe(
                    df,
                    width='stretch',
                    height=min(len(df_rows) * 35 + 38, 300),
                )

                if verification and (verification.get("suspicious_objects") or verification.get("description_extras")):
                    with st.expander("🔍 AI 탐지 검증 결과", expanded=True):
                        suspicious = verification.get("suspicious_objects", [])
                        extras     = verification.get("description_extras", [])
                        notes      = verification.get("notes", "")
                        if suspicious:
                            st.warning(f"⚠️ 오탐 의심 항목 (대화에서 제외됨): **{', '.join(suspicious)}**")
                        if extras:
                            st.info(f"📝 설명에서 추가 확인된 항목: **{', '.join(extras)}**")
                        if notes:
                            st.caption(f"검증 메모: {notes}")

                with st.expander("💡 위치 정보로 보는 아이의 마음", expanded=True):
                    st.write("- **가로 위치**: 왼쪽은 과거/내향성, 오른쪽은 미래/외향성을 의미하기도 해요.")
                    st.write("- **세로 위치**: 상단은 이상/상상력, 하단은 현실 감각을 나타내기도 해요.")
                    st.write("- **크기**: 면적이 클수록 아이에게 중요한 대상일 수 있어요.")
            else:
                st.info("감지된 객체가 없어요. 그림을 다시 확인해줘!")

            drawing_desc = st.session_state.get("drawing_description", "")
            if drawing_desc:
                with st.expander("📖 입력된 그림 설명", expanded=False):
                    st.write(drawing_desc)

        if st.session_state["app_stage"] == "result":
            st.divider()
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                start_chat = st.button("🗣️ 이제 그림에 대해 이야기해보자!", width='stretch')
                retry_flag = st.session_state.pop("retry_qgen_trigger", False)

                if start_chat or retry_flag:
                    st.session_state.pop("q_gen_error", None)
                    with st.spinner("마음친구가 질문을 준비하고 있어요... 💭"):
                        queue               = build_conversation_queue(yolo_result)
                        all_asked: list[str] = []
                        failed              = False
                        drawing_description = st.session_state.get("drawing_description", "")

                        for item in queue:
                            qs = generate_questions_for_group(
                                item,
                                color_result["llm_summary"],
                                st.session_state["child_age"],
                                drawing_description=drawing_description,
                                image_base64=st.session_state.get("image_base64"),
                                existing_questions=all_asked if all_asked else None,
                            )
                            if qs is None:
                                failed = True
                                break

                            if qs == []:
                                item["questions"] = []
                                item["answers"]   = []
                                item["covered_by_description"] = True
                                item["summary"] = (
                                    f"그림 설명에서 충분히 파악됨: {drawing_description[:100]}..."
                                    if len(drawing_description) > 100
                                    else f"그림 설명에서 충분히 파악됨: {drawing_description}"
                                )
                            else:
                                item["questions"] = qs
                                item["covered_by_description"] = False
                                all_asked.extend(qs)

                        if not failed:
                            st.session_state["conv_queue"]        = queue
                            st.session_state["current_group_idx"] = 0
                            st.session_state["current_q_idx"]     = 0
                            st.session_state["app_stage"]         = "chatting"
                            st.rerun()

                if err := st.session_state.get("q_gen_error"):
                    st.error("질문을 만드는 데 실패했어요 😢")
                    with st.expander("🛠️ 에러 상세 보기", expanded=True):
                        st.code(err, language="text")
                    if st.button("🔄 질문 다시 만들기", key="retry_qgen"):
                        st.session_state.pop("q_gen_error", None)
                        st.session_state["conv_queue"]         = []
                        st.session_state["retry_qgen_trigger"] = True
                        st.rerun()


# ============================================================
# STEP 4. 대화
# ── JSON 불러오기 시에는 표시하지 않음
# ============================================================
if current_stage_idx >= stage_index("chatting") and not from_json:
    step4_expanded = (st.session_state["app_stage"] == "chatting")
    step4_label = (
        "4. 우리 같이 도란도란 이야기하자 💬"
        if st.session_state["app_stage"] == "chatting"
        else "✅ 4단계 완료 — 대화"
    )

    with st.expander(step4_label, expanded=step4_expanded):
        if st.session_state["app_stage"] != "chatting":
            conv_queue = st.session_state.get("conv_queue", [])
            for item in conv_queue:
                st.markdown(f"**{item['group']}**")
                if item.get("covered_by_description"):
                    st.caption("📝 그림 설명으로 파악되어 별도 대화 없음")
                else:
                    for q, a in zip(item.get("questions", []), item.get("answers", [])):
                        st.markdown(f"- **Q:** {q}")
                        st.markdown(f"  **A:** {a}")
                st.divider()
        else:
            queue = st.session_state["conv_queue"]
            g_idx = st.session_state["current_group_idx"]

            while g_idx < len(queue) and queue[g_idx].get("covered_by_description", False):
                g_idx += 1
                st.session_state["current_group_idx"] = g_idx
                st.session_state["current_q_idx"]     = 0

            q_idx = st.session_state["current_q_idx"]

            if g_idx >= len(queue):
                all_covered = all(q.get("covered_by_description", False) for q in queue)

                if all_covered:
                    st.success("그림 설명만으로 모든 내용을 충분히 파악했어! 대단해! 🎉")
                    skipped_groups = ", ".join(q["group"] for q in queue)
                    st.info(
                        f"📝 **{skipped_groups}** 모두 그림 설명으로 파악됐어.\n\n"
                        "추가로 이야기하지 않아도 마음친구가 보고서를 잘 써줄 수 있어! 💪",
                    )
                else:
                    st.success("모든 이야기를 다 들었어! 정말 잘했어 😊")

                st.write("")
                col_l, col_c, col_r = st.columns([1, 2, 1])
                with col_c:
                    if st.button("📋 보고서 보러 가볼까? 💌", width='stretch', key="go_to_report"):
                        st.session_state["app_stage"] = "reporting"
                        st.rerun()

            else:
                current_group = queue[g_idx]
                total_groups  = len([q for q in queue if not q.get("covered_by_description", False)])
                done_groups   = len([q for q in queue[:g_idx] if not q.get("covered_by_description", False)])

                if "followup_count" not in current_group:
                    current_group["followup_count"] = 0

                st.progress(
                    done_groups / max(total_groups, 1),
                    text=f"{current_group['group']} 이야기 중 ({done_groups + 1}/{total_groups})",
                )

                skipped = [q["group"] for q in queue if q.get("covered_by_description", False)]
                if skipped:
                    st.info(
                        f"💡 그림 설명으로 이미 파악된 주제는 건너뛰었어: **{', '.join(skipped)}**",
                        icon="📝",
                    )

                with st.sidebar:
                    st.subheader("🖼️ 내 그림")
                    if img_file:
                        st.image(Image.open(img_file), width='stretch')

                    drawing_desc = st.session_state.get("drawing_description", "")
                    if drawing_desc:
                        with st.expander("📖 내가 설명한 그림", expanded=False):
                            st.caption(drawing_desc)

                    if g_idx > 0:
                        st.subheader("📖 지금까지 이야기")
                        for past in queue[:g_idx]:
                            if past.get("covered_by_description"):
                                with st.expander(f"{past['group']} (설명으로 파악)", expanded=False):
                                    st.caption("그림 설명으로 충분히 파악되었어요.")
                            else:
                                with st.expander(past["group"], expanded=False):
                                    st.caption(past.get("summary", "요약 없음"))

                questions = current_group["questions"]
                current_q = questions[q_idx] if q_idx < len(questions) else None

                st.markdown(f"### 🎨 {current_group['group']}에 대해 이야기해요")
                if current_q:
                    st.info(f"💬 {current_q}")

                if reaction_msg := st.session_state.get("last_reaction"):
                    st.success(f"마음친구: {reaction_msg}")

                def _handle_answer(answer_text: str):
                    reaction, followup = get_reaction_and_followup(
                        current_group["group"], current_q, answer_text.strip(),
                        existing_questions=current_group["questions"],
                    )
                    st.session_state["last_reaction"] = reaction

                    result = process_answer(
                        answer_text, current_group, questions, q_idx, reaction, followup
                    )

                    if result["group_done"]:
                        with st.spinner(f"{current_group['group']} 이야기를 정리하는 중..."):
                            current_group["summary"] = summarize_group_dialogue(current_group)
                        st.session_state["current_group_idx"] += 1
                        st.session_state["current_q_idx"]     = 0
                        st.session_state.pop("last_reaction", None)
                    else:
                        st.session_state["current_q_idx"] = result["next_q_idx"]

                    st.rerun()

                show_voice_guide()

                stt_key      = f"stt_result_{g_idx}_{q_idx}"
                editable_key = f"stt_editable_{g_idx}_{q_idx}"

                audio_input = st.audio_input(
                    "아래 🎙️ 버튼을 누르면 녹음이 시작돼요! 말이 끝나면 빨간 🔴 버튼을 눌러 멈춰주세요.",
                    key=f"audio_{g_idx}_{q_idx}",
                )

                if audio_input:
                    audio_bytes        = audio_input.getvalue()
                    current_audio_hash = hash(audio_bytes)

                    if st.session_state.get(f"last_audio_hash_{g_idx}_{q_idx}") != current_audio_hash:
                        with st.spinner("네 목소리를 귀 기울여 듣고 있어... 👂"):
                            transcribed = transcribe_audio(
                                audio_bytes,
                                language_code="ko-KR",
                                enable_auto_punctuation=True,
                                use_enhanced=True,
                            )
                        if transcribed:
                            st.session_state[stt_key]      = transcribed
                            st.session_state[editable_key] = transcribed
                        else:
                            st.session_state.pop(stt_key, None)
                            st.warning(
                                "목소리를 잘 못 들었어 😢\n"
                                "🎙️ 버튼을 다시 눌러서, 더 크고 천천히 말해볼래?",
                                icon=None,
                            )
                        st.session_state[f"last_audio_hash_{g_idx}_{q_idx}"] = current_audio_hash

                st.caption("✏️ 녹음된 내용을 확인하거나, **직접 글로 답변을 적어주세요!**")

                edited_recognized = st.text_area(
                    "🗣️ 답변 (음성 인식 및 직접 입력)",
                    placeholder="녹음하면 여기에 인식된 내용이 나타나요. 마이크가 없다면 직접 타이핑해서 적어주셔도 돼요! ✏️",
                    height=100,
                    key=editable_key,
                    label_visibility="collapsed",
                )

                st.write("")
                has_answer = bool(edited_recognized and edited_recognized.strip())
                col_voice_ok, col_voice_skip = st.columns([2, 1])
                with col_voice_ok:
                    if st.button(
                        "이 내용으로 답하기 ✅",
                        key=f"stt_btn_ok_{g_idx}_{q_idx}",
                        width='stretch',
                        disabled=not has_answer,
                    ):
                        answer_to_submit = edited_recognized.strip()
                        st.session_state.pop(stt_key, None)
                        with st.spinner("생각 중... 💭"):
                            _handle_answer(answer_to_submit)
                with col_voice_skip:
                    if st.button(
                        "이 질문 건너뛰기 ⏭️",
                        key=f"stt_skip_{g_idx}_{q_idx}",
                        width='stretch',
                    ):
                        st.session_state.pop(stt_key, None)
                        next_idx = q_idx + 1
                        if next_idx >= len(questions):
                            with st.spinner(f"{current_group['group']} 이야기를 정리하는 중..."):
                                current_group["summary"] = summarize_group_dialogue(current_group)
                            st.session_state["current_group_idx"] += 1
                            st.session_state["current_q_idx"]     = 0
                            st.session_state.pop("last_reaction", None)
                        else:
                            st.session_state["current_q_idx"] = next_idx
                        st.rerun()
                if not has_answer:
                    st.caption("💡 녹음하거나 직접 입력한 뒤 답하기 버튼을 눌러주세요.")

                past_qa = list(zip(questions[:q_idx], current_group["answers"]))
                if past_qa:
                    with st.expander("📜 이 주제에서 나눈 이야기", expanded=False):
                        for past_q, past_a in past_qa:
                            st.markdown(f"**마음친구:** {past_q}")
                            st.markdown(f"**나:** {past_a}")
                            st.divider()

                with st.expander("🛠️ 디버그 정보", expanded=False):
                    st.caption(
                        f"그룹: {current_group['group']} | "
                        f"질문 {q_idx + 1}/{len(questions)} | "
                        f"후속질문: {current_group.get('followup_count', 0)}/{MAX_FOLLOWUPS}"
                    )

# ============================================================
# STEP 5. 보고서 생성 & 표시
# ============================================================
if current_stage_idx >= stage_index("reporting"):

    # 보고서 생성 중
    if st.session_state["app_stage"] == "reporting" and st.session_state["report"] is None:
        st.header("5. 네 마음 보고서를 만들고 있어! 📋")
        with st.spinner("감정 분석 → 탐색 질문 → 종합 보고서 순서로 작성 중이에요... ✍️ (2~3분 소요)"):
            report = generate_report(
                color_result        = st.session_state["color_result"],
                yolo_result         = st.session_state["yolo_result"],
                conversation_queue  = st.session_state["conv_queue"],
                child_age           = st.session_state["child_age"],
                child_sex           = st.session_state["child_sex"],
                drawing_description = st.session_state.get("drawing_description", ""),
            )
            st.session_state["report"]        = report
            st.session_state["report_errors"] = report.get("report_errors", [])
            st.session_state["app_stage"]     = "done"
        st.rerun()

    # 보고서 표시
    if st.session_state["app_stage"] == "done" and st.session_state["report"]:
        step5_label = "5. 🌈 마음친구가 써준 보고서야!"
        with st.expander(step5_label, expanded=True):
            report = st.session_state["report"]

            # JSON 불러오기 안내
            if from_json:
                st.info("📂 저장된 JSON 보고서를 불러왔어요.", icon="📂")

            col_title, col_regen_top = st.columns([5, 1])
            with col_title:
                st.caption("⚠️ 이 보고서는 교육 목적의 참고 자료이며, 의료적 진단이나 치료를 대체하지 않습니다.")
            with col_regen_top:
                if st.button("🔁 재생성", key="regen_report_top", width='stretch'):
                    with st.spinner("보고서를 다시 작성하고 있어요... ✍️ (2~3분 소요)"):
                        regen_full_report()
                    st.rerun()

            report_errors  = st.session_state.get("report_errors", [])
            global_errors  = [e for e in report_errors if "section" not in e]
            refusal_errors = [e for e in global_errors if "거절 응답" in e.get("step", "")]
            other_errors   = [e for e in global_errors if "거절 응답" not in e.get("step", "")]

            if refusal_errors:
                st.error(
                    "⚠️ 보고서 일부 항목에서 AI가 응답을 거부했어요.  \n"
                    "위의 **🔁 재생성** 버튼으로 다시 생성해주세요.",
                    icon="🚨",
                )
                for e in refusal_errors:
                    st.caption(f"문제 섹션: {e['reason']}")
                st.divider()

            if other_errors:
                with st.expander(
                    f"⚠️ 보고서 생성 중 {len(other_errors)}건의 문제가 발생했어요",
                    expanded=True,
                ):
                    for i, err in enumerate(other_errors, 1):
                        st.markdown(f"**{i}. [{err['step']}]** {err['reason']}")
                        if raw_snippet := err.get("raw"):
                            with st.expander("LLM 원본 응답 보기"):
                                st.code(raw_snippet, language="text")
                st.divider()

            drawing_desc = st.session_state.get("drawing_description", "")
            if drawing_desc:
                with st.expander("📖 분석에 사용된 그림 설명", expanded=False):
                    st.write(drawing_desc)

            st.subheader("📌 전체 요약 및 핵심 인사이트")
            if summary := report.get("executive_summary"):
                st.write(summary)
            else:
                st.caption("전체 요약 내용이 없어요.")

            if findings := report.get("key_findings"):
                st.subheader("🔍 주요 발견 사항")
                for f in findings:
                    st.markdown(f"- {f}")

            st.divider()

            emotion = report.get("emotion_analysis", {})
            if emotion:
                st.subheader("💜 감정 언어 분석")
                col_e1, col_e2 = st.columns([1, 1])
                with col_e1:
                    if emo_list := emotion.get("dominant_emotions"):
                        st.markdown("**주요 감정**")
                        badges_html = " ".join(
                            f'<span style="display:inline-block;border:1px solid #ccc;'
                            f'border-radius:20px;padding:4px 12px;margin:3px;">{e}</span>'
                            for e in emo_list
                        )
                        st.markdown(badges_html, unsafe_allow_html=True)
                    if intensity := emotion.get("intensity_level"):
                        st.markdown(f"**감정 강도:** `{intensity}`")
                with col_e2:
                    if tone := emotion.get("emotional_tone"):
                        st.markdown("**감정적 톤**")
                        st.write(tone)
                if symbols := emotion.get("symbolic_elements"):
                    st.markdown("**상징적 요소**")
                    st.write(" · ".join(symbols))

            st.divider()

            col_u1, col_u2 = st.columns([1, 1])
            with col_u1:
                st.subheader("🧭 활용 방향성")
                if usage := report.get("usage_direction"):
                    st.write(usage)
                else:
                    st.caption("활용 방향성 내용이 없어요.")
            with col_u2:
                st.subheader("🎯 집중 탐색 영역")
                if focus := report.get("focus_areas"):
                    for area in focus:
                        st.markdown(f"- {area}")
                else:
                    st.caption("집중 탐색 영역 내용이 없어요.")

            st.divider()

            st.subheader("🤖 AI 시스템 종합 분석")
            if comprehensive := report.get("comprehensive_analysis"):
                st.write(comprehensive)
            else:
                st.caption("종합 분석 내용이 없어요.")

            st.divider()

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.subheader("🎨 색채 관찰")
                if ca := report.get("color_analysis"):
                    st.write(ca)
                else:
                    st.caption("색채 관찰 내용이 없어요.")
            with col_r2:
                st.subheader("🏠🌳👤 그림 구성 관찰")
                if oa := report.get("object_analysis"):
                    st.write(oa)
                else:
                    st.caption("그림 구성 관찰 내용이 없어요.")
            with col_r3:
                st.subheader("💬 표현 특징")
                if di := report.get("dialogue_insights"):
                    st.write(di)
                else:
                    st.caption("표현 특징 내용이 없어요.")

            st.divider()

            exploration = report.get("exploration_questions", {})
            if exploration and exploration.get("questions"):
                st.subheader("❓ 자기 탐색 질문")
                if purpose := exploration.get("purpose"):
                    st.caption(f"목적: {purpose}")
                categories = exploration.get("categories", [])
                questions  = exploration.get("questions", [])
                for idx, q in enumerate(questions):
                    cat_label = categories[idx] if idx < len(categories) else ""
                    tag = f"`{cat_label}` " if cat_label else ""
                    st.markdown(f"{tag}**Q{idx+1}.** {q}")

            st.divider()

            if recs := report.get("recommendations"):
                st.subheader("💡 선생님·부모님께 드리는 참고 사항")
                for r in recs:
                    st.markdown(f"- {r}")

            st.divider()

            st.subheader("📚 참고 논문")
            faiss_meta = report.get("faiss_meta", [])
            refs       = report.get("references_used", "")

            if faiss_meta:
                with st.expander("🗂️ FAISS에서 검색된 원본 논문 목록", expanded=False):
                    for i, m in enumerate(faiss_meta, 1):
                        label = m.get("title") or m.get("source", "알 수 없음")
                        page  = m.get("page", "?")
                        st.markdown(f"**{i}.** `{label}` — p.{page}")

            if refs:
                with st.expander("📄 참고한 논문 목록 (출처·링크 포함)", expanded=True):
                    for line in refs.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        url_match = re.search(r"(https?://\S+|10\.\d{4,}/\S+)", line)
                        if url_match:
                            url  = url_match.group(1)
                            href = url if url.startswith("http") else f"https://doi.org/{url}"
                            st.markdown(f"- {line.replace(url, f'[{url}]({href})')}")
                        else:
                            st.markdown(f"- {line}")
            else:
                st.caption("참고 논문 정보가 없어요. 🔁 재생성 버튼을 눌러보세요.")

            st.divider()

            with st.expander("📖 전체 대화 기록 보기"):
                conv_queue_for_display = st.session_state.get("conv_queue", [])
                if conv_queue_for_display:
                    for item in conv_queue_for_display:
                        st.markdown(f"**{item['group']}**")
                        if item.get("covered_by_description"):
                            st.caption("📝 그림 설명으로 충분히 파악되어 별도 대화를 진행하지 않았습니다.")
                        else:
                            for q, a in zip(item["questions"], item["answers"]):
                                st.markdown(f"- **Q:** {q}")
                                st.markdown(f"  **A:** {a}")
                        st.divider()
                else:
                    st.caption("대화 기록이 없어요. (JSON으로 불러온 보고서)")

            # ✅ child_age, child_sex 포함해서 저장
            report_for_download = {
                k: v for k, v in report.items()
                if k not in ("report_errors", "faiss_meta")
            }
            report_for_download["drawing_description"] = st.session_state.get("drawing_description", "")
            report_for_download["child_age"]           = st.session_state.get("child_age", 7)
            report_for_download["child_sex"]           = st.session_state.get("child_sex", "남자")

            st.download_button(
                label="📄 보고서 JSON 다운로드",
                data=json.dumps(report_for_download, ensure_ascii=False, indent=2),
                file_name="htp_report.json",
                mime="application/json",
            )

# ============================================================
# 하단 스크롤 여백 확보 (잘림 현상 방지)
# ============================================================
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)