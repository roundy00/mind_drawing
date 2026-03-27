import streamlit as st
from PIL import Image
import time
import random
import base64
import pandas as pd
from ultralytics import YOLO
from streamlit_lottie import st_lottie
import requests
import json
import os
from colorthief import ColorThief
import io


# 1. 페이지 설정
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

# ======================================================================
st.markdown("""
    <style>
    /* 전체 배경색을 따뜻한 미색으로 */
    .stApp {
        background-color: #FFFFFA;
    }
    /* 버튼을 둥글고 예쁜 핑크색으로 */
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
        transform: scale(1.05); /* 마우스 올리면 살짝 커지는 효과 */
    }
    /* 카드 느낌의 박스 디자인 */
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
# ======================================================================
# --- 애니메이션 로드 함수 (로컬 전용) ---
def load_lottiefile(filepath: str):
    if not os.path.exists(filepath):
        return None  # 파일이 없어도 에러 대신 None 반환
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# --- 파일 로드 (파일명은 실제 저장하신 이름으로 맞춰주세요) ---
lottie_hello = load_lottiefile("Bath3_Hi emote.json")    # 인사하는 캐릭터
lottie_loading = load_lottiefile("animal.json") # 분석 중 캐릭터
lottie_success = load_lottiefile("Confetti.json") # 축하 효과

# --- 색채 분석 함수 ---
def analyze_colors(image_file):
    # PIL 이미지를 ColorThief가 읽을 수 있는 바이트 형태로 변환
    img_bytes = io.BytesIO(image_file.getvalue())
    color_thief = ColorThief(img_bytes)
    
    # 1. 가장 지배적인 색상 하나 추출
    dominant_color = color_thief.get_color(quality=1) # (R, G, B)
    
    # 2. 주요 색상 팔레트 추출 (5개)
    palette = color_thief.get_palette(color_count=5)
    
    return dominant_color, palette

# --- 팝업창 정의 ---
@st.dialog("🎨 마음친구가 분석 중이에요!")
def show_analysis_popup(img_file):
    st.write("그림 속 이야기를 찾고 있어요. 잠시만 기다려줘! ✨")
    
    # 로딩 애니메이션 (파일 경로 확인 필요)
    if lottie_loading:
        st_lottie(lottie_loading, height=200, key="popup_loading_ani")

    # [핵심] 팝업 안에서 분석을 수행합니다.
    if not st.session_state.get('yolo_done', False):
        with st.spinner("마음친구가 집중하고 있어요... 🤫"):
            try:
                # 1. 색채 분석
                dom_color, palette = analyze_colors(img_file)
                st.session_state['dominant_color'] = dom_color
                st.session_state['palette'] = palette
                
                # 2. YOLO 분석
                image = Image.open(img_file)
                model = YOLO('best.pt') 
                results = model.predict(source=image, save=False, conf=0.25)
                
                # 결과 데이터 정리
                extracted_data = []
                found_items = []
                friendly_names = {
                    # 나무 관련 (통합)
                    "나무전체": 0, "나무": 0,
                    "기둥": 1,
                    "수관": 2,
                    "가지": 3,
                    "뿌리": 4,
                    "나뭇잎": 5,
                    "꽃": 6, # '꽃' 라벨 통합
                    "열매": 7,
                    "그네": 8,
                    "새": 9,
                    "다람쥐": 10,
                    "구름": 11,
                    "달": 12,
                    "별": 13,
                
                    # 사람 관련 (남자/여자 공통 및 구분)
                    "사람전체": 14,
                    "머리": 15,
                    "얼굴": 16,
                    "눈": 17,
                    "코": 18,
                    "입": 19,
                    "귀": 20,
                    "남자머리카락": 21,
                    "목": 22,
                    "상체": 23,
                    "팔": 24,
                    "손": 25,
                    "다리": 26,
                    "발": 27,
                    "단추": 28,
                    "주머니": 29,
                    "운동화": 30,
                    "남자구두": 31,
                    "여자머리카락": 32,
                    "여자구두": 33,
                
                    # 집 관련 및 배경
                    "집전체": 34,
                    "지붕": 35,
                    "집벽": 36,
                    "문": 37,
                    "창문": 38,
                    "굴뚝": 39,
                    "연기": 40,
                    "울타리": 41,
                    "길": 42,
                    "연못": 43,
                    "산": 44,
                    "잔디": 45,
                    "태양": 46
                }
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        display_name = friendly_names.get(cls_id, r.names[cls_id])
                        found_items.append(display_name)
                        extracted_data.append({"찾은 것": display_name, "크기": "소중한 조각", "느낌": "정성 가득 ✨"})
                
                # 세션 상태 저장
                st.session_state['extracted_data'] = extracted_data
                st.session_state['found_items'] = found_items
                st.session_state['res_plotted'] = results[0].plot()
                st.session_state['yolo_done'] = True # 분석 완료 깃발
                
                # 분석 완료 후 UI를 즉시 업데이트하기 위해 내부 리런
                st.rerun() 
                
            except Exception as e:
                st.error(f"분석 중 에러: {e}")

    # 분석 완료 시 연출 (풍선 및 성공 메시지)
    is_ready = st.session_state.get('yolo_done', False)
    if is_ready:
        st.success("분석이 완료되었어! 보물을 확인할 준비가 됐니?")
        st.balloons() # 완료 의미로 풍선 띄우기
        # if lottie_success:
        #     st_lottie(lottie_success, speed=1, loop=False, height=150, key="popup_success_ani")

    st.write("---")
    # 버튼 배치
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # yolo_done이 True일 때만 버튼이 활성화(disabled=False)됩니다.
        if st.button("보물 상자 열어보기 💎", use_container_width=True, disabled=not is_ready):
            st.session_state['analysis_done'] = True
            st.rerun() # 이제 팝업을 닫고 결과 페이지(st.header 2번 파트)를 보여줍니다.
# ==============================================================================
# CSS 주입
st.markdown("""
    <style>
    /* 1. 웹폰트 불러오기 */
    @font-face {
        font-family: 'OngleipParkDahyeon';
        src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2411-3@1.0/Ownglyph_ParkDaHyun.woff2') format('woff2');
        font-weight: normal;
        font-display: swap;
    }

    /* 2. 전체 요소에 폰트 적용 */
    html, body, [class*="st-"] {
        font-family: 'OngleipParkDahyeon', sans-serif !important;
        font-size: 18px; /* 손글씨체는 약간 크게 설정하는 것이 가독성에 좋습니다 */
    }

    /* 3. 제목 스타일 강조 */
    h1, h2, h3 {
        font-family: 'OngleipParkDahyeon', sans-serif !important;
        color: #5D5D5D;
        margin-bottom: 1rem;
    }

    /* 기존 버튼 및 배경 스타일 유지 */
    .stApp {
        background-color: #FFF9F0;
    }
    div.stButton > button:first-child {
        background-color: #FFB7B2;
        color: white;
        border-radius: 30px;
        border: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
# =========================================================================

# 상단에 환영 인사와 함께 손 흔드는 캐릭터 배치
# st_lottie(lottie_hello, speed=1, loop=True, quality="low", height=200, key="hello")

# 배너이미지 추가 (Base64 변환)
def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    img_path = "main_concept_image.webp"
    img_base64 = get_image_base64(img_path)
    st.markdown(
        f'<img src="data:image/png;base64,{img_base64}" alt="무지개와 유니콘이 그려진 예쁜 배너" style="width:100%; border-radius: 15px;">',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.info("✨ 예쁜 그림 친구가 곧 도착할 거예요!")

# 제목을 다정하게 변경
st.title("🌟 안녕? 나는 네 마음을 읽어주는 '마음친구'야!")
st.write("네가 그린 멋진 그림과 예쁜 목소리를 들려주면, 내가 네 마음을 토닥토닥해줄게. 🥰")

# 2. 내부 처리 (숨김)
if 'models_loaded' not in st.session_state:
    with st.spinner("AI 친구가 기지개를 켜며 준비하고 있어요... 잠시만 기다려줘! 💤"):
        time.sleep(1.5)
        st.session_state['models_loaded'] = True
        st.session_state['analysis_done'] = False
        st.session_state['chat_history'] = []

# 3. 데이터 입력 섹션
st.header("1. 너에 대해 알려줘! 🎈")
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("🖼️ 네가 그린 멋진 그림을 보여줘")
    img_file = st.file_uploader("그림 파일을 여기에 쏙 넣어줘!", type=["jpg", "png", "jpeg"], key="img")
    if img_file:
        image = Image.open(img_file)
        st.image(image, use_container_width=True, caption="와! 정말 멋진 그림이야! ✨")

with col_input2:
    st.subheader("🧒 너에 대해 알려줘!")
    child_age = st.slider("내 나이는 이만큼이야!", 5, 13, 7)
    child_sex = st.radio("너는 남자니, 여자니?", ["남자", "여자"], horizontal=True)
    
    st.write("---")
    analyze_btn = st.button("마음친구야, 내 그림 좀 봐줄래? 🚀", use_container_width=True)

# 4. 분석 결과 및 대화 섹션
if analyze_btn and img_file:
    # 버튼 클릭 시 팝업만 딱 띄웁니다.
    # 분석 상태가 아니라면 초기화
    if not st.session_state.get('yolo_done', False):
        st.session_state['yolo_done'] = False
    show_analysis_popup(img_file)

if st.session_state.get('analysis_done') and st.session_state.get('extracted_data'):
    extracted_data = st.session_state['extracted_data'] # 세션에서 데이터를 가져옴
    found_items = st.session_state.get('found_items', [])
    res_plotted = st.session_state.get('res_plotted')


st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")

col1, col2 = st.columns(2)
    
with col1:
    st.subheader("🎨 마음 지도로 그려본 너의 그림")
    if res_plotted is not None:
        st.image(res_plotted, caption="마음친구가 찾아낸 보물들이야!", use_container_width=True)
        
with col2:
    st.subheader("🔍 찾아낸 마음 조각들")
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        st.table(df)
    else:
        st.info("아직 분석된 결과가 없어요.")
    
st.success(diagnosis_msg)

# 5. 대화 세션
if st.session_state.get('analysis_done'):
    st.divider()
    st.header("3. 우리 같이 도란도란 이야기하자 💬")
    
    st.subheader("🎙️ 네 목소리를 들려줘!")
    st.write("그림에 대해 하고 싶은 말이 있다면 버튼을 누르고 편하게 말해봐!")
    
    audio_file = st.audio_input("여기를 눌러서 목소리를 들려줘! 🎤")
    
    if audio_file:
        with st.spinner("네 목소리를 귀 기울여 듣고 있어...👂"):
            time.sleep(1)
            st.session_state['chat_history'].append({"role": "assistant", "content": "와! 목소리가 정말 예쁘구나. 방금 말해준 고민은 나랑 같이 차근차근 해결해보자. 너는 정말 용기 있는 아이야! ❤️"})

    st.subheader("💬 마음친구랑 채팅하기")
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("하고 싶은 말을 여기에 적어줘! (선생님이나 부모님이 도와주셔도 돼!)"):
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("생각 중...💭"):
            time.sleep(1)
            llm_response = f"'{prompt}'라고 말해줘서 고마워! 네 마음이 어떤지 조금 더 알 것 같아. 우리 다음엔 또 어떤 즐거운 이야기를 해볼까?"
            st.session_state['chat_history'].append({"role": "assistant", "content": llm_response})
            with st.chat_message("assistant"):
                st.markdown(llm_response)
