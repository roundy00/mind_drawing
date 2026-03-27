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
        background-color: #E8E5E3;
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

# --- 팝업(Dialog) 내부에서 활용 ---
@st.dialog("마음친구가 색깔 꾸러미를 분석하고 있어요! 🎨")
def show_analysis_popup(img_file):
    st.write("아이가 어떤 색으로 마음을 표현했는지 살펴보고 있어요.")
    
    # 애니메이션 표시
    if lottie_loading:
        st_lottie(lottie_loading, height=200, key="color_loading")
        
    # 색채 및 YOLO 분석 수행
    dom_color, palette = analyze_colors(img_file)
    
    # 결과 보여주기
    st.subheader("🖼️ 이 그림의 주요 색상들이에요")
    
    # 컬러 팔레트를 가로로 예쁘게 보여주기
    cols = st.columns(5)
    for i, color in enumerate(palette):
        cols[i].markdown(
            f'<div style="background-color: rgb{color}; height: 50px; border-radius: 10px;"></div>',
            unsafe_allow_html=True
        )
        cols[i].caption(f"RGB{color}")

    if st.button("분석 완료! 결과 보러 가기"):
        st.session_state['dominant_color'] = dom_color
        st.session_state['palette'] = palette
        st.rerun()
# ==============================================================================
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
    # 팝업 실행!
    show_analysis_popup()
    st.session_state['analysis_done'] = True
    st.divider()

    # 분석 중 애니메이션
    with st.empty(): # 공간을 확보했다가 분석 끝나면 교체
        
        try:
            model = YOLO('best.pt') 
            # PIL 이미지를 모델에 바로 전달
            results = model.predict(source=image, save=False, conf=0.25)
            
            extracted_data = []
            found_items = [] # 진단 메시지용 리스트 초기화
            
            # 클래스 번호를 다정한 한국어 이름으로 바꾸는 매핑
            # 모델의 'names' 인덱스 순서와 일치해야 합니다.
            friendly_names = {
                0: "나무", 1: "기둥", 2: "수관", 3: "가지", 4: "뿌리", 5: "나뭇잎",
                6: "꽃", 7: "열매", 8: "그네", 9: "새", 10: "다람쥐",
                11: "구름", 12: "달", 13: "별", 14: "사람", 15: "머리",
                16: "얼굴", 17: "눈", 18: "코", 19: "입", 20: "귀",
                # ... (모델 학습 시 설정한 인덱스에 맞춰 계속 추가)
                34: "집", 35: "지붕", 37: "문", 38: "창문", 46: "태양"
            }
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    # 매핑에 있으면 한국어로, 없으면 모델 기본 이름 사용
                    display_name = friendly_names.get(cls_id, r.names[cls_id])
                    
                    # 면적 계산을 통한 크기 묘사
                    bbox = box.xywh[0] # [x, y, w, h]
                    area = float(bbox[2] * bbox[3])
                    size_desc = "커다란" if area > (image.size[0] * image.size[1] * 0.2) else "귀여운"
                    
                    found_items.append(display_name)
                    extracted_data.append({
                        "찾은 것": display_name,
                        "크기": f"{size_desc} {display_name}야!",
                        "느낌": "정성 가득 그려졌어 ✨"
                    })

                    time.sleep(3) # 분석 느낌을 주기 위한 잠깐의 대기
                    st.empty() # 애니메이션 지우기
                    
        except Exception as e:
            st.error(f"모델 분석 중에 문제가 생겼어: {e}")
            extracted_data = []
            found_items = []

    # 분석 완료 후 하트 튀어나오기!
    st_lottie(lottie_success, speed=1, loop=False, height=300, key="success")
    st.balloons()
    
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("🎨 그림 속에서 찾은 이야기들")
        if extracted_data:
            st.table(pd.DataFrame(extracted_data))
            # 분석 결과 이미지 (Bounding Box 포함)
            res_plotted = results[0].plot() # BGR 형태
            st.image(res_plotted, channels="BGR", use_container_width=True, caption="마음친구가 분석한 그림이야!")
        else:
            st.info("그림 속에서 아직 이야기를 찾지 못했어. 조금 더 크게 그려볼까?")

    with col_res2:
        st.subheader("🧬 마음친구가 들려주는 이야기")
        
        # TypeError 방지를 위해 found_items가 있을 때만 join 실행
        if found_items:
            # 중복 제거 후 최대 2개만 노출
            unique_items = list(dict.fromkeys(found_items))
            items_str = ", ".join(unique_items[:2])
            diagnosis_msg = f"와! 그림 속에서 **{items_str}** 등을 찾았어. " \
                            f"정말 따뜻한 느낌이 드는 그림이야! 이 그림을 그릴 때 어떤 기분이었어?"
        else:
            diagnosis_msg = "그림을 보고 있으니 마음이 편안해져. 어떤 이야기를 담고 있는지 더 듣고 싶어!"
            
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
