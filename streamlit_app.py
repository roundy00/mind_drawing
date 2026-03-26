import streamlit as st
from PIL import Image
import time
import random
import base64
import pandas as pd
from ultralytics import YOLO

# 1. 페이지 설정
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

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
    st.session_state['analysis_done'] = True
    st.divider()

    # --- [핵심] YOLO 모델 로드 및 추론 ---
    with st.spinner("AI 친구가 그림을 꼼꼼하게 살펴보고 있어요... 🧐"):
        try:
            model = YOLO('best.pt')  # 다영님의 가중치 파일
            results = model.predict(source=image, save=False, conf=0.25)
            
            # 탐지된 데이터를 담을 리스트
            extracted_data = []
            
            for r in results:
                names = r.names
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    label = names[class_id]
                    confidence = float(box.conf[0])
                # --- [심리상담용 전처리] 레이블을 아이용 친근한 단어로 변경 ---
                # 클래스매핑
                # 통합된 전체 클래스 매핑 (총 47종)
                class_mapping = {
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
                display_name = class_mapping.get(label, label)
                
                # 박스 크기(면적) 계산으로 '크기' 표현 자동화
                bbox = box.xywh[0] # [x, y, w, h]
                area = float(bbox[2] * bbox[3])
                size_desc = "커다란" if area > 50000 else "귀여운" # 이미지 크기에 따라 조정 필요
                
                extracted_data.append({
                    "찾은 것": display_name,
                    "크기": f"{size_desc} {display_name}야!",
                    "느낌": "정성 가득 그려졌어 ✨"
                })
        except Exception as e:
            st.error(f"모델을 불러오는 중에 문제가 생겼어: {e}")
            extracted_data = []
    
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("🎨 그림 속에서 찾은 이야기들")
        if extracted_data:
            # 추출된 실제 데이터를 테이블로 표시
            st.table(pd.DataFrame(extracted_data))
            
            # 탐지된 박스가 그려진 이미지 보여주기
            res_plotted = results[0].plot() # BGR numpy array
            st.image(res_plotted, channels="BGR", use_container_width=True, caption="마음친구가 분석한 그림이야!")
        else:
            st.info("그림 속에서 아직 이야기를 찾지 못했어. 다시 한번 보여줄래?")

    with col_res2:
        st.subheader("🧬 마음친구가 들려주는 이야기")
        
        # 탐지된 객체에 따라 동적으로 진단 멘트 생성
        if extracted_data:
            found_items = [d["찾은 것"] for d in extracted_data]
            diagnosis_msg = f"와! 그림 속에서 {', '.join(found_items[:2])} 등을 찾았어. " \
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
