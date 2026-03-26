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
    
    def run_object_detection(image_path, model_path='best.pt'):
        """
        이미지 경로와 모델 경로를 받아 객체를 탐지하고 결과를 반환합니다.
        """
        # 1. 모델 로드 (다영님이 학습시킨 가중치 파일)
        model = YOLO(model_path)
        
        # 2. 이미지 추론 (PIL 이미지나 경로 모두 가능)
        results = model.predict(source=image_path, save=False, conf=0.25)
        
        # 3. 결과 해석 및 출력
        for r in results:
            # 탐지된 객체의 이름들 확인
            names = r.names
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                
                print(f"탐지 성공: {label} (확률: {confidence:.2f})")
                print(f"위치 좌표: {bbox}")
    
        return results
    
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("🎨 그림 속에서 찾은 이야기들")
        # 아이들이 보기 편하게 '객체' 대신 '찾은 것'으로 표현
        extracted_data = [
            {"찾은 것": "튼튼한 우리 집", "크기": "이만큼 커요!", "느낌": "따뜻해 보여요"},
            {"찾은 것": "쑥쑥 자라는 나무", "크기": "귀여워요", "느낌": "힘이 세 보여요"},
            {"찾은 aristocratic": "반짝이는 눈", "크기": "동글동글해요", "느낌": "궁금한 게 많아 보여요"}
        ]
        st.table(pd.DataFrame(extracted_data)) # 데이터프레임보다 테이블이 더 직관적일 수 있음

    with col_res2:
        st.subheader("🧬 마음친구가 들려주는 이야기")
        initial_diagnosis = "그림 속에 커다란 집이 있는 걸 보니, 네 마음은 지금 아주 따뜻하고 행복한가 봐! 작은 눈을 그린 건 부끄러움이 조금 많아서 그럴 수도 있겠다. 그치? 우리 조금 더 얘기해볼까?"
        st.success(initial_diagnosis)

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
