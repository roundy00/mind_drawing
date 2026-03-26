import streamlit as st
from PIL import Image
import time
import base64
import pandas as pd
from ultralytics import YOLO  # YOLOv8 로드를 위해 필요합니다.

# 1. 페이지 설정
st.set_page_config(page_title="마음 그리는 AI 친구", layout="wide")

# --- 추가: 모델 로드 함수 (캐싱을 통해 속도 향상) ---
@st.cache_resource
def load_yolo_model():
    # 다영님이 파인튜닝하신 모델 경로(예: 'best.pt')를 입력하세요.
    # 테스트용이라면 'yolov8n.pt'를 사용하면 자동으로 다운로드됩니다.
    model = YOLO('best.pt') 
    return model

# 배너이미지 처리 (생략/유지)
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return None

img_base64 = get_image_base64("main_concept_image.webp")
if img_base64:
    st.markdown(f'<img src="data:image/png;base64,{img_base64}" style="width:100%; border-radius: 15px;">', unsafe_allow_html=True)
else:
    st.info("✨ 예쁜 그림 친구가 곧 도착할 거예요!")

st.title("🌟 안녕? 나는 네 마음을 읽어주는 '마음친구'야!")
st.write("네가 그린 멋진 그림과 예쁜 목소리를 들려주면, 내가 네 마음을 토닥토닥해줄게. 🥰")

# 2. 내부 처리 및 모델 로딩
if 'models_loaded' not in st.session_state:
    with st.spinner("AI 친구가 기지개를 켜며 준비하고 있어요... 💤"):
        st.session_state['model'] = load_yolo_model() # 모델 로드
        time.sleep(1)
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

# 4. 분석 결과 섹션 (YOLO 모델 적용)
if analyze_btn and img_file:
    st.session_state['analysis_done'] = True
    
    with st.spinner("그림 속 이야기를 찾아보고 있어요... 🔍"):
        # --- YOLO 추론 시작 ---
        model = st.session_state['model']
        results = model.predict(image) # 업로드된 PIL 이미지 바로 사용
        
        # 결과 데이터 정리
        detected_objects = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                detected_objects.append({
                    "찾은 것": label,
                    "확신 정도": f"{conf*100:.1f}%",
                    "메모": "그림에 그려져 있어요!"
                })
        
        # 분석된 이미지를 화면에 표시하기 위해 결과 렌더링
        res_plotted = results[0].plot() # BGR numpy array
        res_image = Image.fromarray(res_plotted[:, :, ::-1]) # RGB로 변환

    st.divider()
    st.header("2. 네 마음속에 이런 보물이 들어있구나! 💎")
    
    col_res1, col_res2 = st.columns([1.5, 1])
    with col_res1:
        st.subheader("🎨 AI가 분석한 그림 결과")
        st.image(res_image, use_container_width=True, caption="그림에서 이런 것들을 찾았어!")
        
        if detected_objects:
            st.table(pd.DataFrame(detected_objects))
        else:
            st.warning("아직은 그림에서 특별한 조각을 찾지 못했어. 조금 더 명확하게 그려볼까?")

    with col_res2:
        st.subheader("🧬 마음친구가 들려주는 이야기")
        # 실제 탐지된 객체에 따라 메시지 생성 가능
        if detected_objects:
            obj_names = [d["찾은 것"] for d in detected_objects]
            initial_diagnosis = f"그림 속에 {', '.join(obj_names[:2])} 등을 그렸구나! 네 마음속 이야기를 더 들려줄 수 있니?"
        else:
            initial_diagnosis = "그림을 보니 정말 정성스럽게 그린 게 느껴져! 어떤 마음으로 그렸는지 궁금해."
        st.success(initial_diagnosis)

# 5. 대화 세션 (기존 코드 유지)
if st.session_state.get('analysis_done'):
    st.divider()
    st.header("3. 우리 같이 도란도란 이야기하자 💬")
    # ... (이후 대화 및 오디오 입력 코드는 동일하게 유지)
