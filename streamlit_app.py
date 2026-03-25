import streamlit as st
from PIL import Image
import time
import random

# 1. 페이지 설정 및 제목 (메인 통합형)
st.set_page_config(page_title="아동 심리 대화형 AI", layout="wide")

# --- 배너 이미지 추가 (수정된 부분) ---
# 로컬 폴더에 이미지가 있다고 가정합니다.
try:
    banner_image = Image.open("main_concept_image.webp")
    # 이미지를 메인 상단에 배너처럼 시원하게 넣습니다.
    # use_container_width=True를 사용해 화면 너비에 맞춥니다.
    st.image(banner_image, caption="아동 그림 심리 분석 배너", use_container_width=True)
except FileNotFoundError:
    # 이미지가 없을 경우를 대비한 안전장치
    st.error("'image_2.png' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    
st.title("🎨 아동 그림 기반 멀티모달 대화형 AI")
st.caption("3차 프로젝트: SAM2 + Wav2Vec2 + RAG 통합 프로토타입")

# 2. 내부 처리 (숨김) - 가중치 로드 시뮬레이션
if 'models_loaded' not in st.session_state:
    with st.spinner("시스템 초기화 중... (가중치 로딩 및 내부 처리)"):
        time.sleep(1.5) # 시뮬레이션
        st.session_state['models_loaded'] = True
        st.session_state['analysis_done'] = False
        st.session_state['chat_history'] = [] # 대화 내용 저장

# 3. 데이터 입력 섹션 (메인 상단)
st.header("1. 아동 데이터 입력")
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("🖼️ 그림 이미지 업로드")
    img_file = st.file_uploader("아동의 그림을 선택하세요", type=["jpg", "png", "jpeg"], key="img")
    if img_file:
        image = Image.open(img_file)
        st.image(image, use_column_width=True, caption="업로드된 그림")

with col_input2:
    st.subheader("🧒 아동 정보 입력")
    child_age = st.slider("아동 연령", 5, 13, 7)
    child_sex = st.radio("성별", ["남", "여"], horizontal=True)
    
    st.write("---")
    analyze_btn = st.button("AI 분석 및 대화 시작 🚀", use_container_width=True)

# 4. 분석 결과 및 대화 섹션
if analyze_btn and img_file:
    st.session_state['analysis_done'] = True
    st.divider()
    
    # --- (Step 1) 이미지 분석 결과 전시 ---
    st.header("2. AI 통합 분석 결과 (YOLO + SAM2)")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("🎨 객체 세분화 및 데이터 추출")
        # 시뮬레이션: YOLO/SAM2 가중치를 활용한 분석 결과
        time.sleep(1)
        # 예시: 47개 클래스 기반 추출 데이터
        extracted_data = [
            {"객체": "집-전체", "비율": 0.35, "특징": "크기가 큼"},
            {"객체": "나무-뿌리", "비율": 0.05, "특징": "지면에 노출됨"},
            {"객체": "사람-눈", "비율": 0.01, "특징": "크기가 매우 작음"}
        ]
        st.dataframe(pd.DataFrame(extracted_data), use_container_width=True)

    with col_res2:
        st.subheader("🧬 초기 심리 소견")
        # RAG 시뮬레이션: 논문 기반 초기 분석
        initial_diagnosis = "그림에서 나타나는 '노출된 뿌리'와 '작은 눈'은 정서적 불안과 대인관계 위축을 시사할 수 있습니다. 추가 대화를 통해 확인이 필요합니다."
        st.success(initial_diagnosis)
        st.session_state['initial_diagnosis'] = initial_diagnosis

# --- (Step 2) LLM과의 연속 대화 및 음성 분석 (선택사항) ---
if st.session_state.get('analysis_done'):
    st.divider()
    st.header("3. AI 심리 상담사와의 대화 (선택사항)")
    
    # (A) 음성 녹음 기능 (Wav2Vec2 분석용)
    st.subheader("🎙️ 사용자의 음성 녹음 및 분석")
    st.write("그림 분석 결과에 대해 이야기해 볼까요? 버튼을 눌러 녹음을 시작하세요.")
    
    # 스트림릿 자체 녹음 컴포넌트 사용 (시뮬레이션)
    audio_file = st.audio_input("대화 내용을 녹음하세요")
    
    if audio_file:
        with st.spinner("Wav2Vec2 음성 분석 및 특징 추출 중..."):
            time.sleep(1)
            # Wav2Vec2 가중치 활용 특징 추출 시뮬레이션
            st.session_state['audio_analysis'] = "불안 톤 감지, 발화 속도 느림"
            st.info(f"🎙️ 음성 분석 결과: {st.session_state['audio_analysis']}")
            
            # 음성 데이터를 텍스트로 변환 (STT 시뮬레이션)
            user_speech_text = "우리 아이가 요즘 좀 불안해하는 것 같나요? 눈을 작게 그린 게 마음에 걸려요."
            st.session_state['chat_history'].append({"role": "user", "content": f"(음성 질문) {user_speech_text}"})
            
            # LLM 응답 생성 (RAG 통합 시뮬레이션)
            with st.spinner("AI 상담사가 답변을 생성 중입니다..."):
                time.sleep(1.5)
                # 오디오 분석 결과와 그림 분석 결과를 종합하여 응답 생성
                llm_response = f"부모님의 목소리에서도 걱정이 느껴지네요. 그림 분석 결과('작은 눈')와 음성 특징('불안 톤')을 종합해 볼 때, 아이가 대인관계에서 위축감을 느끼고 있을 가능성이 있습니다. 혹시 최근 아이가 새로운 환경에 노출되었거나 힘들어하는 일이 있었나요?"
                st.session_state['chat_history'].append({"role": "assistant", "content": llm_response})

    # (B) 실시간 대화창
