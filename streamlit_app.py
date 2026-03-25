import streamlit as st
import torch
import time
from PIL import Image
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="Multi-Modal Child AI", layout="centered")

# 헤더 섹션
st.title("🧠 아동 심리 통합 분석 시스템")
st.markdown("---")

# 2. 모델 및 환경 설정 섹션 (메인 상단)
with st.expander("🛠️ 시스템 설정 및 가중치 로드", expanded=True):
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        img_model_type = st.selectbox(
            "이미지 객체 추출 모델",
            ["YOLOv8", "SAM2", "Mask2Former"]
        )
    with col_set2:
        st.write("**시스템 상태**")
        if st.button("모델 가중치 체크"):
            with st.spinner("가중치 확인 중..."):
                time.sleep(1)
                st.success(f"{img_model_type} & Wav2Vec2 로드 완료")

# 3. 데이터 업로드 섹션 (중앙)
st.header("📁 데이터 업로드")
up_col1, up_col2 = st.columns(2)

with up_col1:
    st.subheader("🖼️ 아동 그림")
    img_file = st.file_uploader("그림 이미지 (JPG/PNG)", type=["jpg", "png", "jpeg"], key="img")
    if img_file:
        st.image(img_file, use_column_width=True)

with up_col2:
    st.subheader("🎙️ 아동 음성")
    audio_file = st.file_uploader("음성 녹음 (WAV)", type=["wav"], key="audio")
    if audio_file:
        st.audio(audio_file)

# 4. 분석 실행 버튼
st.markdown("---")
if st.button("통합 심리 분석 시작 🚀", use_container_width=True):
    if img_file and audio_file:
        
        # 분석 시뮬레이션
        progress_text = "멀티모달 데이터를 병렬 분석 중입니다..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # 5. 결과 전시 섹션
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(f"🎨 {img_model_type} 결과")
            # 다영님이 매핑한 47개 클래스 기반 결과 예시
            st.info("탐지 객체: 집(창문, 문), 나무(뿌리), 사람(눈)")
            st.json({"창문_비율": 0.12, "뿌리_노출": "심함", "눈_크기": "작음"})

        with res_col2:
            st.subheader("🎙️ Wav2Vec2 결과")
            st.info("감정 톤: 위축/불안")
            st.json({"주파수_변동": "낮음", "발화_속도": "느림", "감정_스코어": 0.78})

        # 6. RAG 통합 진단 (최종)
        st.markdown("---")
        st.header("🧬 RAG 기반 종합 심리 진단")
        st.write("**참조 논문:** 아동 그림 검사(HTP) 및 음성 특성 간의 상관관계 연구(2025)")
        
        st.success("""
        **종합 소견:** 그림에서 나타나는 '뿌리 노출'과 음성 데이터의 '낮은 주파수 변동'이 공통적으로 정서적 위축을 시사합니다. 
        RAG 분석 결과, 현재 아동은 환경 변화에 따른 적응 스트레스를 겪고 있을 가능성이 높으므로 세밀한 관찰이 필요합니다.
        """)
        
    else:
        st.error("분석을 위해 그림과 음성 파일을 모두 업로드해주세요.")
else:
    st.write("이미지와 음성 데이터를 업로드한 후 분석 버튼을 눌러주세요.")
