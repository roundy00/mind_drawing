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
            st.info("감정
