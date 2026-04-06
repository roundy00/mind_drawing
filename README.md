# 🌟 마음친구 (MindDrawing AI)

> 아이가 그린 그림 한 장으로 마음을 읽어주는 AI 미술 심리 분석 서비스

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📖 프로젝트 소개

**마음친구**는 아동이 직접 그린 HTP(House-Tree-Person) 그림을 업로드하면, AI가 그림 속 요소를 분석하고 아이와 대화를 나눈 뒤 심리 관찰 보고서를 생성해주는 서비스입니다.

미술 심리 검사의 접근성을 높이고, 부모·교사가 아이의 내면을 더 깊이 이해할 수 있도록 돕는 것을 목표로 합니다.

> ⚠️ 본 서비스는 **교육 목적의 참고 자료**를 제공하며, 의료적 진단이나 치료를 대체하지 않습니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🖼️ **그림 업로드** | JPG/PNG 그림 파일 업로드 및 아동 정보(나이, 성별) 입력 |
| 🎨 **색채 분석** | ColorThief 기반 주요 색상 및 팔레트 자동 추출 |
| 🤖 **객체 탐지 (YOLO)** | 커스텀 학습된 YOLOv8 모델로 집·나무·사람 구성 요소 47종 탐지 |
| 🔍 **탐지 검증** | GPT-4o가 아동 설명을 기반으로 YOLO 오탐을 자동 보정 |
| 🎤 **음성 입력 (STT)** | Google Cloud STT를 이용한 한국어 음성 인식 |
| 💬 **AI 대화** | 집·나무·사람 그룹별로 맞춤 질문을 생성하고 대화 진행 |
| 📋 **심리 관찰 보고서** | 감정 분석 → 탐색 질문 → 추천 대화법 → 종합 보고서 자동 생성 |
| 📚 **RAG 논문 인용** | FAISS 기반 미술 심리 관련 논문 검색 및 보고서 내 인용 |
| 💾 **보고서 저장/불러오기** | JSON 다운로드 및 이전 보고서 재불러오기 지원 |

---

## 🛠️ 기술 스택

```
Frontend     │ Streamlit
LLM          │ OpenAI GPT-4o
STT          │ Google Cloud Speech-to-Text
객체 탐지    │ Ultralytics YOLOv8 (커스텀 학습, best.pt)
색채 분석    │ ColorThief
임베딩 (RAG) │ OpenAI text-embedding-3-large + HuggingFace all-MiniLM-L6-v2
벡터 DB      │ FAISS (LangChain Community)
이미지 처리  │ Pillow
데이터       │ Pandas
```

---

## 🗂️ 서비스 흐름

```
1. 그림 업로드 + 아동 정보 입력
        ↓
2. YOLO 객체 탐지 + 색채 분석
        ↓
3. 아동/보호자 그림 설명 입력 (텍스트 or 음성)
        ↓
4. GPT-4o로 YOLO 오탐 검증 및 보정
        ↓
5. 집 / 나무 / 사람 그룹별 AI 대화
   (GPT-4o가 개방형 질문 생성 → 음성/텍스트 답변 → 후속 질문)
        ↓
6. 보고서 생성
   ├─ 감정 언어 분석
   ├─ 자기 탐색 질문 생성
   ├─ 추천 대화법 & 대화 주제 생성
   ├─ FAISS RAG 논문 검색
   └─ 종합 심리 관찰 보고서 작성
        ↓
7. JSON 보고서 다운로드
```

---

## 🏗️ 주요 컴포넌트

### 🔎 YOLO 객체 탐지 (`run_yolo`)
커스텀 학습된 `best.pt` 모델로 47가지 클래스(집·나무·사람 구성 요소)를 탐지합니다. 탐지된 각 객체는 위치(상단/중간/하단, 왼쪽/중앙/오른쪽), 면적 비율, 신뢰도 정보와 함께 기록됩니다.

```
탐지 클래스 예시:
집   → 집전체, 지붕, 집벽, 문, 창문, 굴뚝, 연기, 울타리
나무 → 나무전체, 기둥, 수관, 가지, 뿌리, 나뭇잎, 꽃, 열매
사람 → 사람전체, 머리, 얼굴, 눈, 코, 입, 귀, 팔, 손, 다리, 발 ...
기타 → 태양, 구름, 달, 별, 산, 잔디, 연못 등
```

### 🔍 YOLO 탐지 검증 (`verify_yolo_with_description`)
아동이 직접 설명한 그림 내용과 YOLO 결과를 GPT-4o로 비교하여, 설명과 명확히 모순되는 항목만 오탐으로 분류하고 설명에서 추가로 확인된 요소를 보완합니다.

### 💬 대화 큐 시스템 (`build_conversation_queue`)
HTP 세 그룹(집·나무·사람)별로 대화 큐를 구성하고, 그림 설명만으로 충분히 파악된 그룹은 자동으로 건너뜁니다. 최대 3개의 기본 질문과 1개의 후속 질문을 생성합니다.

### 📚 RAG 보고서 생성 (`generate_report`)
두 개의 FAISS 벡터 DB(OpenAI 임베딩 기반 + SentenceTransformer 기반)에서 관련 미술 심리 논문을 검색하여 보고서에 근거로 인용합니다.

---

## 👥 팀 소개 — 사고뭉치

**프로젝트 기간:** 2026.03.10 – 2026.04.07 (총 29일)

| 이름 | 담당 개발 파트 | 주요 작업 |
|------|-----------|-----------|
| **김학범** (팀장) | 🎙️ 오디오 / 🖥️ 프로토타입 | Google Cloud STT 연동, 음성 입력 UI 구현, 오디오 감정 데이터 전처리 및 EDA, 음성 기반 감정 인식 모델 후보 선정, 프로토타입 구현 |
| **이하정** | 🖼️ 이미지 | 이미지 전처리, SAM2 파인튜닝, 그림 데이터 EDA |
| **정다영** | 🖼️ 이미지 + 📚 RAG + 🖥️ 프로토타입 | 이미지 전처리, 색채 분석 파이프라인 구축, FAISS 벡터 DB 구축(OpenAI 임베딩), Color-Pedia 전처리 및 감정 레이블 그룹화, 프로토타입 구현 |
| **조강재** | 📚 RAG | PDF OCR 보완, 텍스트 임베딩 파이프라인(all-MiniLM-L6-v2), FAISS 인덱스 생성 |

---

### 🎙️ 오디오 파트 상세 (김학범)

아동과의 대화에서 발생하는 음성 입력을 처리하고, 향후 음성 기반 감정 인식(SER)으로 확장하기 위한 데이터 파이프라인을 구축했습니다.

#### 활용 데이터셋

- **감정 분류를 위한 대화 음성 데이터셋** (AI-Hub): 총 43,991건(wav + csv), 7감정 라벨. 5명 평가자 과반수 필터링 후 3클래스(긍정/중립/부정)로 재분류. Fleiss' Kappa 0.46(7감정) / 0.50(5감정) 확인.
- **감정이 태깅된 자유대화 — 청소년** (AI-Hub): 3,467 세션 사용(전체의 32.3%). 발화 분할 후 5자·0.5초 미만 필터링으로 739,118건(67.2%) 확보, 층화 분할 적용.

#### 모델 후보

| 모델 | 용도 |
|------|------|
| `kresnik/wav2vec2-large-xlsr-korean` | 한국어 음성 인식 (ASR) |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | 음성 기반 감정 인식 (SER) |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | 음성 기반 연령·성별 인식 |
| `FunAudioLLM/SenseVoiceSmall` | 초저지연 다국어 ASR + SER + AED |

#### 현재 구현 상태

현재는 **Google Cloud STT** 기반 음성 인식을 사용하며, 위 모델들은 향후 온디바이스 감정 분석 확장을 위한 후보군입니다. 오디오 해시 기반 중복 전사 방지 로직 적용, 인식 텍스트는 편집 가능한 `text_area`로 노출됩니다.

#### 향후 확장 방향

| 단계 | 내용 |
|------|------|
| 1단계 *(현재)* | Google Cloud STT → 텍스트 변환 후 GPT-4o 분석 |
| 2단계 | wav2vec2 기반 SER 추가 → 어조·운율 보조 피처화 |
| 3단계 | 이미지(YOLO) + 텍스트(LLM) + 오디오(SER) 멀티모달 교차 검증 통합 |
| 4단계 | 엣지(Edge) 기반 구동 검토 → 민감 아동 데이터 온디바이스 처리 |

---

### 🖼️ 이미지 파트 상세 (이하정·정다영)

그림 이미지에서 심리 분석에 필요한 객체 및 색채 정보를 추출하는 파이프라인을 구축했습니다.

#### 주요 작업

- **YOLO 학습 데이터 전처리**: AI-Hub 제공 HTP 그림 데이터(50,400건)의 JSON 라벨을 YOLO 규격(txt + data.yaml)으로 변환. 총 47개 클래스로 재구조화 및 성별 분기 처리(남자/여자 머리카락 등).
- **색채 분석**: ColorThief 기반 주요 색상 및 5색 팔레트 자동 추출. Color-Pedia(100,000건) 데이터를 활용하여 SentenceTransformer로 13개 감정 그룹에 자동 매핑.
- **SAM2 검토**: 세그멘테이션 모델(SAM2)의 HTP 그림 적용 가능성 검토.
- **그림 데이터 EDA**: 객체 위치·크기·동시 등장 패턴 분석, 카테고리별 바운딩 박스 분포 시각화.

---

### 📚 RAG 파트 상세 (정다영·조강재)

HTP 관련 미술 심리 논문을 벡터화하여 보고서 생성 시 근거 자료로 활용하는 검색 파이프라인을 구축했습니다.

#### 주요 작업

- **논문 DB 구축**: 수집 된 HTP 및 색채 심리 관련 논문 PDF 파일 → 텍스트 추출(OCR 보완) → 청크 분할.
- **이중 벡터 DB 구성**:
  - `text-embedding-3-large` (OpenAI) 기반 FAISS 인덱스 → `final_index/`
  - `all-MiniLM-L6-v2` (SentenceTransformer) 기반 FAISS 인덱스 → `st_index/`
- **검색 전략**: 두 DB에서 동시 검색 후 중복 제거, 보고서 생성 시 관련 논문 청크를 프롬프트에 삽입하여 근거 기반 답변 생성.

---

## 🚀 로컬 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-org/mind_drawing.git
cd mind_drawing
```

### 2. 패키지 설치

> Python **3.10** 환경을 권장합니다.

```bash
pip install -r requirements.txt
```

### 3. Secrets 설정

`.streamlit/secrets.toml` 파일을 생성하고 아래 키를 입력합니다:

```toml
OPENAI_API_KEY = ""
GCP_STT_API_KEY = ""
```

### 4. 필수 파일 준비

```
프로젝트 루트/
├── best.pt                    # YOLOv8 커스텀 학습 모델
├── final_index/               # OpenAI 임베딩 FAISS 인덱스
│   └── *.faiss
├── st_index/                  # SentenceTransformer FAISS 인덱스
│   └── *.faiss
├── main_concept_image.webp    # 배너 이미지 (선택)
└── utils/
    └── stt.py                 # Google Cloud STT 유틸리티
```

### 5. 실행

```bash
streamlit run streamlit_app.py
```

---

## 📁 프로젝트 구조

```
mind_drawing/
├── streamlit_app.py        # 메인 Streamlit 앱
├── best.pt                 # YOLOv8 커스텀 모델
├── requirements.txt
├── final_index/            # FAISS 벡터 DB (OpenAI 임베딩)
├── st_index/               # FAISS 벡터 DB (SentenceTransformer)
├── utils/
│   └── stt.py              # Google STT 래퍼
└── main_concept_image.webp # 배너 이미지
```

---

## 🔐 환경 변수

| 키 | 설명 |
|----|------|
| `OPENAI_API_KEY` | OpenAI API 키 (GPT-4o, text-embedding-3-large 사용) |
| `GCP_STT_API_KEY` | Google Cloud STT API 키 |

---

## 📊 보고서 구성

생성되는 심리 관찰 보고서는 다음 항목으로 구성됩니다:

- **전체 요약 및 핵심 인사이트** (Executive Summary)
- **주요 발견 사항** (Key Findings)
- **감정 언어 분석** (주요 감정, 감정적 톤, 상징적 요소, 강도)
- **활용 방향성 및 집중 탐색 영역**
- **AI 종합 분석** (색채·구성·대화 기반)
- **자기 탐색 질문** (부모·교사 활용용)
- **추천 대화법 & 추천 대화 주제**
- **참고 논문 인용** (RAG 기반)

---

## 📄 라이선스

본 프로젝트는 교육·연구 목적으로 제작되었습니다.

---

<p align="center">
  Made with ❤️ by <b>사고뭉치</b> 팀
</p>
