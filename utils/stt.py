"""
utils/stt.py
─────────────────────────────────────────
Google Cloud Speech-to-Text  (REST API + API Key 방식)

인증 설정:
  .streamlit/secrets.toml 에 아래 한 줄이 있으면 됩니다.

  GCP_STT_API_KEY = "AIza..."

  ※ GCP 콘솔 → API 및 서비스 → 사용자 인증 정보 → API 키
  ※ Cloud Speech-to-Text API 가 해당 프로젝트에서 활성화되어 있어야 합니다.

필요 패키지:
  pip install requests
  (google-cloud-speech 불필요)
"""

from __future__ import annotations

import base64
import io
import logging
import wave

import requests

logger = logging.getLogger(__name__)

_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"


# ══════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ══════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    """st.secrets 에서 GCP_STT_API_KEY 를 읽어 반환한다."""
    import streamlit as st
    return st.secrets["GCP_STT_API_KEY"]


def _get_audio_info(audio_bytes: bytes) -> tuple[int, int]:
    """
    WAV 헤더에서 (sample_rate, channels) 를 읽는다.
    WAV 가 아닌 경우 기본값 (16000, 1) 반환.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            return wf.getframerate(), wf.getnchannels()
    except Exception:
        return 16000, 1


def _detect_encoding(audio_bytes: bytes) -> str:
    """
    오디오 포맷을 자동 감지하여 REST API 용 encoding 문자열 반환.
    Streamlit st.audio_input 은 WAV(LINEAR16) 를 반환하므로 기본적으로 LINEAR16.
    """
    if audio_bytes[:4] == b"RIFF":
        return "LINEAR16"
    if audio_bytes[:4] in (b"OggS", b"\x1aE\xdf\xa3"):
        return "OGG_OPUS"
    return "ENCODING_UNSPECIFIED"


# ══════════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════════

def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = "ko-KR",
    *,
    enable_auto_punctuation: bool = True,
    use_enhanced: bool = True,
) -> str | None:
    """
    Google Cloud Speech-to-Text V1 REST API 로 음성을 텍스트로 변환한다.
    인증은 st.secrets["GCP_STT_API_KEY"] 를 사용한다.

    Parameters
    ----------
    audio_bytes : bytes
        Streamlit st.audio_input 으로부터 받은 오디오 바이트.
    language_code : str
        인식 언어 (기본값: 'ko-KR').
    enable_auto_punctuation : bool
        자동 문장 부호 삽입 여부 (기본값: True).
    use_enhanced : bool
        True  → 'latest_long' 모델 (대화·장문 최적화)
        False → 'command_and_search' 모델 (짧은 명령)

    Returns
    -------
    str | None
        인식된 텍스트, 실패 시 None.
    """
    if not audio_bytes:
        logger.warning("transcribe_audio: 빈 오디오 데이터")
        return None

    try:
        api_key               = _get_api_key()
        encoding              = _detect_encoding(audio_bytes)
        sample_rate, channels = _get_audio_info(audio_bytes)
        model                 = "latest_long" if use_enhanced else "command_and_search"
        audio_b64             = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "config": {
                "encoding":                   encoding,
                "sampleRateHertz":            sample_rate,
                "audioChannelCount":          channels,
                "languageCode":               language_code,
                "enableAutomaticPunctuation": enable_auto_punctuation,
                "model":                      model,
                # 아동 심리 맥락 단어 힌트 — 인식률 향상
                "speechContexts": [
                    {
                        "phrases": [
                            "집", "나무", "사람", "하늘", "구름", "가족",
                            "친구", "학교", "행복", "슬픔", "무서워", "좋아",
                        ],
                        "boost": 10.0,
                    }
                ],
            },
            "audio": {
                "content": audio_b64,
            },
        }

        response = requests.post(
            _STT_ENDPOINT,
            params={"key": api_key},
            json=payload,
            timeout=15,
        )

        if response.status_code != 200:
            logger.error(
                f"transcribe_audio HTTP {response.status_code}: {response.text}"
            )
            return None

        results = response.json().get("results", [])

        if not results:
            logger.info("transcribe_audio: 인식 결과 없음")
            return None

        alt        = results[0]["alternatives"][0]
        transcript = alt.get("transcript", "").strip()
        confidence = alt.get("confidence", 0.0)
        logger.info(f"transcribe_audio: '{transcript}' (신뢰도: {confidence:.2f})")

        return transcript or None

    except requests.exceptions.Timeout:
        logger.error("transcribe_audio: 요청 타임아웃 (15s)")
        return None
    except Exception as exc:
        logger.error(f"transcribe_audio 오류: {exc}", exc_info=True)
        return None