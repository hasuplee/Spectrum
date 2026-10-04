"""학습/예측과 무관한 질문을 LLM 호출 전에 거절하는 범위 가드 (Plan.md Step 5A, G3).

키워드 휴리스틱이며 의미 기반 판별이 아니다. 따라서 통과시켜 놓고 LLM이 틀리게 답할 수 있는 경우
(예: 정의를 묻는 문장에 행동 표현이 함께 오는 "PaiNN 정의 알려줘")는 Agent의 instructions(Step 5C)가
두 번째 방어선이다. AGNO 가드레일(`pre_hooks`)로 쓰면 범위 밖 입력은 LLM 서버를 호출하지 않는다.
"""

import re

from agno.exceptions import CheckTrigger, InputCheckError
from agno.guardrails.base import BaseGuardrail

REFUSAL_MESSAGE = (
    "저는 인광 OLED 스펙트럼의 학습과 예측만 도와드릴 수 있어요. "
    "예를 들어 'PaiNN으로 학습해줘', '학습 기본값 보여줘', '이 분자의 spectrum 예측해줘'처럼 물어봐 주세요."
)

# UI(Step 6)가 보여 주는 예시 질문의 단일 출처. 테스트가 가드 판정과 데모 서버의 이해 여부와 일치함을 검증한다.
# 학습 시작 / 기본값·설정 / 상태 / 예측 / 목록·도움말 순.
ALLOWED_EXAMPLES = [
    "PaiNN으로 학습해줘",
    "Geoformer로 학습해줘",
    "Equiformer로 학습해줘",
    "그냥 학습해줘",
    "PaiNN으로 20 step만 학습해줘",
    "Geoformer로 배치 크기 4로 학습해줘",
    "학습 기본값 보여줘",
    "PaiNN 기본 설정 알려줘",
    "Equiformer 학습 기본값 보여줘",
    "학습 상태 알려줘",
    "학습 진행 상황 알려줘",
    "학습 끝났어?",
    "학습 로그 보여줘",
    "이 분자의 spectrum 예측해줘",
    "cn1_cn1_nn1 스펙트럼 예측해줘",
    "cn2_cn2_nn3 spectrum 예측해줘",
    "cn3_cn3_nn5 스펙트럼 보여줘",
    "분자 목록 보여줘",
    "어떤 분자가 있어?",
    "학습된 모델 알려줘",
    "사용 가능한 모델 알려줘",
    "뭘 할 수 있어?",
]
REFUSED_EXAMPLES = [
    "오늘 날씨가 뭐야?",
    "반도체는 뭐지",
    "OLED의 정의는",
    "파이썬 코드 짜줘",
    "PaiNN이 뭐야?",
    "점심 메뉴 추천해줘",
    "영어로 번역해줘",
    "Ir 착물이 뭐야?",
    "딥러닝이란 무엇인가요?",
    "농담 하나 해줘",
]

_DOMAIN_KEYWORDS = (
    "학습", "훈련", "train", "예측", "predict", "스펙트럼", "spectrum",
    "painn", "geoformer", "equiformer", "모델", "체크포인트", "checkpoint",
    "분자", "molecule", "irdb", "step", "스텝", "배치", "batch",
    "기본값", "파라미터", "parameter", "상태", "로그", "워커", "worker",
    # 대화 중 도메인 단어 없이 쓰이는 후속 표현 (Step 5C-1: 5A 리뷰에서 발견한 거짓 거절 보강)
    "진행 상황", "설정", "default", "결과", "그래프", "실패", "얼마나", "끝났", "뭘 할 수", "사용법", "기본으로",
)
_MOLECULE_ID = re.compile(r"[a-z]+\d*_[a-z]+\d*_nn\d+")  # 예: cn1_cn1_nn1

# 정의/설명을 묻는 표현. 행동 표현이 함께 없으면 범위 밖으로 본다.
_EXPLANATION_MARKERS = ("뭐야", "뭐지", "뭔가요", "뭐에요", "무엇", "정의", "설명", "이란", "what is", "define")
_ACTION_WORDS = ("해줘", "해 줘", "해주세요", "시작", "실행", "돌려", "보여", "알려", "조회", "확인", "취소", "중단")

# 대화 중 짧은 후속 답변. 모든 토큰이 이 목록에 있을 때만 허용한다 (예: "네, 진행해줘").
_REPLY_WORDS = {
    "응", "네", "예", "아니", "아니요", "좋아", "좋아요", "그래", "맞아", "취소", "중단",
    "진행", "진행해줘", "진행해주세요", "확인", "시작", "시작해줘", "해줘", "해주세요", "그대로", "그렇게",
    "ok", "okay", "yes", "no", "y", "n",
}


def is_in_scope(text) -> bool:
    """학습/예측과 관련된 입력이면 True. 빈 입력, 무관한 질문, 정의를 묻는 질문은 False."""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in _EXPLANATION_MARKERS) and not any(
            action in normalized for action in _ACTION_WORDS):
        return False
    if _is_short_reply(normalized):
        return True
    return any(keyword in normalized for keyword in _DOMAIN_KEYWORDS) or bool(_MOLECULE_ID.search(normalized))


def _is_short_reply(normalized) -> bool:
    tokens = [re.sub(r"[^\w]", "", token) for token in re.split(r"[\s,]+", normalized)]
    tokens = [token for token in tokens if token]
    return bool(tokens) and all(token in _REPLY_WORDS for token in tokens)


class ScopeGuardrail(BaseGuardrail):
    """범위 밖 입력이면 InputCheckError(REFUSAL_MESSAGE)를 던져 LLM 호출 없이 Agent 실행을 끝낸다."""

    def check(self, run_input) -> None:
        if not is_in_scope(run_input.input_content_string()):
            raise InputCheckError(REFUSAL_MESSAGE, check_trigger=CheckTrigger.INPUT_NOT_ALLOWED)

    async def async_check(self, run_input) -> None:
        self.check(run_input)
