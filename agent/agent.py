"""학습/예측 전용 AGNO Agent 조립 (Plan.md Step 5C-1, G3).

가드(agent.guard) · tool(agent.assistant_tools) · instructions · 대화 기억(InMemoryDb) · LLM(AGNO `VLLM`)을 합친다.
LLM은 OpenAI 호환 서버(vLLM 등)이며 환경변수로 설정한다: VLLM_BASE_URL, VLLM_MODEL(필수), VLLM_API_KEY(선택).
Agent는 **대화 세션마다 하나씩** 만든다 — tool의 미리보기 상태와 대화 기억이 Agent 인스턴스별이기 때문이다.
"""

import os

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.vllm import VLLM
from agno.run.base import RunStatus

from agent.assistant_tools import build_assistant_tools
from agent.guard import REFUSAL_MESSAGE, ScopeGuardrail
from agent.tools._shared import PROJECT_ROOT

INSTRUCTIONS = [
    "당신은 인광 OLED 발광 스펙트럼 예측 모델(PaiNN, Geoformer, Equiformer)의 학습과 예측만 도와주는 한국어 도우미입니다.",
    "학습과 예측 이외의 질문(날씨, 일반 상식, 용어 설명, 코드 작성 등)은 답하지 않고 정중히 거절한 뒤, 할 수 있는 일(학습 시작, 학습 기본값 조회, 학습 상태 확인, 분자 스펙트럼 예측)을 안내하세요.",
    "tool 결과에 없는 수치나 사실을 지어내지 마세요. tool 결과의 message와 status를 근거로 사용자에게 알려 주세요.",
    "학습 요청: 모델이 정해지지 않았으면 PaiNN, Geoformer, Equiformer 중 무엇으로 학습할지 먼저 되물으세요. "
    "모델이 정해지면 show_training_defaults로 기본값을 보여 주세요. device가 cpu이면 train_steps가 크면 매우 오래 걸린다고 알리고, 줄일지 물어보세요.",
    "학습을 시작하기 전에 반드시 preview_training을 호출하고, 결과의 최종 설정(모델, step 수, 배치 크기 등)을 사용자에게 보여 준 뒤 이대로 진행할지 물어보세요. "
    "사용자가 명시적으로 동의한 뒤에만 같은 인자로 start_training_confirmed를 호출하세요. 사용자의 동의 없이 start_training_confirmed를 호출하지 마세요.",
    "미리보기 결과의 will_overwrite가 true이면 이미 학습된 결과가 있어 다시 학습하면 기존 결과가 삭제된다고 분명히 알리고 별도로 확인하세요. "
    "사용자가 삭제에 동의한 경우에만 start_training_confirmed를 overwrite=True로 호출하세요. already_trained 응답이 오면 같은 방식으로 안내하세요.",
    "학습 상태를 물으면 check_training_status를 호출해 state(running/finished/failed), 경과 시간, 로그 마지막 줄을 알려 주세요. "
    "finished이면 예측할 수 있다고 안내하고, failed이면 로그 마지막 줄을 전달하세요. busy 응답은 이미 학습이 진행 중이라 끝난 뒤 다시 시작해야 한다고 알려 주세요.",
    "예측 요청: 분자 ID가 필요합니다. 모르면 list_molecule_ids로 찾도록 도와주세요. predict_molecule_spectrum의 결과는 피크 파장과 파장별 상대 강도(samples)로 설명하세요. "
    "needs_training 응답이면 아직 학습된 모델이 없다고 알리고 학습을 먼저 제안하세요.",
    "not_previewed 응답은 설정을 사용자에게 보여 주는 단계를 건너뛴 것이므로 preview_training부터 다시 진행하세요.",
]


class ModelConfigError(ValueError):
    """LLM 연결 환경변수가 올바르지 않을 때."""


def build_model(environ=None) -> VLLM:
    """환경변수로 AGNO `VLLM` 모델을 만든다. environ을 생략하면 os.environ.

    VLLM_BASE_URL, VLLM_MODEL은 필수(기본 주소로 조용히 붙지 않게 함). VLLM_API_KEY는 선택이며, AGNO `VLLM`이
    키를 요구하므로 인증 없는 서버를 위해 없으면 더미 값 "EMPTY"를 쓴다. 재시도는 1회로 줄여 연결 실패를 빨리 드러낸다.
    """
    environ = os.environ if environ is None else environ
    missing = [name for name in ("VLLM_BASE_URL", "VLLM_MODEL") if not environ.get(name)]
    if missing:
        raise ModelConfigError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing)}")
    return VLLM(
        id=environ["VLLM_MODEL"],
        base_url=environ["VLLM_BASE_URL"],
        api_key=environ.get("VLLM_API_KEY") or "EMPTY",
        max_retries=1,
    )


def build_agent(model=None, project_root=PROJECT_ROOT, results_root=None) -> Agent:
    """대화 세션 하나를 위한 Agent. model을 생략하면 build_model()로 환경변수에서 만든다."""
    return Agent(
        name="Spectrum Assistant",
        model=model if model is not None else build_model(),
        tools=build_assistant_tools(project_root=project_root, results_root=results_root),
        instructions=INSTRUCTIONS,
        pre_hooks=[ScopeGuardrail()],
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=10,
        tool_call_limit=8,
        telemetry=False,
    )


def chat(agent, message, session_id="default") -> str:
    """Agent에 메시지를 보내고 응답을 문자열로 돌려준다 (가드 거절과 LLM 서버 오류도 문자열로 정규화)."""
    result = agent.run(message, session_id=session_id)
    content = result.content if isinstance(result.content, str) else ("" if result.content is None else str(result.content))
    if result.status != RunStatus.error:
        return content
    if content == REFUSAL_MESSAGE:
        return REFUSAL_MESSAGE
    return (f"LLM 서버 호출에 실패했습니다: {content} "
            "VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY 설정과 서버 상태를 확인해 주세요.")
