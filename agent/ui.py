"""Gradio 화면과 실행 진입점 (Plan.md Step 6B, G4): `python -m agent.ui [--demo] [--port 7860]`.

화면은 로직을 갖지 않는다 — agent.ui_logic(6A)의 핸들러를 project_root/results_root에 묶어 이벤트에 연결할 뿐이다.
이벤트에는 `api_name`을 주어 브라우저 없이 gradio_client로 테스트할 수 있다.
탭 3개: 채팅(Agent, LLM 필요) / 학습 / 예측(둘은 LLM 없이 tool을 직접 호출).
"""

import argparse
import sys

import gradio as gr

from agent import ui_logic as logic
from agent.guard import ALLOWED_EXAMPLES, REFUSED_EXAMPLES
from agent.tools._shared import PROJECT_ROOT

MODELS = ["PaiNN", "Geoformer", "Equiformer"]
MODE_LABELS = {"real": "🟢 실제 LLM", "demo": "🧪 데모 모드", "disconnected": "🔴 LLM 연결 안 됨"}


def build_ui(llm_status, project_root=PROJECT_ROOT, results_root=None) -> gr.Blocks:
    """llm_status(agent.ui_logic.LlmStatus)에 따라 채팅 탭의 동작이 정해지는 화면. 학습/예측 탭은 LLM과 무관하다."""

    def on_chat(text, history, agent):
        history, agent = logic.chat_turn(text, history, agent, llm_status, project_root=project_root, results_root=results_root)
        return history, agent, ""

    def on_preview(model, use_quick, custom_json):
        return logic.training_preview(model, use_quick, custom_json, project_root=project_root)

    def on_start(model, use_quick, custom_json, confirmed, overwrite):
        return logic.training_start(model, use_quick, custom_json, confirmed, overwrite, project_root=project_root)

    def on_status():
        status, log, _running = logic.training_status_view(project_root=project_root)
        return status, log

    def on_molecules(query):
        return gr.update(choices=logic.molecule_choices(query, limit=50, project_root=project_root))

    def on_load_molecules():
        molecule_ids = logic.molecule_choices(limit=50, project_root=project_root)
        return gr.update(choices=molecule_ids, value=molecule_ids[0] if molecule_ids else None)

    def on_predict(molecule_id, model, show_experimental):
        return logic.prediction_view(molecule_id, model, show_experimental, results_root=results_root, project_root=project_root)

    with gr.Blocks(title="인광 OLED 스펙트럼 학습·예측") as demo:
        gr.Markdown("# 인광 OLED 발광 스펙트럼 학습·예측")
        gr.Markdown(f"{MODE_LABELS[llm_status.mode]} — {llm_status.message}", elem_id="llm-banner")

        with gr.Tabs():
            with gr.Tab("채팅"):
                chatbot = gr.Chatbot(label="대화", height=420)
                message = gr.Textbox(label="메시지", placeholder="예: PaiNN으로 학습해줘")
                send = gr.Button("보내기", variant="primary")
                agent_state = gr.State()  # 브라우저 세션마다 Agent 하나 (미리보기 상태와 대화 기억이 세션별)
                gr.Examples(examples=ALLOWED_EXAMPLES, inputs=message, label="이렇게 물어보세요 (누르면 입력창에 채워집니다)",
                            examples_per_page=len(ALLOWED_EXAMPLES))  # 한 쪽에 모두 보여 준다
                gr.Examples(examples=REFUSED_EXAMPLES, inputs=message, label="거절 시연 (학습/예측 이외 질문)",
                            examples_per_page=len(REFUSED_EXAMPLES))
                message.submit(on_chat, [message, chatbot, agent_state], [chatbot, agent_state, message], api_name="chat")
                send.click(on_chat, [message, chatbot, agent_state], [chatbot, agent_state, message], api_name=False)

            with gr.Tab("학습"):
                model = gr.Dropdown(MODELS, value="PaiNN", label="모델")
                defaults_md = gr.Markdown()
                model.change(logic.training_defaults_markdown, model, defaults_md, api_name="training_defaults")
                demo.load(logic.training_defaults_markdown, model, defaults_md, api_name=False)
                quick = gr.Checkbox(True, label="작은 설정으로 빠르게 시험 (CPU 권장 — 위의 기본 설정은 매우 오래 걸릴 수 있습니다)")
                custom = gr.Textbox(label="추가 설정 (JSON, 선택)", placeholder='{"train_steps": 50}')
                preview_md = gr.Markdown()
                gr.Button("미리보기").click(on_preview, [model, quick, custom], preview_md, api_name="training_preview")
                confirmed = gr.Checkbox(False, label="설정을 확인했습니다")
                overwrite = gr.Checkbox(False, label="기존 학습 결과를 삭제하고 덮어쓰기")
                start_md = gr.Markdown()
                gr.Button("학습 시작", variant="primary").click(
                    on_start, [model, quick, custom, confirmed, overwrite], start_md, api_name="training_start")
                gr.Markdown("### 학습 상태")
                status_md = gr.Markdown()
                log_box = gr.Textbox(label="로그", lines=12, interactive=False)
                gr.Timer(3.0).tick(on_status, None, [status_md, log_box], api_name="training_status")

            with gr.Tab("예측"):
                query = gr.Textbox(label="분자 검색 (ID 일부, Enter)")
                molecule = gr.Dropdown(choices=[], allow_custom_value=True, label="분자")  # 목록은 페이지 로드 시 채운다 (화면 생성은 데이터에 의존하지 않음)
                demo.load(on_load_molecules, None, molecule, api_name=False)
                query.submit(on_molecules, query, molecule, api_name="molecules")
                model_choice = gr.Dropdown(["자동", *MODELS], value="자동", label="모델")
                experimental = gr.Checkbox(True, label="실험 스펙트럼 함께 보기")
                plot = gr.LinePlot(x="wavelength_nm", y="intensity", color="종류", x_title="파장 (nm)", y_title="상대 강도", height=400)
                summary = gr.Markdown()
                gr.Button("예측", variant="primary").click(
                    on_predict, [molecule, model_choice, experimental], [plot, summary], api_name="predict")

    return demo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.ui", description="인광 OLED 스펙트럼 학습·예측 UI")
    parser.add_argument("--demo", action="store_true", help="LLM 없이 규칙 기반 데모 서버로 채팅 (진짜 LLM이 아님)")
    parser.add_argument("--port", type=int, default=7860, help="UI 포트 (기본 7860)")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    llm_status = logic.resolve_llm(demo=args.demo)
    print(f"[{llm_status.mode}] {llm_status.message}", flush=True)
    try:
        build_ui(llm_status).launch(server_name="127.0.0.1", server_port=args.port)  # 127.0.0.1에만 바인딩
    finally:
        logic.shutdown_demo_server()
    return 0


if __name__ == "__main__":
    sys.exit(main())
