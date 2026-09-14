"""Plan.md Step 6 [구조 이동, 외부 시그니처 불변]: engine.py의 evaluate()가
model(f_in=..., pos=..., ...)를 직접 호출하던 부분을
common/adapters._shared.engine_style_forward로 교체하기 전에, 그 공유
함수의 존재와 PaiNN/Equiformer adapter가 정확히 이 함수를 가리키는지부터
검증한다 (중복 제거 확인).
"""

import torch

from common.adapters._shared import engine_style_forward
from common.adapters import painn_adapter, equiformer_adapter
from tests.support.golden import assert_matches_golden
from tests.support.tiny_batches import make_tiny_pyg_batch, n_mode_target_count


def test_painn_adapter와_equiformer_adapter의_forward는_같은_공유_함수를_가리킨다():
    assert painn_adapter.forward is engine_style_forward
    assert equiformer_adapter.forward is engine_style_forward


def test_engine_style_forward는_PaiNN_Step1_오라클과_동일하다():
    from types import SimpleNamespace

    num_classes = n_mode_target_count()
    args = SimpleNamespace(out_channels=num_classes, radius=5.0, num_basis=8, embed_dim=8, num_layers=1)
    torch.manual_seed(7)
    model = painn_adapter.build(args)
    model.eval()
    batch = make_tiny_pyg_batch()

    with torch.no_grad():
        output = engine_style_forward(model, batch)

    assert_matches_golden("painn_forward_output", output)
