"""Plan.md Step 2A: spectrum/ 내부 재구성 — 곡선 복원을 CSV 저장에서 분리한다.

- 특성화 테스트(1~4): 현재 spectrum.write.save_spectrum의 CSV 출력을 골든으로 고정한다.
  리팩토링(save_spectrum이 reconstruct_spectrum을 호출하도록 변경) 전후로 출력이 같아야 한다.
- 신규 인터페이스 테스트(5~10): spectrum.reconstruct가 아직 없으므로 실패해야 한다 (RED).
  새 모듈 import는 각 테스트 안에서 하여, 모듈이 없어도 특성화 테스트는 독립적으로 통과한다.
"""

import pandas as pd
import pytest
import torch

from spectrum.write import save_spectrum
from tests.support.golden import assert_matches_golden

NUM_WAVELENGTHS = 800  # 400~800nm, 0.5nm 간격


def _fc_preds(n_mode: int = 3) -> torch.Tensor:
    """[S_1..S_n, C, E0, h_1..h_n], 2개 분자."""
    base_s = [1.0, 0.6, 0.3, 0.2, 0.1, 0.05][:n_mode]
    base_h = [0.18, 0.15, 0.10, 0.09, 0.08, 0.07][:n_mode]
    row_a = base_s + [0.08, 2.30] + base_h
    row_b = [v * 1.1 for v in base_s] + [0.09, 2.20] + [v * 0.9 for v in base_h]
    return torch.tensor([row_a, row_b], dtype=torch.float32)


def _gmm_preds() -> torch.Tensor:
    """[a2, a3, b1, b2, b3, c1, c2, c3], 2개 분자."""
    return torch.tensor(
        [
            [0.50, 0.30, 2.30, 2.15, 2.00, 0.08, 0.10, 0.12],
            [0.40, 0.20, 2.25, 2.10, 1.95, 0.09, 0.11, 0.13],
        ],
        dtype=torch.float32,
    )


def _naive_preds() -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.rand(2, NUM_WAVELENGTHS, generator=generator) * 3.0 - 1.0


def _write_and_read_intensity(tmp_path, preds, spectrum_type, **kwargs) -> torch.Tensor:
    """save_spectrum으로 CSV를 쓰고 Intensity 열을 (분자수, 800) 텐서로 읽는다."""
    csv_path = tmp_path / "p_spec.csv"
    ids = [f"mol_{i}" for i in range(preds.shape[0])]
    save_spectrum(preds, ids, str(csv_path), spectrum_type, **kwargs)
    df = pd.read_csv(csv_path)
    rows = [[float(v) for v in text.split()] for text in df["Intensity"]]
    return torch.tensor(rows, dtype=torch.float32)


# --- 특성화 테스트: 현재(리팩토링 전) save_spectrum 출력 고정 -------------------------


def test_save_spectrum_FC_CSV가_리팩토링_전과_동일하다(tmp_path):
    intensity = _write_and_read_intensity(tmp_path, _fc_preds(), "FC", kernel_kind="gaussian", beta=2.0)

    assert_matches_golden("save_spectrum_fc_intensity", intensity)


def test_save_spectrum_GMM_CSV가_리팩토링_전과_동일하다(tmp_path):
    intensity = _write_and_read_intensity(tmp_path, _gmm_preds(), "GMM")

    assert_matches_golden("save_spectrum_gmm_intensity", intensity)


def test_save_spectrum_Naive_CSV가_리팩토링_전과_동일하다(tmp_path):
    intensity = _write_and_read_intensity(tmp_path, _naive_preds(), "Naive")

    assert_matches_golden("save_spectrum_naive_intensity", intensity)


def test_save_spectrum_미지원_타입은_Undefined_spectrum_type_예외를_낸다(tmp_path):
    with pytest.raises(Exception, match="Undefined spectrum type"):
        save_spectrum(_fc_preds(), ["a", "b"], str(tmp_path / "x.csv"), "XYZ")


# --- 신규 인터페이스 테스트: spectrum.reconstruct (RED) ------------------------------


def test_FC_곡선복원은_save_spectrum_CSV값과_일치한다(tmp_path):
    from spectrum.reconstruct import reconstruct_spectrum

    preds = _fc_preds()
    expected = _write_and_read_intensity(tmp_path, preds, "FC", kernel_kind="gaussian", beta=2.0)

    curves = reconstruct_spectrum(preds, "FC", kernel_kind="gaussian", beta=2.0)

    assert curves.shape == (2, NUM_WAVELENGTHS)
    assert torch.allclose(curves, expected, atol=1e-6)  # CSV는 소수점 6자리로 기록됨


def test_GMM_곡선복원은_save_spectrum_CSV값과_일치한다(tmp_path):
    from spectrum.reconstruct import reconstruct_spectrum

    preds = _gmm_preds()
    expected = _write_and_read_intensity(tmp_path, preds, "GMM")

    curves = reconstruct_spectrum(preds, "GMM")

    assert curves.shape == (2, NUM_WAVELENGTHS)
    assert torch.allclose(curves, expected, atol=1e-6)


def test_Naive_곡선복원은_save_spectrum_CSV값과_일치한다(tmp_path):
    from spectrum.reconstruct import reconstruct_spectrum

    preds = _naive_preds()
    expected = _write_and_read_intensity(tmp_path, preds, "Naive")

    curves = reconstruct_spectrum(preds, "Naive")

    assert curves.shape == (2, NUM_WAVELENGTHS)
    assert torch.allclose(curves, expected, atol=1e-6)


def test_FC_n_mode가_2일때도_곡선을_복원한다(tmp_path):
    from spectrum.reconstruct import reconstruct_spectrum

    preds = _fc_preds(n_mode=2)  # 열 수 6 = 2*2+2
    expected = _write_and_read_intensity(tmp_path, preds, "FC", kernel_kind="gaussian", beta=2.0)

    curves = reconstruct_spectrum(preds, "FC", kernel_kind="gaussian", beta=2.0)

    assert curves.shape == (2, NUM_WAVELENGTHS)
    assert torch.allclose(curves, expected, atol=1e-6)


def test_파장_격자는_400에서_800nm까지_0점5nm_간격_800점이다():
    from spectrum.reconstruct import wavelength_grid_nm

    grid = torch.as_tensor(wavelength_grid_nm(), dtype=torch.float32)

    assert grid.shape == (NUM_WAVELENGTHS,)
    assert grid[0].item() == 400.0
    assert grid[-1].item() == 799.5
    assert torch.allclose(grid[1:] - grid[:-1], torch.full((NUM_WAVELENGTHS - 1,), 0.5))


def test_미지원_spectrum_type은_예외를_낸다():
    from spectrum.reconstruct import reconstruct_spectrum

    with pytest.raises(Exception, match="Undefined spectrum type"):
        reconstruct_spectrum(_fc_preds(), "XYZ")
