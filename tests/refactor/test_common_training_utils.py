"""Plan.md Step 2 [구조 이동]: train_PaiNN.py/train_Equiformer.py에 중복되어 있던
load_split_from_npz / save_pred / warmup_exponential_decay / 스펙트럼 타겟 목록 생성
로직을 common/training_utils.py로 옮긴 뒤, 그 새 위치에서 정확히 동일하게 동작하는지
검증한다. 로직 자체는 옮기기 전 두 파일에서 토씨 하나 다르지 않았으므로(레거시 오라클),
여기서는 "새 위치에 존재하고 레거시와 같은 값을 낸다"만 확인한다.
"""

import numpy as np
import pandas as pd
import torch

from common.training_utils import (
    build_spectrum_targets,
    load_split_from_npz,
    save_pred,
    warmup_exponential_decay,
)


def test_load_split_from_npz는_train_val_test_인덱스를_리스트로_반환한다(tmp_path):
    npz_path = tmp_path / "splits.npz"
    np.savez(
        npz_path,
        idx_train=np.array([0, 1, 2]),
        idx_val=np.array([3]),
        idx_test=np.array([4, 5]),
    )

    idx_train, idx_val, idx_test = load_split_from_npz(str(npz_path))

    assert idx_train == [0, 1, 2]
    assert idx_val == [3]
    assert idx_test == [4, 5]


def test_load_split_from_npz는_파일이_없으면_예외를_낸다(tmp_path):
    missing_path = tmp_path / "missing.npz"

    try:
        load_split_from_npz(str(missing_path))
        assert False, "예외가 발생해야 한다"
    except Exception as e:
        assert "is not exist" in str(e)


def test_save_pred는_molecule_id_컬럼을_맨_앞에_추가한_csv를_만든다(tmp_path):
    preds = np.array([[0.1, 0.2], [0.3, 0.4]])
    ids = ["mol_a", "mol_b"]
    out_path = tmp_path / "pred.csv"

    save_pred(preds, ids, str(out_path), col_names=["S1", "S2"])

    df = pd.read_csv(out_path)
    assert list(df.columns) == ["molecule_id", "S1", "S2"]
    assert list(df["molecule_id"]) == ids
    assert df["S1"].tolist() == [0.1, 0.3]


def test_warmup_exponential_decay는_warmup_시작에서_warmup_factor이다():
    class H:
        lr_warmup_steps = 100
        lr_warmup_factor = 0.1
        decay_rate = 0.1
        decay_step = 10000

    assert warmup_exponential_decay(0, H()) == 0.1


def test_warmup_exponential_decay는_warmup_이후_decay만_적용된다():
    class H:
        lr_warmup_steps = 100
        lr_warmup_factor = 0.1
        decay_rate = 0.5
        decay_step = 1000

    value = warmup_exponential_decay(1000, H())

    assert value == 0.5 ** (1000 / 1000)


def test_build_spectrum_targets_FC_n_mode_3은_레거시_8개_타겟과_동일하다():
    targets, standardize = build_spectrum_targets("FC", n_mode=3)

    assert targets == ["S1", "S2", "S3", "C", "E0", "h1", "h2", "h3"]
    assert standardize is True


def test_build_spectrum_targets_FC_n_mode_2는_괄호_표기_타겟을_만든다():
    targets, standardize = build_spectrum_targets("FC", n_mode=2)

    assert targets == ["S1(2)", "S2(2)", "C(2)", "E0(2)", "h1(2)", "h2(2)"]
    assert standardize is True


def test_build_spectrum_targets_GMM은_8개_고정_타겟이다():
    targets, standardize = build_spectrum_targets("GMM", n_mode=3)

    assert targets == ["A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    assert standardize is True


def test_build_spectrum_targets_Naive는_y0에서_y799까지_800개이다():
    targets, standardize = build_spectrum_targets("Naive", n_mode=3)

    assert len(targets) == 800
    assert targets[0] == "y0" and targets[-1] == "y799"
    assert standardize is False


def test_build_spectrum_targets_알수없는_타입은_예외를_낸다():
    try:
        build_spectrum_targets("Unknown", n_mode=3)
        assert False, "예외가 발생해야 한다"
    except Exception as e:
        assert "Spectrum type Error" in str(e)
