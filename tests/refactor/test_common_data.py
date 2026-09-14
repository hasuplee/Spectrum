"""Plan.md Step 5 [구조 이동]: train_PaiNN.py/train_Equiformer.py에 중복되어 있던
dataset 선택 + split 로딩 + dataset 통계 계산 + DataLoader 생성 로직을
common/data.py로 이동한다. 실제 IrDB 데이터셋(작은 4개 인덱스만 사용)을 대상으로
동작을 확인한다.
"""

from types import SimpleNamespace

import numpy as np

from common.data import build_dataloaders, load_dataset_splits


def _tiny_args(tmp_path, targets, data_path="IrDB"):
    npz_path = tmp_path / "splits.npz"
    np.savez(npz_path, idx_train=np.array([0, 1]), idx_val=np.array([2]), idx_test=np.array([3]))
    return SimpleNamespace(
        data_path=data_path,
        targets=targets,
        split_index_npz=str(npz_path),
        standardize=True,
        seed=0,
        distributed=False,
        batch_size=2,
        workers=0,
        pin_mem=False,
    )


def test_load_dataset_splits는_IrDB에서_train_val_test를_분리한다(tmp_path):
    args = _tiny_args(tmp_path, targets=["S1", "S2", "S3", "C", "E0", "h1", "h2", "h3"])

    train_dataset, val_dataset, test_dataset, task_mean, task_std = load_dataset_splits(args)

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert len(test_dataset) == 1
    assert len(task_mean) == 8
    assert len(task_std) == 8


def test_load_dataset_splits는_standardize_false면_mean0_std1이다(tmp_path):
    args = _tiny_args(tmp_path, targets=["S1", "S2"])
    args.standardize = False

    _, _, _, task_mean, task_std = load_dataset_splits(args)

    assert task_mean == [0.0, 0.0]
    assert task_std == [1.0, 1.0]


def test_load_dataset_splits는_알수없는_data_path에서_예외를_낸다(tmp_path):
    args = _tiny_args(tmp_path, targets=["S1"], data_path="Unknown")

    try:
        load_dataset_splits(args)
        assert False, "예외가 발생해야 한다"
    except Exception as e:
        assert "IrDB" in str(e) or "PtDB" in str(e)


def test_build_dataloaders는_batch_size대로_배치를_만든다(tmp_path):
    args = _tiny_args(tmp_path, targets=["S1", "S2", "S3", "C", "E0", "h1", "h2", "h3"])
    train_dataset, val_dataset, test_dataset, _, _ = load_dataset_splits(args)

    train_loader, val_loader, test_loader = build_dataloaders(args, train_dataset, val_dataset, test_dataset)

    train_batch = next(iter(train_loader))
    assert train_batch.num_graphs == 2
    val_batch = next(iter(val_loader))
    assert val_batch.num_graphs == 1
