"""Plan.md Step 3A [신규 인터페이스]: 모델별 입력 배치 구성 common.inference.build_batch.

build_batch가 아직 없으므로 실패해야 한다 (RED).
PaiNN/Equiformer는 PyG Batch, Geoformer는 GeoformerDataCollator가 만드는 dict가 입력이다.
새 함수 import는 각 테스트 안에서 하여 개별 실패로 확인한다.
"""

import pytest
import torch
from torch_geometric.data import Batch

from dataset.IrDB import IrDB
from tests.support.tiny_batches import make_tiny_geoformer_batch, make_tiny_pyg_batch

FC_TARGETS = ['S1', 'S2', 'S3', 'C', 'E0', 'h1', 'h2', 'h3']


@pytest.mark.parametrize("base_model", ["PaiNN", "Equiformer"])
def test_PaiNN과_Equiformer_배치는_PyG_Batch로_묶인다(base_model):
    from common.inference import build_batch

    original = make_tiny_pyg_batch()

    batch = build_batch(base_model, original.to_data_list())

    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert torch.equal(batch.pos, original.pos)
    assert torch.equal(batch.z, original.z)
    assert torch.equal(batch.batch, original.batch)


def test_Geoformer_배치는_z와_pos를_가진_dict이다():
    from common.inference import build_batch

    original = make_tiny_geoformer_batch()
    data_list = make_tiny_pyg_batch().to_data_list()

    batch = build_batch("Geoformer", data_list)

    assert isinstance(batch, dict)
    assert torch.equal(batch["z"], original["z"])
    assert torch.equal(batch["pos"], original["pos"])


def test_실제_IrDB_분자로_세_모델의_배치를_만들_수_있다():
    from common.inference import build_batch

    dataset = IrDB(root="IrDB", dataset_arg=FC_TARGETS)
    data_list = [dataset[0], dataset[1]]
    num_atoms = [d.z.shape[0] for d in data_list]

    for base_model in ("PaiNN", "Equiformer"):
        batch = build_batch(base_model, data_list)
        assert batch.num_graphs == 2
        assert batch.pos.shape == (sum(num_atoms), 3)

    batch = build_batch("Geoformer", data_list)
    assert batch["z"].shape == (2, max(num_atoms))
    assert batch["pos"].shape == (2, max(num_atoms), 3)


def test_지원하지_않는_모델이면_ValueError가_발생한다():
    from common.inference import build_batch

    with pytest.raises(ValueError, match="GPT"):
        build_batch("GPT", make_tiny_pyg_batch().to_data_list())
