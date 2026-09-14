"""Plan.md Step 8 [구조 이동 — 최종 학습 결과 불변]: train.py의 서브프로세스 커맨드
조립 로직을 os.system() 호출과 분리된, 테스트 가능한 순수 함수로 뽑아낸다.

동시에 CLAUDE.md에 기록된 알려진 CPU 이슈(train_Geoformer.py는 --accelerator
기본값이 gpu라서, train.py가 이 플래그를 넘기지 않으면 CPU 환경에서
train.py --base-model Geoformer가 실패한다)를 해결한다: GPU가 있을 때는 legacy와
완전히 동일한 커맨드를 만들고(GPU 동작 불변, G1), GPU가 없을 때만
"--accelerator cpu --ndevices 1"을 추가한다 (CLAUDE.md 제약 2의 명시된 예외).
"""

from train import build_command, resolve_split_npz


def test_resolve_split_npz_IrDB는_CV811_경로를_만든다():
    assert resolve_split_npz("IrDB", 0, 0) == "IrDB/raw/CV811/splits.0.0.npz"


def test_resolve_split_npz_IrDB_uff도_CV811_경로를_만든다():
    assert resolve_split_npz("IrDB_uff", 0, 0) == "IrDB_uff/raw/CV811/splits.0.0.npz"


def test_resolve_split_npz_IrDB_murcko는_CV_murcko_경로를_만든다():
    assert resolve_split_npz("IrDB_murcko", 0, 0) == "IrDB_murcko/raw/CV_murcko/splits.0.0.npz"


def test_resolve_split_npz_PtDB는_CV_10fold_경로를_만든다():
    assert resolve_split_npz("PtDB", 0, 0) == "PtDB/raw/CV_10fold/splits.0.0.npz"


def _args(base_model, spectrum_type="FC", batch_size=16, data_path="IrDB"):
    from types import SimpleNamespace
    return SimpleNamespace(base_model=base_model, spectrum_type=spectrum_type,
                            batch_size=batch_size, data_path=data_path)


def test_build_command_PaiNN은_legacy_커맨드와_동일하다():
    cmd = build_command(_args("PaiNN"), i_seed=0, i_fold=0, split_npz="IrDB/raw/CV811/splits.0.0.npz")

    assert cmd == (
        "python -m train_PaiNN --spectrum-type FC --output-dir results_PaiNN/0/0 "
        "--split-index-npz IrDB/raw/CV811/splits.0.0.npz --seed 0 --batch-size 16 --data-path IrDB"
    )


def test_build_command_Equiformer은_legacy_커맨드와_동일하다():
    cmd = build_command(_args("Equiformer"), i_seed=0, i_fold=0, split_npz="IrDB/raw/CV811/splits.0.0.npz")

    assert cmd == (
        "python -m train_Equiformer --spectrum-type FC --output-dir results_Equiformer/0/0 "
        "--split-index-npz IrDB/raw/CV811/splits.0.0.npz --seed 0 --batch-size 16 --data-path IrDB"
    )


def test_build_command_Geoformer_GPU에서는_legacy_커맨드와_동일하다(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    cmd = build_command(_args("Geoformer"), i_seed=0, i_fold=0, split_npz="IrDB/raw/CV811/splits.0.0.npz")

    assert cmd == (
        "python -m train_Geoformer --conf geoformer/examples/FC.yml --log-dir results_Geoformer/0/0 "
        "--seed 0 --splits IrDB/raw/CV811/splits.0.0.npz --batch-size 16 --dataset-root IrDB"
    )


def test_build_command_Geoformer_CPU에서는_accelerator_cpu가_추가된다(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    cmd = build_command(_args("Geoformer"), i_seed=0, i_fold=0, split_npz="IrDB/raw/CV811/splits.0.0.npz")

    assert cmd == (
        "python -m train_Geoformer --conf geoformer/examples/FC.yml --log-dir results_Geoformer/0/0 "
        "--seed 0 --splits IrDB/raw/CV811/splits.0.0.npz --batch-size 16 --dataset-root IrDB "
        "--accelerator cpu --ndevices 1"
    )
