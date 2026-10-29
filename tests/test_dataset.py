"""Exercise repository routing and selective extraction with real local archives."""

import json
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from physical_ai_av.dataset import DEFAULT_SPLIT_DEPRECATION_WARNING, PhysicalAIAVDatasetInterface

REVISION = "a" * 40


@pytest.fixture
def dataset_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., PhysicalAIAVDatasetInterface]:
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.is_offline_mode", lambda: True, raising=False)

    def create(
        repo_id: str = "nvidia/PhysicalAI-Autonomous-Vehicles",
        chunk: int = 1,
        payload: bytes = b"original",
        feature_sources: dict[str, PhysicalAIAVDatasetInterface] | None = None,
        clip_ids: tuple[str, ...] = ("clip",),
        legacy_split: str | None = None,
        split_files: dict[str, pd.DataFrame] | None = None,
        **kwargs,
    ) -> PhysicalAIAVDatasetInterface:
        cache_dir = tmp_path / repo_id.split("/")[-1]
        root = cache_dir / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / REVISION
        (root / "metadata").mkdir(parents=True)
        pd.DataFrame(
            [
                (
                    "payload",
                    "labels",
                    "chunk_{chunk_id}.zip",
                    json.dumps({"data": "{clip_id}/data.bin"}),
                ),
                ("direct", "labels", "chunk_{chunk_id}.parquet", None),
            ],
            columns=["feature", "directory", "chunk_path", "clip_files_in_zip"],
        ).to_csv(root / "features.csv", index=False)
        clip_index = pd.DataFrame({"chunk": chunk}, index=pd.Index(clip_ids, name="clip_id"))
        if legacy_split is not None:
            clip_index["split"] = legacy_split
        clip_index.to_parquet(root / "clip_index.parquet")
        pd.DataFrame({"payload": True, "direct": True}, index=clip_index.index).to_parquet(
            root / "metadata/feature_presence.parquet"
        )
        pd.DataFrame(index=clip_index.index).to_parquet(root / "metadata/data_collection.parquet")
        for name, labels in (split_files or {}).items():
            (root / "splits").mkdir(exist_ok=True)
            labels.to_parquet(root / f"splits/{name}.parquet")
        with zipfile.ZipFile(root / f"chunk_{chunk}.zip", "w") as archive:
            for clip_id in clip_ids:
                archive.writestr(f"{clip_id}/data.bin", payload)
        return PhysicalAIAVDatasetInterface(
            REVISION,
            repo_id=repo_id,
            cache_dir=cache_dir,
            feature_sources=feature_sources,
            **kwargs,
        )

    return create


def _split_file(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.Index(["clip"], name="clip_id"))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.name.startswith("physical_ai_av")]


def test_selected_source_owns_chunk_cache_and_decoder(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
) -> None:
    """Read and extract the selected source's clip rather than the requester's."""
    source = dataset_factory(repo_id="example/source", chunk=8, payload=b"source")
    dataset = dataset_factory(feature_sources={"payload": source})
    decoded = dataset.get_clip_feature("clip", "payload")
    assert decoded["data"].read() == b"source"
    path = dataset.get_clip_feature_paths("clip", "payload", members="data")["data"]
    assert path.read_bytes() == b"source"
    assert path.is_relative_to(source.cache_dir)


def test_download_groups_features_by_source_chunk(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download each feature from its owning source, including a forwarded source."""
    source = dataset_factory(repo_id="example/source", chunk=8)
    forwarder = dataset_factory(repo_id="example/forwarder", feature_sources={"payload": source})
    forwarder.features.ALL.remove("payload")
    dataset = dataset_factory(feature_sources={"payload": forwarder})
    local_download = Mock()
    source_download = Mock()
    monkeypatch.setattr(dataset, "download_files", local_download)
    monkeypatch.setattr(source, "download_files", source_download)
    dataset.download_clip_features("clip", ["direct", "payload"], dry_run=True)
    local_download.assert_called_once_with(["chunk_1.parquet"], dry_run=True)
    source_download.assert_called_once_with(["chunk_8.zip"], dry_run=True)


def test_default_split_logs_deprecation_warning(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Read `splits/default.parquet` and warn once that the default split is deprecated."""
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory(split_files={"default": _split_file(split=["val"])})
    assert dataset.clip_index["split"].tolist() == ["val"]
    assert _warnings(caplog) == [DEFAULT_SPLIT_DEPRECATION_WARNING]


def test_legacy_default_split_logs_deprecation_warning(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn as well when the deprecated split is the `split` column of an older clip index."""
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory(legacy_split="train")
    assert dataset.clip_index["split"].tolist() == ["train"]
    assert _warnings(caplog) == [DEFAULT_SPLIT_DEPRECATION_WARNING]


def test_named_split_adds_its_columns_without_warning(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attach every column of a partial split file to `clip_index`; unlisted clips get NaN."""
    labels = _split_file(split=["test"], t0=[[2_000_000, 5_000_000]])
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory(
            clip_ids=("clip", "other"), split_files={"anchors": labels}, split_name="anchors"
        )
    assert dataset.clip_index.loc["clip", "split"] == "test"
    assert list(dataset.clip_index.loc["clip", "t0"]) == [2_000_000, 5_000_000]
    assert dataset.clip_index.loc["other", ["split", "t0"]].isna().all()
    assert _warnings(caplog) == []


def test_split_name_none_reads_no_split_file(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip the split file, and with it the deprecation warning, when `split_name` is None."""
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory(
            split_files={"default": _split_file(split=["val"])}, split_name=None
        )
    assert "split" not in dataset.clip_index.columns
    assert _warnings(caplog) == []


def test_split_name_none_keeps_legacy_split_column(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leave an older clip index's own `split` column untouched, without the warning."""
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory(legacy_split="train", split_name=None)
    assert dataset.clip_index["split"].tolist() == ["train"]
    assert _warnings(caplog) == []


def test_unavailable_default_split_adds_no_column(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without `splits/default.parquet` and a legacy column there is no split and no warning."""
    with caplog.at_level(logging.WARNING, logger="physical_ai_av"):
        dataset = dataset_factory()
    assert "split" not in dataset.clip_index.columns
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "labels, match",
    [
        (_split_file(t0=[[2_000_000]]), "no `split` column"),
        (_split_file(split=["test"], chunk=[1]), "already exist"),
        (
            pd.DataFrame({"split": ["test"]}, index=pd.Index(["unknown"], name="clip_id")),
            "absent from",
        ),
        (
            pd.DataFrame({"split": ["test", "val"]}, index=pd.Index(["clip"] * 2, name="clip_id")),
            "more than once",
        ),
    ],
)
def test_invalid_split_file_raises(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    labels: pd.DataFrame,
    match: str,
) -> None:
    """Reject split files without labels, with clashing columns, or with unknown/repeated clips."""
    with pytest.raises(ValueError, match=match):
        dataset_factory(split_files={"bad": labels}, split_name="bad")
