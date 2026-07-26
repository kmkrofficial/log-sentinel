"""Immutable catalog for provisioned LogSentinel datasets and model assets."""

from dataclasses import dataclass
from typing import Literal


ZENODO_RECORD_ID = "8196385"
ZENODO_CONTENT_URL = "https://zenodo.org/api/records/{record_id}/files/{filename}/content"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    archive_filename: str
    archive_format: Literal["zip", "tar.gz"]
    expected_size: int
    md5: str
    raw_file_patterns: tuple[str, ...]
    required_raw_filenames: tuple[str, ...]
    preparation_strategy: Literal["bgl_window", "hdfs_block", "thunderbird_window", "hadoop_detect"]

    @property
    def download_url(self) -> str:
        return ZENODO_CONTENT_URL.format(record_id=ZENODO_RECORD_ID, filename=self.archive_filename)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    folder_name: str
    requires_hf_token: bool


DATASET_SPECS: dict[str, DatasetSpec] = {
    "BGL": DatasetSpec(
        dataset_id="BGL",
        display_name="BlueGene/L",
        archive_filename="BGL.zip",
        archive_format="zip",
        expected_size=57_489_019,
        md5="4452953c470f2d95fcb32d5f6e733f7a",
        raw_file_patterns=("BGL.log",),
        required_raw_filenames=("BGL.log",),
        preparation_strategy="bgl_window",
    ),
    "HDFS_v1": DatasetSpec(
        dataset_id="HDFS_v1",
        display_name="HDFS v1",
        archive_filename="HDFS_v1.zip",
        archive_format="zip",
        expected_size=186_645_559,
        md5="76a24b4d9a6164d543fb275f89773260",
        raw_file_patterns=("HDFS.log", "anomaly_label.csv"),
        required_raw_filenames=("HDFS.log", "anomaly_label.csv"),
        preparation_strategy="hdfs_block",
    ),
    "Hadoop": DatasetSpec(
        dataset_id="Hadoop",
        display_name="Hadoop",
        archive_filename="Hadoop.zip",
        archive_format="zip",
        expected_size=3_416_419,
        md5="34e28a9943704fd54933e2b455829fcc",
        raw_file_patterns=("*.log", "*.txt", "*.csv"),
        required_raw_filenames=(),
        preparation_strategy="hadoop_detect",
    ),
    "Thunderbird": DatasetSpec(
        dataset_id="Thunderbird",
        display_name="Thunderbird",
        archive_filename="Thunderbird.tar.gz",
        archive_format="tar.gz",
        expected_size=2_016_100_298,
        md5="0891b048df2919dc78c99c4428686b44",
        raw_file_patterns=("Thunderbird.log",),
        required_raw_filenames=("Thunderbird.log",),
        preparation_strategy="thunderbird_window",
    ),
}


MODEL_SPECS: dict[str, ModelSpec] = {
    "encoder": ModelSpec(
        key="encoder",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        folder_name="all-MiniLM-L6-v2",
        requires_hf_token=False,
    ),
    "llama": ModelSpec(
        key="llama",
        model_id="meta-llama/Llama-3.2-1B",
        folder_name="Llama-3.2-1B",
        requires_hf_token=True,
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_id]
    except KeyError as error:
        supported = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unsupported dataset '{dataset_id}'. Supported datasets: {supported}.") from error


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model key '{model_key}'. Supported models: {supported}.") from error