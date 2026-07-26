import hashlib
import io
import tarfile
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mlcore.prepareData import pipeline
from mlcore.setup_catalog import get_dataset_spec
from mlcore import setup_manager
from mlcore.utils import runtime_compat


class _RangeRequestHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return

        start_text, end_text = range_header.removeprefix("bytes=").split("-", maxsplit=1)
        start = int(start_text)
        end = int(end_text) if end_text else len(self.payload) - 1
        body = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class _FlakyRangeResponse:
    def __init__(self, payload, start, end, interrupt):
        self.payload = payload[start : end + 1]
        self.start = start
        self.end = end
        self.interrupt = interrupt
        self.offset = 0
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getcode(self):
        return 206

    def read(self, size):
        if self.interrupt and self.offset:
            raise TimeoutError("simulated range timeout")
        if self.offset >= len(self.payload):
            return b""
        read_size = size
        if self.interrupt:
            read_size = min(read_size, max(1, len(self.payload) // 2))
        block = self.payload[self.offset : self.offset + read_size]
        self.offset += len(block)
        return block


class SetupManagerTests(unittest.TestCase):
    def test_bitsandbytes_selects_compatible_same_major_cuda_binary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_dir = Path(temporary_directory)
            (package_dir / "libbitsandbytes_cuda130.so").touch()
            (package_dir / "libbitsandbytes_cuda129.so").touch()

            with patch.dict("os.environ", {}, clear=True):
                binary = runtime_compat.resolve_bitsandbytes_cuda_binary("13.2", package_dir)

            self.assertIsNotNone(binary)
            self.assertEqual(binary.tag, "130")
            self.assertEqual(binary.source, "compatible")

    def test_torch_compile_is_skipped_when_optional_toolchain_is_unavailable(self):
        model = object()
        logs = []
        readiness = runtime_compat.TorchCompileReadiness(False, "Python.h is unavailable.")

        with patch.object(runtime_compat, "torch_compile_readiness", return_value=readiness):
            result = runtime_compat.maybe_compile_model(model, logs.append)

        self.assertIs(result, model)
        self.assertEqual(logs, ["Skipping optional torch.compile acceleration: Python.h is unavailable."])

    def test_safe_extracts_discover_required_files_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "BGL.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("archive-root/BGL.log", "- sample log\n")

            extracted_dir = root / "extracted"
            setup_manager.safe_extract_zip(archive_path, extracted_dir)
            discovered = setup_manager.discover_raw_files(extracted_dir, get_dataset_spec("BGL"))
            self.assertEqual(discovered["BGL.log"], "archive-root/BGL.log")

            unsafe_archive = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe_archive, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(setup_manager.SetupError):
                setup_manager.safe_extract_zip(unsafe_archive, root / "unsafe")

    def test_safe_extracts_tar_archives(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "Thunderbird.tar.gz"
            payload = b"- sample Thunderbird log\n"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("Thunderbird/Thunderbird.log")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            extracted_dir = root / "extracted"
            setup_manager.safe_extract_tar(archive_path, extracted_dir)
            discovered = setup_manager.discover_raw_files(extracted_dir, get_dataset_spec("Thunderbird"))
            self.assertEqual(discovered["Thunderbird.log"], "Thunderbird/Thunderbird.log")

    def test_download_with_resume_promotes_verified_completed_partial(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            payload = b"setup-download-fixture" * 1024
            source.write_bytes(payload)
            expected_md5 = hashlib.md5(payload).hexdigest()
            destination = root / "archive.bin"

            setup_manager.download_with_resume(source.as_uri(), destination, len(payload), expected_md5=expected_md5)
            self.assertEqual(destination.read_bytes(), payload)

            destination.unlink()
            destination.with_name(".archive.bin.part").write_bytes(payload)
            setup_manager.download_with_resume(source.as_uri(), destination, len(payload), expected_md5=expected_md5)
            self.assertEqual(destination.read_bytes(), payload)

    def test_parallel_download_reuses_serial_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = b"parallel-range-fixture" * 16_384
            expected_md5 = hashlib.md5(payload).hexdigest()
            _RangeRequestHandler.payload = payload
            server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeRequestHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            original_threshold = setup_manager.PARALLEL_DOWNLOAD_THRESHOLD_BYTES
            original_min_range = setup_manager.PARALLEL_DOWNLOAD_MIN_RANGE_BYTES
            original_connections = setup_manager.DEFAULT_DOWNLOAD_CONNECTIONS
            try:
                setup_manager.PARALLEL_DOWNLOAD_THRESHOLD_BYTES = 1
                setup_manager.PARALLEL_DOWNLOAD_MIN_RANGE_BYTES = 16 * 1024
                setup_manager.DEFAULT_DOWNLOAD_CONNECTIONS = 4
                destination = root / "parallel.bin"
                destination.with_name(".parallel.bin.part").write_bytes(payload[:25_000])
                setup_manager.download_with_resume(
                    f"http://127.0.0.1:{server.server_port}/archive.bin",
                    destination,
                    len(payload),
                    expected_md5=expected_md5,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertFalse(destination.with_name(".parallel.bin.ranges").exists())
            finally:
                setup_manager.PARALLEL_DOWNLOAD_THRESHOLD_BYTES = original_threshold
                setup_manager.PARALLEL_DOWNLOAD_MIN_RANGE_BYTES = original_min_range
                setup_manager.DEFAULT_DOWNLOAD_CONNECTIONS = original_connections
                server.shutdown()
                server.server_close()

    def test_range_download_retries_from_saved_partial_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = b"range-retry-fixture" * 256
            segment_path = Path(temporary_directory) / "segment.part"
            requested_starts = []

            def open_response(url, start=None, end=None):
                requested_starts.append(start)
                return _FlakyRangeResponse(payload, start, end, interrupt=len(requested_starts) == 1)

            with patch.object(setup_manager, "_open_download_response", side_effect=open_response):
                setup_manager._download_range_segment(
                    "https://example.invalid/archive",
                    0,
                    len(payload) - 1,
                    segment_path,
                    {"downloaded": 0, "last_reported_percent": -1},
                    threading.Lock(),
                    None,
                    len(payload),
                    0.0,
                    1.0,
                    "archive",
                )

            self.assertEqual(segment_path.read_bytes(), payload)
            self.assertEqual(requested_starts, [0, len(payload) // 2])


class _NonBooleanModel:
    def __bool__(self):
        raise TypeError("unexpected truthiness")

    def __len__(self):
        raise TypeError("unexpected length")


class ControllerCleanupTests(unittest.TestCase):
    def _assert_cleanup_does_not_evaluate_model_truthiness(self, controller_type, cuda_availability_path):
        model = _NonBooleanModel()
        controller = controller_type.__new__(controller_type)
        controller.model = model

        with patch(cuda_availability_path, return_value=False):
            controller._cleanup()
            self.assertIsNone(controller.model)
            controller._cleanup(model_to_clean=model)

    def test_training_cleanup_does_not_evaluate_model_truthiness(self):
        from mlcore.engine.training_controller import TrainingController

        self._assert_cleanup_does_not_evaluate_model_truthiness(
            TrainingController,
            "mlcore.engine.training_controller.torch.cuda.is_available",
        )

    def test_inference_cleanup_does_not_evaluate_model_truthiness(self):
        from mlcore.engine.inference_controller import InferenceController

        self._assert_cleanup_does_not_evaluate_model_truthiness(
            InferenceController,
            "mlcore.engine.inference_controller.torch.cuda.is_available",
        )


class PreparationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache = self.root / "cache"
        self.original_pipeline_data_dir = pipeline.DATA_DIR
        self.original_setup_data_dir = setup_manager.DATA_DIR
        self.original_cache_dir = setup_manager.DATA_CACHE_DIR
        self.original_downloads_dir = setup_manager.DOWNLOADS_DIR
        self.original_state_path = setup_manager.SETUP_STATE_PATH
        pipeline.DATA_DIR = self.root
        setup_manager.DATA_DIR = self.root
        setup_manager.DATA_CACHE_DIR = self.cache
        setup_manager.DOWNLOADS_DIR = self.cache / "downloads"
        setup_manager.SETUP_STATE_PATH = self.cache / "setup-state.json"

    def tearDown(self):
        pipeline.DATA_DIR = self.original_pipeline_data_dir
        setup_manager.DATA_DIR = self.original_setup_data_dir
        setup_manager.DATA_CACHE_DIR = self.original_cache_dir
        setup_manager.DOWNLOADS_DIR = self.original_downloads_dir
        setup_manager.SETUP_STATE_PATH = self.original_state_path
        self.temporary_directory.cleanup()

    def _assert_output_contract(self, dataset_id):
        for split_name in ("train", "validation", "test"):
            output_path = self.root / dataset_id / f"{split_name}.csv"
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.read_text(encoding="utf-8").splitlines()[0], "Content,Label")

    def test_bgl_hdfs_and_thunderbird_generate_split_contracts(self):
        bgl_dir = self.root / "BGL"
        bgl_dir.mkdir()
        with (bgl_dir / "BGL.log").open("w", encoding="latin-1") as log_file:
            for index in range(300):
                label = "ALERT" if index % 90 == 0 else "-"
                log_file.write(f"{label} {index} 2005.06.03 C1 10:00 C2 A B INFO BGL message {index}\n")
        bgl_result = pipeline.prepare_dataset("BGL")
        self.assertEqual(sum(bgl_result["sequence_counts"].values()), 2)
        self._assert_output_contract("BGL")

        hdfs_raw = self.root / "HDFS_v1" / "raw"
        hdfs_raw.mkdir(parents=True)
        (hdfs_raw / "anomaly_label.csv").write_text("BlockId,Label\nblk_1,Normal\nblk_2,Anomaly\n", encoding="utf-8")
        (hdfs_raw / "HDFS.log").write_text(
            "0101 000001 1 INFO DataNode: reading blk_1\n"
            "0101 000002 1 INFO DataNode: writing blk_2\n"
            "0101 000003 1 INFO DataNode: done blk_1 blk_2\n",
            encoding="latin-1",
        )
        hdfs_result = pipeline.prepare_dataset("HDFS_v1")
        self.assertEqual(sum(hdfs_result["sequence_counts"].values()), 2)
        self._assert_output_contract("HDFS_v1")

        thunderbird_raw = self.root / "Thunderbird" / "raw"
        thunderbird_raw.mkdir(parents=True)
        with (thunderbird_raw / "Thunderbird.log").open("w", encoding="latin-1") as log_file:
            for index in range(300):
                label = "ALERT" if index % 100 == 0 else "-"
                log_file.write(f"{label} {index} 2005.01.01 host Jan 1 00:00 addr Thunderbird message {index}\n")
        thunderbird_result = pipeline.prepare_dataset(
            "Thunderbird",
            options={"start_line": 0, "end_line": 300, "window_size": 100, "step_size": 100, "oversampling_factor": 2},
        )
        self.assertEqual(sum(thunderbird_result["sequence_counts"].values()), 6)
        self._assert_output_contract("Thunderbird")

    def test_hadoop_without_an_approved_label_source_is_blocked(self):
        raw_dir = self.root / "Hadoop" / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "Hadoop.log").write_text("unlabeled source message\n", encoding="utf-8")

        status = pipeline.get_preparation_status("Hadoop")
        self.assertFalse(status["eligible"])
        self.assertIn("No approved Hadoop", status["blocker"])
        with self.assertRaises(pipeline.PreparationBlockedError):
            pipeline.prepare_dataset("Hadoop")


if __name__ == "__main__":
    unittest.main()