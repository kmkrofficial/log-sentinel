import argparse
import json

from mlcore.engine.inference_controller import InferenceController
from mlcore.engine.training_controller import TrainingController


def _parse_hyperparameter_overrides(raw_value: str | None) -> dict:
    if not raw_value:
        return {}

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("Hyperparameter overrides must decode to a JSON object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LogSentinel ML core tasks from the command line.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training workflow.")
    train_parser.add_argument("--dataset-name", required=True, help="Dataset name inside mlcore/datasets.")
    train_parser.add_argument("--is-test-run", action="store_true", help="Run a reduced quick-test workflow.")
    train_parser.add_argument("--test-run-percentage", type=float, default=0.3, help="Sampling fraction for quick test runs.")
    train_parser.add_argument(
        "--hyperparameters-json",
        default="",
        help="JSON object with hyperparameter overrides, for example '{\"micro_batch_size\": 16}'.",
    )

    inference_parser = subparsers.add_parser("inference", help="Run an inference workflow.")
    inference_parser.add_argument("--model-run-path", required=True, help="Path to a trained run directory or its output_model folder.")
    inference_parser.add_argument("--dataset-name", required=True, help="Dataset name inside mlcore/datasets.")
    inference_parser.add_argument("--is-test-run", action="store_true", help="Run a reduced quick-test workflow.")
    inference_parser.add_argument("--test-run-percentage", type=float, default=0.3, help="Sampling fraction for quick test runs.")
    inference_parser.add_argument("--manual-nickname", default=None, help="Optional nickname prefix for the inference execution directory.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        controller = TrainingController(
            dataset_name=args.dataset_name,
            callback=print,
            is_test_run=args.is_test_run,
            test_run_percentage=args.test_run_percentage,
        )

        overrides = _parse_hyperparameter_overrides(args.hyperparameters_json)
        if overrides:
            controller.hp.update(overrides)
            controller.num_workers = controller.hp.get("dataloader_num_workers", controller.num_workers)

        result = controller.run()
    else:
        controller = InferenceController(
            model_run_path=args.model_run_path,
            dataset_name=args.dataset_name,
            callback=print,
            is_test_run=args.is_test_run,
            test_run_percentage=args.test_run_percentage,
            manual_nickname=args.manual_nickname,
        )
        result = controller.run_inference()

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()