"""Unified DexBench task selector; all profiles run the same inference loop."""

import argparse

from utils.client_config import REPO_ROOT, load_client_config


TASKS = ("1", "1-q20", "2", "3", "4")


def parse_task_args(argv=None):
    parser = argparse.ArgumentParser(
        description="DexBench ImageWAM mix client: select one task/model at startup.",
        epilog=("Additional arguments are forwarded to the shared inference loop, e.g. "
                "--camera-latency 0.200, --inference-mode sync, --post-chunk-sleep 1.5."),
        allow_abbrev=False,
    )
    parser.add_argument("--task", required=True, metavar="{" + ",".join(TASKS) + "}",
                        help="1/2/3/4 = MANO models; 1-q20 = task 1 Q20 full-data model.")
    parser.add_argument("--config", help="Override the selected task's client YAML.")
    parser.add_argument("--print-config", action="store_true",
                        help="Print selected YAML path for tunnel setup; no ROS or model connection.")
    args, remaining = parser.parse_known_args(argv)
    config_path = args.config or str(REPO_ROOT / "config" / f"client_dex_bench_mix_{args.task}.yaml")
    return args, config_path, remaining


def main(argv=None):
    args, config_path, remaining = parse_task_args(argv)
    if args.print_config:
        print(config_path)
        return
    config = load_client_config(config_path)
    forwarded = ["--config", config_path]
    instruction_flags = ("--instruction", "--task_prompt", "--instruction-sequence-file")
    if not any(arg.split("=", 1)[0] in instruction_flags for arg in remaining):
        forwarded += ["--instruction", config["instruction"]]
    print(f"[mix] task={args.task}, protocol={config['policy']['protocol']}, "
          f"server={config['policy']['host']}:{config['policy']['port']}")
    # Keep --help and --print-config usable on the server without ROS installed.
    from scripts.inference_leo.inference_multiview_leo_sync import main as run_inference
    run_inference(forwarded + remaining, use_mix_client=True)


if __name__ == "__main__":
    main()
