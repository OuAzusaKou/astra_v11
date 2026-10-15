"""Validate the full client-to-server path with recorded real five-image samples."""
import argparse
import json
from pathlib import Path
import uuid

from .protocol import Client, request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="10.128.40.203")
    p.add_argument("--port", type=int, default=29573)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    client, session = Client(args.host, args.port), uuid.uuid4().hex
    rows = []
    try:
        for sample in json.loads(args.samples.read_text()):
            l3, l2 = sample["task"].removeprefix("L3: ").split("\nL2: ", 1)
            msg = request([(args.samples.parent / name).read_bytes() for name in sample["images"]],
                          l3, l2, session, sample["id"])
            result = client.infer(msg)
            if not result.get("valid"):
                raise ValueError("invalid generated progress: " + result.get("text", ""))
            rows.append({"sample_id": sample["id"], "gt_progress": sample["gt_progress"], **result})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
