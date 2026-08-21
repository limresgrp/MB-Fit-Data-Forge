"""CLI for recording workflow-stage inputs, outputs, and parameters."""

import argparse
import json

from dataforge.src.stage_metadata import record_stage


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--root", required=True)
    record_parser.add_argument("--stage", required=True)
    record_parser.add_argument("--status", default="completed")
    record_parser.add_argument("--inputs", nargs="*", default=[])
    record_parser.add_argument("--outputs", nargs="*", default=[])
    record_parser.add_argument("--parameters-json", default="{}")
    record_parser.add_argument("--command", default=None)
    parsed = parser.parse_args(args)

    if parsed.mode == "record":
        record = record_stage(
            dataset_root=parsed.root,
            stage=parsed.stage,
            status=parsed.status,
            inputs=parsed.inputs,
            outputs=parsed.outputs,
            parameters=json.loads(parsed.parameters_json),
            command=parsed.command,
        )
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
