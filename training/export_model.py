"""Convert the merged model (training/merge_lora.py's output) to GGUF.

Shells out to a local llama.cpp checkout's conversion + quantization
tooling rather than reimplementing GGUF packing. Requires LLAMA_CPP_DIR to
point at a checkout of https://github.com/ggml-org/llama.cpp with
requirements installed and `llama-quantize` built (cmake --build ... --target
llama-quantize).

Output is the artifact that actually ships into the main app's Docker
Compose stack, served by a llama.cpp container (see docs/design.md).

Usage:
    LLAMA_CPP_DIR=/path/to/llama.cpp python training/export_model.py --config configs/serving.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/serving.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    merge_config = config["merge"]
    export_config = config["export"]

    llama_cpp_dir = Path(os.environ.get("LLAMA_CPP_DIR") or export_config["llama_cpp_dir"])
    if not llama_cpp_dir.exists():
        raise FileNotFoundError(
            f"llama.cpp checkout not found at {llama_cpp_dir}. "
            "Set LLAMA_CPP_DIR or configs/serving.yaml's export.llama_cpp_dir."
        )

    merged_dir = Path(merge_config["merged_dir"])
    output_path = Path(export_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intermediate_path = output_path.with_suffix(f".{export_config['outtype']}.gguf")

    run(
        [
            sys.executable,
            str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outtype",
            export_config["outtype"],
            "--outfile",
            str(intermediate_path),
        ]
    )

    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    run(
        [
            str(quantize_bin),
            str(intermediate_path),
            str(output_path),
            export_config["quantization"],
        ]
    )

    intermediate_path.unlink(missing_ok=True)
    print(f"exported quantized GGUF -> {output_path}")


if __name__ == "__main__":
    main()
