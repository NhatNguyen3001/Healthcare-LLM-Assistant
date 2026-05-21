"""
Unsloth's save_pretrained_gguf path breaks on Windows because its dynamic patch
of llama.cpp's convert_hf_to_gguf.py emits a bad `import conversion` that has
no corresponding file on disk. This script does the same four steps directly:

  1. Merge: load fp16 base + LoRA adapter, peft.merge_and_unload(), save HF dir
  2. Convert: call llama.cpp's convert_hf_to_gguf.py -> F16 GGUF
  3. Quantize: call built llama-quantize.exe -> Q4_K_M (or chosen) GGUF
  4. Modelfile: hand-write the Ollama spec with the Qwen2.5 chat template

Prerequisites (one-time):
    pip install -r C:\\Users\\Admin\\.unsloth\\llama.cpp\\requirements\\requirements-convert_hf_to_gguf.txt

Run:
    python scripts/export_to_ollama_manual.py
    python scripts/export_to_ollama_manual.py --quant Q5_K_M
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT     = Path(__file__).resolve().parent.parent
ADAPTER_DIR   = REPO_ROOT / "models" / "qwen-medqa-adapter"
MERGED_DIR    = REPO_ROOT / "models" / "qwen-medqa-merged"
GGUF_DIR      = REPO_ROOT / "models" / "qwen-medqa-gguf"
LLAMA_CPP_DIR = Path(r"C:\Users\Admin\.unsloth\llama.cpp")

# We merge against the un-quantized HF base, not Unsloth's bnb-4bit wrapper.
# LoRA weights are architecture-keyed, not precision-keyed, so this is correct.
BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

QUANT_DESCRIPTIONS = {
    "Q4_K_M": "balanced default, ~1.0 GB",
    "Q5_K_M": "safer for fine-tunes, ~1.1 GB",
    "Q8_0":   "near-lossless, ~1.7 GB",
    "F16":    "no quantization, ~3 GB",
}

# Ollama uses Go templates. This matches Qwen2.5's training-time chat template
# exactly -- the same one train_qlora.py applied via get_chat_template(..., "qwen-2.5").
QWEN25_TEMPLATE = (
    "{{ if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
    "{{ .Prompt }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
    "{{ .Response }}<|im_end|>\n"
)


def step1_merge(base_id: str, adapter_dir: Path, merged_dir: Path) -> None:
    print(f"[merge] loading base {base_id} in fp16")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[merge] attaching adapter from {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] peft.merge_and_unload()")
    merged = peft_model.merge_and_unload()
    print(f"[merge] saving merged checkpoint to {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    del merged, peft_model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def step2_convert_to_gguf(merged_dir: Path, gguf_path: Path, llama_cpp_dir: Path) -> None:
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"missing {script}")
    print(f"[convert] {merged_dir.name} -> {gguf_path.name} (f16)")
    subprocess.run(
        [sys.executable, str(script), str(merged_dir),
         "--outfile", str(gguf_path), "--outtype", "f16"],
        check=True,
    )


def step3_quantize(input_gguf: Path, output_gguf: Path, quant: str, llama_cpp_dir: Path) -> None:
    quantize = llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe"
    if not quantize.exists():
        raise SystemExit(f"missing {quantize}")
    print(f"[quantize] {input_gguf.name} -> {output_gguf.name} ({quant})")
    subprocess.run(
        [str(quantize), str(input_gguf), str(output_gguf), quant],
        check=True,
    )


def step4_write_modelfile(gguf_dir: Path, gguf_name: str, system_prompt: str) -> None:
    parts = [f"FROM ./{gguf_name}", ""]
    if system_prompt:
        parts += [f'SYSTEM """{system_prompt}"""', ""]
    parts += [
        f'TEMPLATE """{QWEN25_TEMPLATE.rstrip(chr(10))}"""',
        "",
        'PARAMETER stop "<|im_start|>"',
        'PARAMETER stop "<|im_end|>"',
        "PARAMETER temperature 0.3",
        "PARAMETER top_p 0.9",
    ]
    (gguf_dir / "Modelfile").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[modelfile] wrote {gguf_dir / 'Modelfile'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir",   type=Path, default=ADAPTER_DIR)
    ap.add_argument("--merged-dir",    type=Path, default=MERGED_DIR)
    ap.add_argument("--output-dir",    type=Path, default=GGUF_DIR)
    ap.add_argument("--llama-cpp-dir", type=Path, default=LLAMA_CPP_DIR)
    ap.add_argument("--base", default=BASE_MODEL_DEFAULT,
                    help=f"HF base model id (default: {BASE_MODEL_DEFAULT})")
    ap.add_argument("--quant", choices=sorted(QUANT_DESCRIPTIONS), default="Q4_K_M",
                    help="Quantization method. " +
                         "; ".join(f"{k}={v}" for k, v in QUANT_DESCRIPTIONS.items()))
    ap.add_argument("--keep-merged-hf", action="store_true",
                    help="Keep models/qwen-medqa-merged/ (default: delete; ~3 GB)")
    ap.add_argument("--keep-f16-gguf", action="store_true",
                    help="Keep the intermediate F16 GGUF (default: delete; ~3 GB)")
    ap.add_argument("--system", default="",
                    help="Optional Modelfile SYSTEM prompt")
    args = ap.parse_args()

    # Pre-flight checks
    if not (args.adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {args.adapter_dir} -- run train_qlora.py first")
    try:
        import gguf  # noqa: F401
    except ImportError:
        raise SystemExit(
            "missing python deps for convert_hf_to_gguf.py.\nRun:\n"
            f'    pip install -r "{args.llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"}"'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp16_gguf  = args.output_dir / "model.F16.gguf"
    final_gguf = args.output_dir / f"model.{args.quant}.gguf"

    step1_merge(args.base, args.adapter_dir, args.merged_dir)
    step2_convert_to_gguf(args.merged_dir, fp16_gguf, args.llama_cpp_dir)

    if args.quant == "F16":
        if fp16_gguf != final_gguf:
            shutil.move(str(fp16_gguf), str(final_gguf))
    else:
        step3_quantize(fp16_gguf, final_gguf, args.quant, args.llama_cpp_dir)
        if not args.keep_f16_gguf:
            fp16_gguf.unlink(missing_ok=True)

    step4_write_modelfile(args.output_dir, final_gguf.name, args.system)

    if not args.keep_merged_hf and args.merged_dir.exists():
        print(f"[cleanup] removing {args.merged_dir}")
        shutil.rmtree(args.merged_dir)

    print()
    print(f"[done] artifacts in {args.output_dir}:")
    for p in sorted(args.output_dir.iterdir()):
        if p.is_file():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"        {p.name:<32} {mb:>8.1f} MB")
    print()
    print("[done] register with Ollama:")
    print(f"          cd {args.output_dir}")
    print( "          ollama create medqa-qwen -f Modelfile")
    print( "          ollama run medqa-qwen \"What are the side effects of metformin?\"")


if __name__ == "__main__":
    main()
