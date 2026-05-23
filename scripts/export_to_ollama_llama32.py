"""
Sibling of export_to_ollama_qwen.py for the Llama-3.2-1B adapter. Same four
manual steps because Unsloth's save_pretrained_gguf still breaks on Windows
the same way regardless of base model.

  1. Merge: load fp16 base + LoRA adapter, peft.merge_and_unload(), save HF dir
  2. Convert: call llama.cpp's convert_hf_to_gguf.py -> F16 GGUF
  3. Quantize: call built llama-quantize.exe -> Q4_K_M (or chosen) GGUF
  4. Modelfile: hand-write the Ollama spec with the Llama-3 chat template

Prerequisites (one-time):
    pip install -r C:\\Users\\Admin\\.unsloth\\llama.cpp\\requirements\\requirements-convert_hf_to_gguf.txt

Run:
    python scripts/export_to_ollama_llama32.py
    python scripts/export_to_ollama_llama32.py --quant Q5_K_M
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
ADAPTER_DIR   = REPO_ROOT / "models" / "llama32-medqa-adapter"
MERGED_DIR    = REPO_ROOT / "models" / "llama32-medqa-merged"
GGUF_DIR      = REPO_ROOT / "models" / "llama32-medqa-gguf"
LLAMA_CPP_DIR = Path(r"C:\Users\Admin\.unsloth\llama.cpp")

# We merge against the un-quantized HF base, not Unsloth's bnb-4bit wrapper.
# LoRA weights are architecture-keyed, not precision-keyed, so this is correct.
# Llama-3.2 is gated; the user needs HF licence acceptance + `huggingface-cli login`
# for this base id to download.
BASE_MODEL_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"

QUANT_DESCRIPTIONS = {
    "Q4_K_M": "balanced default, ~0.8 GB",
    "Q5_K_M": "safer for fine-tunes, ~0.9 GB",
    "Q8_0":   "near-lossless, ~1.3 GB",
    "F16":    "no quantization, ~2.4 GB",
}

# Ollama uses Go templates. This matches Llama-3.2's training-time chat template
# exactly -- the same one train_qlora_llama32.py applied via
# get_chat_template(..., "llama-3.2"). Llama-3.2 1B/3B text models reuse the
# Llama-3.1 token format (same <|begin_of_text|>, <|start_header_id|>...<|end_header_id|>,
# <|eot_id|> markers), so this template block is identical regardless of which
# "llama-3.x" template key was used at train time.
LLAMA3_TEMPLATE = (
    "<|begin_of_text|>"
    "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n"
    "{{ .System }}<|eot_id|>"
    "{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ .Prompt }}<|eot_id|>"
    "{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ .Response }}<|eot_id|>"
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
        f'TEMPLATE """{LLAMA3_TEMPLATE}"""',
        "",
        # Llama-3 uses <|eot_id|> as the turn terminator; <|end_of_text|> ends
        # the sequence. Both as stops so Ollama trims correctly in chat mode.
        'PARAMETER stop "<|eot_id|>"',
        'PARAMETER stop "<|end_of_text|>"',
        'PARAMETER stop "<|start_header_id|>"',
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
                    help="Keep models/llama32-medqa-merged/ (default: delete; ~2.5 GB)")
    ap.add_argument("--keep-f16-gguf", action="store_true",
                    help="Keep the intermediate F16 GGUF (default: delete; ~2.5 GB)")
    ap.add_argument("--system", default="",
                    help="Optional Modelfile SYSTEM prompt")
    args = ap.parse_args()

    # Pre-flight checks
    if not (args.adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {args.adapter_dir} -- run train_qlora_llama32.py first")
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
    print( "          ollama create medqa-llama32 -f Modelfile")
    print( "          ollama run medqa-llama32 \"What are the side effects of metformin?\"")


if __name__ == "__main__":
    main()
