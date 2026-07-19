#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/course_env/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-load", action="store_true", help="Only inspect class/method signatures.")
    return parser.parse_args()


FORBIDDEN_IMPORT_ROOTS = {
    "anthropic",
    "httpx",
    "llama_cpp",
    "llamacpp",
    "openai",
    "requests",
    "text_generation",
    "text_generation_inference",
    "vllm",
}
FORBIDDEN_TRANSFORMERS_NAMES = {
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "GenerationConfig",
    "TextGenerationPipeline",
    "pipeline",
}
FORBIDDEN_BENCHMARK_STRINGS = {
    "anchor words",
    "decode_cache_stress",
    "decode_throughput",
    "hidden_",
    "mixed_serving",
    "public_baseline",
    "public_decode",
    "public_long_context",
    "public_mixed",
    "required_keywords",
    "required_substrings",
    "serving_schedule",
    "sharegpt_",
    "sgpt-public",
}
EXCLUDED_SCAN_DIRS = {
    ".cache",
    "__pycache__",
    "data",
    "examples",
    "results",
    "scripts",
    "utils",
}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def fail(message: str) -> None:
    raise SystemExit(f"Strict validation failed: {message}")


def check_static_rules(tree: ast.AST, source_path: Path) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    fail(f"forbidden import '{alias.name}' in {source_path.name}")
                if root == "transformers" and alias.name != "transformers":
                    fail("import specific tokenizer classes instead of broad Transformers model APIs")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", maxsplit=1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                fail(f"forbidden import from '{module}' in {source_path.name}")
            if root == "transformers":
                for alias in node.names:
                    if alias.name in FORBIDDEN_TRANSFORMERS_NAMES:
                        fail(f"forbidden Transformers API: {alias.name}")

        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_TRANSFORMERS_NAMES:
                fail(f"forbidden Transformers API: {node.attr}")

        if isinstance(node, ast.Name):
            if node.id == "suite_name" and isinstance(node.ctx, ast.Load):
                fail("branching on suite_name is forbidden; generate() must run the same inference path")

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for marker in FORBIDDEN_BENCHMARK_STRINGS:
                if marker in lowered:
                    fail(f"benchmark-specific string literal is forbidden: {marker!r}")

        if isinstance(node, ast.Call):
            func_name = dotted_name(node.func)
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            receiver = dotted_name(node.func.value) if isinstance(node.func, ast.Attribute) else ""

            if attr == "generate":
                fail("calling .generate() is forbidden; implement greedy decode yourself")
            if attr == "forward":
                fail("calling .forward() is forbidden; implement the forward computation yourself")
            if attr == "from_pretrained" and receiver not in {"AutoTokenizer", "transformers.AutoTokenizer"}:
                fail("only AutoTokenizer.from_pretrained() is allowed")
            if func_name in {"pipeline", "transformers.pipeline"}:
                fail("Transformers pipeline is forbidden")
            if func_name in {"__import__", "importlib.import_module"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    root = arg.value.split(".", maxsplit=1)[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        fail(f"dynamic import of '{arg.value}' is forbidden")


def load_student_tree() -> ast.AST:
    source_path = PROJECT_ROOT / "student_engine.py"
    student_tree: ast.AST | None = None
    candidate_paths = []
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in EXCLUDED_SCAN_DIRS for part in rel_parts[:-1]):
            continue
        candidate_paths.append(path)

    for path in candidate_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        check_static_rules(tree, path)
        if path == source_path:
            student_tree = tree
    if student_tree is None:
        raise SystemExit("student_engine.py is required.")
    return student_tree


def main() -> None:
    args = parse_args()
    tree = load_student_tree()
    if args.skip_load:
        cls = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StudentEngine"),
            None,
        )
        if cls is None:
            raise SystemExit("student_engine.py must define class StudentEngine.")
        generate = next(
            (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate"),
            None,
        )
        if generate is None:
            raise SystemExit("StudentEngine must define generate().")
        params = [arg.arg for arg in generate.args.args]
        for name in ["prompts", "max_new_tokens", "batch_size", "suite_name"]:
            if name not in params:
                raise SystemExit(f"StudentEngine.generate() missing parameter: {name}")
        print("Static strict-rule and signature checks passed.")
        return

    import student_engine

    cls = getattr(student_engine, "StudentEngine", None)
    if cls is None:
        raise SystemExit("student_engine.py must define class StudentEngine.")

    if not callable(getattr(cls, "generate", None)):
        raise SystemExit("StudentEngine must define generate().")

    generate_sig = inspect.signature(cls.generate)
    for name in ["prompts", "max_new_tokens", "batch_size", "suite_name"]:
        if name not in generate_sig.parameters:
            raise SystemExit(f"StudentEngine.generate() missing parameter: {name}")

    print("Signature check passed.")

    init_kwargs = {
        "model_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    init_sig = inspect.signature(cls)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in init_sig.parameters.values()):
        init_kwargs = {key: value for key, value in init_kwargs.items() if key in init_sig.parameters}

    engine = cls(**init_kwargs)
    prompts = [
        "Repeat this identifier exactly once: VALIDATE-1001.",
        "Repeat this identifier exactly once: VALIDATE-1002.",
    ]
    outputs = engine.generate(prompts, max_new_tokens=16, batch_size=2, suite_name="validate")
    if not isinstance(outputs, list):
        raise SystemExit("generate() must return list[str].")
    if len(outputs) != len(prompts):
        raise SystemExit(f"generate() returned {len(outputs)} outputs for {len(prompts)} prompts.")
    if not all(isinstance(item, str) for item in outputs):
        raise SystemExit("Every generate() output must be str.")
    if not all(item.strip() for item in outputs):
        raise SystemExit("Every generate() output must be non-empty for the runtime sanity check.")
    print("Runtime interface check passed.")


if __name__ == "__main__":
    main()
