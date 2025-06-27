# deepfloyd_if_hud.py  – Stable HUD for DeepFloyd-IF (v1.5)
# ============================================================
"""Self-contained launcher that
──────────────────────────────
* **Checks first – installs only if missing / incompatible.** No more redundant
  uninstall-reinstall loops on every launch.
* **Robust installer**  (3 retries + SHA-purge) for giant CUDA wheels.
* **Compatible dependency matrix**
    └─ Gradio ≥ 4.46 ★ HF-Hub ≥ 0.24 ★ Torch 1.13.1 cu117
* **Config persistence** (`settings.json`) & rotating log (`deepfloyd_if_hud.log`).
* **Gradio UI**  Prompt, Negative Prompt, Up-/Downscale selector, Batch slider,
  Checkpoint folder picker, GPU/CPU auto-detect, gallery + download.

Run inside a Python-3.10 venv:

```powershell
python deepfloyd_if_hud.py
```

The first run installs missing packages, afterwards the app opens instantly at
<http://localhost:7860>.
"""
from __future__ import annotations
import importlib.metadata as _ilm
from packaging.specifiers import SpecifierSet  # type: ignore
from packaging.version    import Version       # type: ignore
import subprocess, sys, logging, pathlib, time, json, shutil, os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
WHEEL_DIR = os.environ.get("HUD_WHEEL_DIR")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG = pathlib.Path(__file__).with_suffix(".log")
_hdlr = logging.FileHandler(LOG, encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_hdlr, logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("HUD")

# Optional Hugging Face token file. Users may place their token here so
# downloads from gated repositories succeed without manual login.
HF_TOKEN_PATH = pathlib.Path("hf_token.txt")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
if HF_TOKEN is None and HF_TOKEN_PATH.exists():
    token_text = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
    if token_text:
        HF_TOKEN = token_text
if HF_TOKEN is None:
    log.warning(
        "HF_TOKEN not set; downloads from gated models may fail. "
        "Create hf_token.txt or set the environment variable HF_TOKEN."
    )

# for optional checkpoint sync
from typing import List

# ---------------------------------------------------------------------------
# Required packages (name, version-spec).  Use PEP 440 specifiers.
# ---------------------------------------------------------------------------
REQS: list[tuple[str, str]] = [
    ("numpy",            "<2"),  # PyTorch 1.13 wheels built against NumPy 1.x
    ("torch",            "==1.13.1+cu117"),
    ("torchvision",      "==0.14.1+cu117"),
    ("torchaudio",       "==0.13.1+cu117"),
    ("diffusers",        "==0.16.0"),
    ("deepfloyd-if",     "==1.0.2rc0"),  # installed with --no-deps later
    # diffusers 0.16.0 still depends on the deprecated
    # `cached_download()` API. Versions of huggingface-hub
    # >=0.26 removed this function, so we pin to 0.25.2
    # for compatibility.
    ("huggingface-hub",  "==0.25.2"),
    # The latest 4.x release of gradio is 4.44.1. Older versions crash with
    # pydantic>=2.11, so we pin exactly to 4.44.1 and keep pydantic below 2.11.
    ("gradio",           "==4.44.1"),
    ("pydantic",         "<2.11"),
    ("accelerate",       "==0.15.0"),
    ("ftfy",             "==6.1.1"),
    ("matplotlib",       "==3.8.4"),
    ("tqdm",             "==4.65.0"),
]

CUDA_IDX = "https://download.pytorch.org/whl/cu117"
PIP_BASE = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off"]

# Use fp16 safetensors only to save space
ALLOW_PATTERNS = ["*.json", "*.fp16.safetensors"]

# common settings for diffusers pipelines
import torch
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

COMMON_KW = dict(
    variant="fp16",
    use_safetensors=True,
    local_files_only=bool(os.getenv("HF_HUB_OFFLINE")),
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

from functools import lru_cache
from diffusers import DiffusionPipeline

@lru_cache(maxsize=None)
def load_upscaler(repo_id: str) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(repo_id, **COMMON_KW)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("cuda" if torch.cuda.is_available() else "cpu", device_map="auto")
    return pipe

@lru_cache(maxsize=None)
def load_stage(path: pathlib.Path) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(str(path), **COMMON_KW)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("cuda" if torch.cuda.is_available() else "cpu", device_map="auto")
    return pipe

# Stage repositories to verify/download
STAGES = {
    "IF-I-XL-v1.0":  "DeepFloyd/IF-I-XL-v1.0",
    "IF-II-L-v1.0":  "DeepFloyd/IF-II-L-v1.0",
}
# Stage III is not available publicly. The pipeline will fall back to the Stable
# Diffusion x4 upscaler for 1024 px output, so we omit this stage from the
# download list to avoid repeated 404 errors.
OPTIONAL_STAGES: set[str] = set()
POINTER_THRESHOLD = 200  # bytes: files smaller likely Git-LFS pointers

STAGE_NAMES = set(STAGES.keys())

def resolve_root(path: pathlib.Path) -> pathlib.Path:
    """Return the actual model root directory."""
    for part in reversed([path] + list(path.parents)):
        if part.name in STAGE_NAMES:
            return part.parent
    return path

# ---------------------------------------------------------------------------
# Helper: is a package with given spec already satisfied?
# ---------------------------------------------------------------------------

def _satisfied(name: str, spec: str) -> bool:
    try:
        current = Version(_ilm.version(name))
    except _ilm.PackageNotFoundError:
        return False

    if not spec:
        return True
    if spec.startswith("=="):
        return current == Version(spec[2:])
    spec_set = SpecifierSet(spec)
    return current in spec_set

# ---------------------------------------------------------------------------
# Helper: build a pip requirement string from (name,spec)
# ---------------------------------------------------------------------------

def _req_str(name: str, spec: str) -> str:
    if spec.startswith(("==", ">", "<")):
        return f"{name}{spec}"
    if spec.startswith("~="):
        return f"{name}{spec}"
    if spec:
        return f"{name}=={spec}"
    return name

# ---------------------------------------------------------------------------
# Install routine with retries & SHA-purge
# ---------------------------------------------------------------------------

def _install(name: str, spec: str, tries: int = 3, extra_args: list[str] | None = None) -> None:
    req = _req_str(name, spec)
    args = PIP_BASE[:]
    if WHEEL_DIR:
        args += ["--no-index", "--find-links", WHEEL_DIR]
    elif "+cu117" in spec and name.startswith("torch"):
        args += ["--extra-index-url", CUDA_IDX]
    args.append(req)
    if extra_args:
        args.extend(extra_args)
    for i in range(1, tries + 1):
        log.info("Installing %s (Try %s/%s)…", req, i, tries)
        code = subprocess.call(args)
        log.info("pip exit-code: %s", code)
        if code == 0:
            return
        cache_dir = pathlib.Path.home()/".cache"/"pip"/"http"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        time.sleep(2)
    raise RuntimeError(f"Failed to install {req} after {tries} tries")

# ---------------------------------------------------------------------------
# Ensure all requirements are satisfied
# ---------------------------------------------------------------------------

def ensure_requirements() -> None:
    for name, spec in REQS:
        if name == "deepfloyd-if":
            if not _satisfied(name, spec):
                _install(name, spec, extra_args=["--no-deps"])
            else:
                log.info("\u2713 %s%s already satisfied", name, spec)
            continue

        if _satisfied(name, spec):
            log.info("\u2713 %s%s already satisfied", name, spec)
            continue
        _install(name, spec)


# ---------------------------------------------------------------------------
# Ensure checkpoints are fully downloaded (no Git-LFS pointers)
# ---------------------------------------------------------------------------

ESSENTIAL_SUBDIRS = ["text_encoder", "unet"]
ALLOWED_WEIGHT_SUFFIXES = {".bin", ".safetensors", ".ckpt"}

def _log_warning_once(msg: str) -> None:
    if msg not in _log_warning_once.seen:
        log.warning(msg)
        _log_warning_once.seen.add(msg)

_log_warning_once.seen = set()

def human_size(num: int) -> str:
    """Return a human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"

def dir_size(path: pathlib.Path) -> int:
    """Return total size of all files under ``path``."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def log_dir_tree(root: pathlib.Path, max_entries: int = 1000) -> None:
    """Log a directory tree starting at ``root`` (limited to ``max_entries``).

    Each file entry includes the file size in a human-readable form.
    """
    if not root.exists():
        log.info("[dir] %s (missing)", root)
        return
    def _h(num: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if num < 1024:
                return f"{num:.1f} {unit}"
            num /= 1024
        return f"{num:.1f} PB"
    from itertools import islice
    for p in islice(root.rglob("*"), max_entries):
        indent = "│   " * (len(p.relative_to(root).parts) - 1)
        if p.is_file():
            size = _h(p.stat().st_size)
            log.info("%s├── %s (%s)", indent, p.name, size)
        else:
            log.info("%s├── %s", indent, p.name)

def _stage_complete(stage_dir: pathlib.Path) -> bool:
    """Return True if all blobs in ``stage_dir`` appear fully downloaded."""
    if not stage_dir.exists():
        return False

    found_weight = False
    for file in stage_dir.rglob("*"):
        if not file.is_file():
            continue
        size = file.stat().st_size
        if file.suffix in ALLOWED_WEIGHT_SUFFIXES:
            found_weight = True
            if size < 1_000_000:
                return False
        elif file.suffix == ".json" and size < POINTER_THRESHOLD:
            return False

    for sub in ESSENTIAL_SUBDIRS:
        if not any((stage_dir / sub).glob("*")):
            _log_warning_once(f"Subdir {sub} missing in {stage_dir}")

    return found_weight

def _repo_size(repo: str, token: str | None) -> tuple[int, int]:
    """Return (num_files, total_bytes) for ``repo``."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo_id=repo, token=token)
    total = sum(s.size or 0 for s in info.siblings)
    return len(info.siblings), total


def _download_stage(repo: str, target: pathlib.Path, token: str | None, tries: int = 3) -> bool:
    """Download ``repo`` into ``target`` sequentially.

    Retries on network errors and handles missing optional stages gracefully.
    """
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub import HfHubHTTPError  # not present in very old versions
    except Exception:  # pragma: no cover - fallback for hub<0.8
        HfHubHTTPError = Exception

    try:
        count, size = _repo_size(repo, token)
        log.info("%s: %s files, %.1f MB", repo, count, size / 1e6)
    except Exception:
        pass

    for attempt in range(1, tries + 1):
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                resume_download=True,
                local_dir_use_symlinks="auto",
                max_workers=1,
                token=token,
                allow_patterns=ALLOW_PATTERNS,
                local_files_only=bool(os.getenv("HF_HUB_OFFLINE")),
            )
            return True
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404 and repo in OPTIONAL_STAGES:
                log.warning("Optional repo %s not found", repo)
                return False
            log.warning("Download failed (%s/%s): %s", attempt, tries, e)
            time.sleep(5)
        except Exception as e:  # network hiccup or interruption
            log.warning("Download failed (%s/%s): %s", attempt, tries, e)
            time.sleep(5)
    return False


def ensure_stage_blobs(base_dir: pathlib.Path, token: str | None) -> List[str]:
    """Ensure all stages exist, downloading missing files. Returns a list of
    stage names that were fetched."""
    downloaded: List[str] = []
    for folder, repo in STAGES.items():
        target = base_dir / folder
        if not _stage_complete(target):
            log.info("Syncing %s", repo)
            ok = _download_stage(repo, target, token)
            if ok:
                downloaded.append(folder)
            elif folder not in OPTIONAL_STAGES:
                log.error("Failed to download %s", repo)
    return downloaded

# ---------------------------------------------------------------------------
# Model management helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "IF"

def scan_models(root: pathlib.Path) -> list[str]:
    """Return a list of available model sets in ``root``.

    A model set is either ``root`` itself (named ``DEFAULT_MODEL``) if it
    directly contains the stage folders, or any subdirectory that contains at
    least one stage folder. The list is sorted alphabetically and empty results
    fall back to ``[DEFAULT_MODEL]`` so the user can install from scratch.
    """
    root = resolve_root(root)
    if not root.exists():
        return [DEFAULT_MODEL]

    models: list[str] = []

    # root-level model (legacy layout)
    if any((root / s).exists() for s in STAGES):
        models.append(DEFAULT_MODEL)

    for d in root.iterdir():
        if not d.is_dir():
            continue
        # ignore stage directories that may have been created directly
        if d.name in STAGES:
            continue
        if any((d / s).exists() for s in STAGES):
            models.append(d.name)

    return sorted(set(models)) or [DEFAULT_MODEL]


def model_status(model_dir: pathlib.Path) -> str:
    """Return status symbol for the model directory.

    Symbols:
        "❌"  - at least one stage directory missing
        "🟠" - stage directory present but incomplete
        "✅"  - all stages complete
    """
    incomplete = False
    for stage in STAGES:
        sd = model_dir / stage
        if not sd.exists():
            if stage in OPTIONAL_STAGES:
                continue
            return "❌"
        if not _stage_complete(sd):
            if stage in OPTIONAL_STAGES:
                continue
            incomplete = True
    return "🟠" if incomplete else "✅"

def model_complete(model_dir: pathlib.Path) -> bool:
    return model_status(model_dir) == "✅"


def install_model(root: pathlib.Path, name: str, token: str | None) -> list[str]:
    """Ensure the given model has all stages and return list of downloaded stages.

    If ``name`` equals a stage directory, treat it as the default model to avoid
    creating nested stage folders.
    """
    root = resolve_root(root)
    if name == DEFAULT_MODEL or name in STAGES:
        model_dir = root
    else:
        model_dir = root / name
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    return ensure_stage_blobs(model_dir, token)


def delete_model(root: pathlib.Path, name: str) -> None:
    root = resolve_root(root)
    target = root if name == DEFAULT_MODEL or name in STAGES else root / name
    shutil.rmtree(target, ignore_errors=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_requirements()

    import torch, gradio as gr, json
    from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline

    SETTINGS = pathlib.Path("settings.json")
    cfg: dict = {}
    if SETTINGS.exists():
        cfg = json.loads(SETTINGS.read_text())

    def generate(prompt: str, neg: str, steps: int, res: str, batch: int, cp_root: str, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "Generating | model=%s | prompt=%s | res=%s | batch=%s | device=%s",
            model_name,
            prompt,
            res,
            batch,
            device,
        )
        root = resolve_root(pathlib.Path(cp_root))
        model_dir = root if model_name == DEFAULT_MODEL else root / model_name
        stage1 = load_stage(model_dir / "IF-I-XL-v1.0")
        images = stage1(
            prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=7.0,
            num_images_per_prompt=batch,
            output_type="pil",
        ).images

        if res == "256" or res == "1024":
            stage2 = load_stage(model_dir / "IF-II-L-v1.0")
            images = stage2(
                prompt,
                image=images,
                negative_prompt=neg,
                num_inference_steps=steps,
                guidance_scale=4.0,
                output_type="pil",
            ).images

        if res == "1024":
            stage3_dir = model_dir / "IF-III-L-v1.0"
            if _stage_complete(stage3_dir):
                stage3 = load_stage(stage3_dir)
                images = stage3(
                    prompt,
                    image=images,
                    negative_prompt=neg,
                    num_inference_steps=steps,
                    guidance_scale=4.0,
                    output_type="pil",
                ).images
            else:
                up = load_upscaler("stabilityai/stable-diffusion-x4-upscaler")
                images = [up(prompt=prompt, image=im).images[0] for im in images]

        return images

    with gr.Blocks(theme="gradio/soft", title="DeepFloyd-IF HUD") as ui:
        gr.Markdown("## DeepFloyd-IF Local HUD \U0001F680")
        with gr.Row():
            prompt   = gr.Textbox(label="Prompt", lines=2)
            neg      = gr.Textbox(label="Negative Prompt", lines=2)
        with gr.Row():
            steps    = gr.Slider(10, 100, value=50, step=5, label="Diffusion Steps")
            res      = gr.Radio(["256", "512", "1024"], value="256", label="Resolution")
            batch    = gr.Slider(1, 4, value=1, step=1, label="Batch")
        cp_root = gr.Textbox(value=cfg.get("cp_root", ""), label="Models Folder")
        refresh = gr.Button("Refresh Models")
        model_sel = gr.Dropdown(label="Model", choices=[], value=cfg.get("model"), allow_custom_value=True)
        with gr.Row():
            install_btn = gr.Button("Install/Update")
            delete_btn = gr.Button("Delete Model")
        out = gr.Gallery(label="Results", columns=4)
        status = gr.Markdown()
        btn = gr.Button("Generate")

        def _refresh(cp_root):
            root_path = resolve_root(pathlib.Path(cp_root))
            log.info("[refresh] scanning %s", root_path)
            log_dir_tree(root_path)
            models = scan_models(root_path)
            info = []
            for m in models:
                sym = model_status(root_path / m)
                size = 0
                for s in STAGES:
                    d = (root_path if m == DEFAULT_MODEL else root_path / m) / s
                    if d.exists():
                        size += dir_size(d)
                info.append(f"{sym} {m} ({human_size(size)})")
            status = "\n".join(info) if info else "No models found"
            value = models[0] if models else None
            return gr.update(choices=models, value=value), status

        refresh.click(lambda r: _refresh(r), inputs=[cp_root], outputs=[model_sel, status])

        def _install(cp_root, model_name):
            missing = install_model(resolve_root(pathlib.Path(cp_root)), model_name, HF_TOKEN)
            msg = "✅ Model complete" if not missing else "⬇️ " + ", ".join(missing)
            return msg

        install_btn.click(_install, inputs=[cp_root, model_sel], outputs=status)

        def _delete(cp_root, model_name):
            delete_model(resolve_root(pathlib.Path(cp_root)), model_name)
            return _refresh(cp_root)

        delete_btn.click(_delete, inputs=[cp_root, model_sel], outputs=[model_sel, status])

        def _wrap(prompt, neg, steps, res, batch, cp_root, model_name):
            cfg.update(cp_root=cp_root, model=model_name)
            SETTINGS.write_text(json.dumps(cfg, indent=2))
            root = resolve_root(pathlib.Path(cp_root))
            missing = install_model(root, model_name, HF_TOKEN)
            model_dir = root if model_name == DEFAULT_MODEL or model_name in STAGES else root / model_name
            if not model_complete(model_dir):
                msg = "\n".join(
                    ["❌ Model incomplete. Could not download:"] + missing
                ) if missing else "❌ Model incomplete."
                return [], msg
            imgs = generate(prompt, neg, steps, res, batch, str(root), model_name)
            return imgs, ""

        btn.click(_wrap, inputs=[prompt, neg, steps, res, batch, cp_root, model_sel], outputs=[out, status])

        ui.load(lambda: _refresh(cp_root.value), None, [model_sel, status])

    ui.launch(server_name="127.0.0.1", show_error=True, inbrowser=True)

if __name__ == "__main__":
    if not (3,10) <= sys.version_info < (3,11):
        sys.exit("\u274C Python 3.10.x required")
    main()
