from __future__ import annotations

import modal

app = modal.App("tribe-demographic-mux")

volume = modal.Volume.from_name("tribe-demographic-mux-vol", create_if_missing=True)

mux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libegl1-mesa", "libosmesa6")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "pyarrow",
        "webdataset",
        "scipy",
        "nilearn",
        "nibabel",
        "pyyaml",
        "tqdm",
        "modal",
        "pydantic",
        "scikit-learn",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /root/tribev2",
        'pip install -e "/root/tribev2[plotting]"',
    )
    .add_local_python_source("salience")
    .add_local_python_source("data_prep")
    .add_local_python_source("scout_core")
)

M0_REPORT_PATH = "/mnt/mux/viability/m0_report.json"
M0_SUBJECTS_CSV = "/mnt/mux/viability/subjects.csv"
M0_PREFLIGHT_PATH = "/mnt/mux/viability/preflight_report.json"

m0_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "nilearn",
        "nibabel",
        "boto3",
        "pyyaml",
        "tqdm",
        "pydantic",
        "scikit-learn",
    )
    .add_local_python_source("data_prep")
    .add_local_python_source("scout_core")
    .add_local_python_source("salience")
    .add_local_dir("configs", remote_path="/root/configs")
)
