import modal
from pathlib import Path

# 1. Define the App and persistent storage for weights
app = modal.App("tribe-v2-brain-sim")
volume = modal.Volume.from_name("tribe-weights-vol", create_if_missing=True)

# 2. Build the container image with all dependencies
# Note: we pin numpy to avoid the 'ImportError: cannot import name _center' bug
tribe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "xvfb",
        # VTK/PyVista headless on Modal (no DISPLAY): avoids EGL/OSMesa SIGSEGV
        "libegl1-mesa",
        "libosmesa6",
    )
    .pip_install("numpy>=1.26.4,<2.1.0") 
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "accelerate", "nilearn", "pyvista", "huggingface_hub",
        "imageio[ffmpeg]",  # ffmpeg-backed IO (also used by some plotting paths)
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /root/tribev2",
        # Quote extras so bash does not treat `[plotting]` as a glob.
        'pip install -e "/root/tribev2[plotting]"',
    )
)
@app.cls(
    gpu="A100", 
    image=tribe_image, 
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")], # Needs HF_TOKEN
    timeout=1200
)
class TribeInference:
    @modal.enter()
    def load_model(self):
        from tribev2.demo_utils import TribeModel
        # This will download weights to the persistent volume on first run
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2", 
            cache_folder="/cache"
        )

    @modal.method()
    def predict_and_visualize(self, video_bytes: bytes):
        """Single GPU pass: predict cortical time series and render side-view MP4."""
        import io
        import os
        import subprocess
        import tempfile
        import time

        import numpy as np

        if not os.environ.get("DISPLAY"):
            subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "+extension", "GLX"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = ":99"
            time.sleep(1.0)

        from tribev2.plotting import PlotBrain

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_bytes)
            video_path = temp_video.name

        df = self.model.get_events_dataframe(video_path=video_path)
        preds, _ = self.model.predict(events=df)

        plotter = PlotBrain(mesh="fsaverage5")
        output_path = "/tmp/brain_sim.mp4"
        plotter.plot_timesteps_mp4(preds, output_path, views="left")

        buf = io.BytesIO()
        np.savez_compressed(buf, preds=np.asarray(preds, dtype=np.float32))
        return buf.getvalue(), Path(output_path).read_bytes()

    @modal.method()
    def predict_brain(self, video_bytes: bytes):
        import tempfile

        # Save incoming bytes to a temp file for FFmpeg to process
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_bytes)
            video_path = temp_video.name

        # 1. Generate the event dataframe (Syncing Video/Audio/Text)
        df = self.model.get_events_dataframe(video_path=video_path)

        # 2. Run the prediction (fsaverage5 cortical mesh, ~20k vertices)
        # Returns: (n_timesteps, n_vertices)
        preds, segments = self.model.predict(events=df)

        return preds.tolist()

    @modal.method()
    def visualize_brain(self, video_bytes: bytes):
        import os
        import subprocess
        import tempfile
        import time

        # VTK wants an X display; PyVista wheels here omit start_xvfb(), so run Xvfb ourselves.
        if not os.environ.get("DISPLAY"):
            subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "+extension", "GLX"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = ":99"
            time.sleep(1.0)

        from tribev2.plotting import PlotBrain

        # 1. Run prediction
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_bytes)
            video_path = temp_video.name
        
        df = self.model.get_events_dataframe(video_path=video_path)
        preds, _ = self.model.predict(events=df)

        # 2. Headless surface movie (PlotBrain is PlotBrainPyvista — no save_gif).
        # preds shape: (n_timesteps, n_vertices); see tribev2.plotting.base.plot_timesteps_mp4
        plotter = PlotBrain(mesh="fsaverage5")
        output_path = "/tmp/brain_sim.mp4"
        plotter.plot_timesteps_mp4(preds, output_path, views="left")

        return Path(output_path).read_bytes()

def _default_video_path() -> Path:
    return Path(__file__).resolve().parent / "videoplayback.mp4"


def _persist_session(video_path: Path, preds_npz_bytes: bytes, video_mp4_bytes: bytes):
    """Save preds.npz, SQLite summaries, side-view MP4, and interactive viewer bundle."""
    import io
    import sys

    import numpy as np

    from activation_store import DB_PATH, DATA_DIR, resolve_vertex_regions, save_cortical_timeseries

    preds = np.load(io.BytesIO(preds_npz_bytes))["preds"]
    vertex_map = resolve_vertex_regions()
    session_id = save_cortical_timeseries(
        preds,
        source_video=video_path.name,
        mesh_name="fsaverage5",
        vertex_to_region=vertex_map,
        top_k_peaks=64,
        notes="TRIBE v2 prediction via modal run tribe.py",
    )

    session_dir = DATA_DIR / "sessions" / session_id
    (session_dir / "brain_results.mp4").write_bytes(video_mp4_bytes)
    Path("brain_results.mp4").write_bytes(video_mp4_bytes)

    scripts_dir = Path(__file__).resolve().parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_brain_viewer import export_session_viewer  # noqa: WPS433

    viewer_dir = export_session_viewer(session_id)
    return session_id, preds.shape, session_dir, viewer_dir, DB_PATH


@app.local_entrypoint()
def main():
    """Predict, save preds.npz + SQLite, render MP4, and export rotatable 3D viewer data.

    Usage: ``modal run tribe.py``  (same as before, but now persists all artifacts)
    """
    video_path = _default_video_path()
    video_data = video_path.read_bytes()
    predictor = TribeInference()

    print(f"Loading {video_path} — predicting, rendering, and saving locally...")
    preds_npz_bytes, video_mp4_bytes = predictor.predict_and_visualize.remote(video_data)

    session_id, shape, session_dir, viewer_dir, db_path = _persist_session(
        video_path, preds_npz_bytes, video_mp4_bytes
    )

    print(f"Session id: {session_id}")
    print(f"preds shape (T×V): {shape[0]} × {shape[1]}")
    print(f"Dense matrix: {session_dir / 'preds.npz'}")
    print(f"SQLite (summaries + top-64 peaks/timestep): {db_path}")
    print(f"Side-view video: brain_results.mp4 and {session_dir / 'brain_results.mp4'}")
    print(f"3D viewer: open {viewer_dir / 'index.html'} in a browser")
    print("Inspect numbers: python scripts/inspect_session.py --session-id", session_id)


@app.local_entrypoint()
def record():
    """Run Tribe prediction and persist cortical time series + SQLite summaries/peaks.

    Full matrix: scout_data/sessions/<session_id>/preds.npz (array ``preds``, shape T×V).
    Query tables: sessions, timestep_summary, activation_peak in scout_data/activations.sqlite.

    Optional atlas: configs/vertex_regions.csv → activation_peak.brain_sector.

    Interpretation lookups (see activation_store.py):
    - stimulation_band: timestep/session percentiles → qualitative stimulation labels.
    - location_coarse_bands.csv optional → remap vertex_fraction [0,1] bins (default seeds are mesh-index proxies only).

    Prefer ``modal run tribe.py`` — same persistence plus MP4 and 3D viewer export.

    Usage: ``modal run tribe.py::record`` (predict only, no MP4 / viewer)
    """
    from activation_store import DB_PATH, DATA_DIR, resolve_vertex_regions, save_cortical_timeseries

    import io
    import numpy as np

    video_path = _default_video_path()
    video_data = video_path.read_bytes()
    predictor = TribeInference()

    print(f"Predicting cortical time series for {video_path} …")
    preds_list = predictor.predict_brain.remote(video_data)
    preds = np.asarray(preds_list, dtype=np.float32)

    vertex_map = resolve_vertex_regions()
    session_id = save_cortical_timeseries(
        preds,
        source_video=video_path.name,
        mesh_name="fsaverage5",
        vertex_to_region=vertex_map,
        top_k_peaks=64,
        notes="TRIBE v2 demo prediction; emotion/Yeo semantics require downstream mapping.",
    )

    print(f"Session id: {session_id}")
    print(f"Dense preds (T×V): {DATA_DIR / 'sessions' / session_id / 'preds.npz'}")
    print(f"SQLite: {DB_PATH}")
    print("Tip: use `modal run tribe.py` to also render MP4 and export the 3D viewer.")
    if vertex_map is None:
        print(
            "Tip: add configs/vertex_regions.csv (vertex_index,region_name) "
            "to label brain_sector on peak rows."
        )

