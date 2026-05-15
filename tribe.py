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

# Updated Local Entrypoint
@app.local_entrypoint()
def main():
    video_path = Path(__file__).resolve().parent / "videoplayback.mp4"
    video_data = video_path.read_bytes()
    predictor = TribeInference()

    print(f"🧠 Loading {video_path} — predicting and rendering...")
    video_out = predictor.visualize_brain.remote(video_data)

    output_filename = "brain_results.mp4"
    Path(output_filename).write_bytes(video_out)
    print(f"✅ Success! Visualization saved to {output_filename}")


@app.local_entrypoint()
def record():
    """Run Tribe prediction and persist cortical time series + SQLite summaries/peaks.

    Full matrix: scout_data/sessions/<session_id>/preds.npz (array ``preds``, shape T×V).
    Query tables: sessions, timestep_summary, activation_peak in scout_data/activations.sqlite.

    Optional atlas: configs/vertex_regions.csv → activation_peak.brain_sector.

    Interpretation lookups (see activation_store.py):
    - stimulation_band: timestep/session percentiles → qualitative stimulation labels.
    - location_coarse_bands.csv optional → remap vertex_fraction [0,1] bins (default seeds are mesh-index proxies only).

    Usage (same venv as main): ``modal run tribe.py::record``
    """
    import numpy as np

    from activation_store import DB_PATH, DATA_DIR, resolve_vertex_regions, save_cortical_timeseries

    video_path = Path(__file__).resolve().parent / "videoplayback.mp4"
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
    if vertex_map is None:
        print(
            "Tip: add configs/vertex_regions.csv (vertex_index,region_name) "
            "to label brain_sector on peak rows."
        )

