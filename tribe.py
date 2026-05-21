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
    def extract_frame_attention(
        self,
        frame_bytes: bytes,
        capture_h: int = 1080,
        capture_w: int = 1920,
    ) -> bytes:
        """Run DINOv2 ViT with output_attentions=True on a single frame.

        Loads ``facebook/dinov2-large`` from the persistent volume on first call
        (lazy, separate from TribeModel).  Subsequent calls reuse the cached
        instance on the same worker.

        Integration note (R0):
            This method loads DINOv2 *independently* of TribeModel to guarantee
            ``output_attentions=True`` works without modifying the predict path.
            If tribev2 exposes the DINOv2 backbone directly in a future release,
            ``self._dinov2`` can be wired to ``self.model.visual_encoder`` instead.

        Args:
            frame_bytes: JPEG- or PNG-encoded bytes of a single video frame.
            capture_h:   Target heatmap height (Playwright capture height, e.g. 1080).
            capture_w:   Target heatmap width  (Playwright capture width,  e.g. 1920).

        Returns:
            Serialised float32 ndarray ``(capture_h, capture_w)`` as numpy .npy
            bytes (use ``numpy.load(io.BytesIO(result))`` to deserialise).
            Values are ≥ 0 (raw attention, not normalised to 1).
        """
        import io

        import numpy as np
        import torch
        from PIL import Image
        from scipy.ndimage import zoom as ndimage_zoom
        from transformers import AutoImageProcessor, AutoModel

        # Lazy-load DINOv2 (reuses weights already on the persistent volume).
        if not hasattr(self, "_dinov2"):
            self._dinov2_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-large", cache_dir="/cache"
            )
            self._dinov2 = AutoModel.from_pretrained(
                "facebook/dinov2-large", cache_dir="/cache"
            )
            self._dinov2 = self._dinov2.cuda().eval()

        # Decode and preprocess frame (→ 224×224 tensor).
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        inputs = self._dinov2_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].cuda()  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = self._dinov2(pixel_values, output_attentions=True)

        # Final-block attentions: tuple element shape = (1, num_heads, S, S).
        last_attn = outputs.attentions[-1]          # [1, num_heads, S, S]
        attn_np = last_attn[0].cpu().numpy()        # [num_heads, S, S]

        # Derive square patch grid from seq_len (CLS excluded).
        n_patches = attn_np.shape[-1] - 1
        grid_size = int(n_patches ** 0.5)

        # Mean-across-heads [CLS] → patch attention; reshape to 2-D grid.
        cls_attn = attn_np[:, 0, 1:]               # [num_heads, n_patches]
        patch_grid = cls_attn.mean(axis=0).reshape(grid_size, grid_size).astype(np.float32)

        # Bilinear upscale to Playwright capture resolution.
        heatmap = ndimage_zoom(
            patch_grid,
            (capture_h / grid_size, capture_w / grid_size),
            order=1,
        )
        heatmap = np.clip(heatmap, 0.0, None).astype(np.float32)

        # Serialise as numpy bytes.
        buf = io.BytesIO()
        np.save(buf, heatmap)
        return buf.getvalue()

    @modal.method()
    def ground_spike_at_t(
        self,
        t_spike: int,
        frame_bytes: bytes,
        manifest: dict,
        capture_h: int = 1080,
        capture_w: int = 1920,
    ) -> dict:
        """Decode frame, extract heatmap, run DOM intersection; return grounding payload.

        Combines ``extract_frame_attention`` + DOM attention-density scoring in a
        single Modal call for callers that do not need to cache heatmaps locally.

        Callers that *do* need cached heatmaps (e.g. ``analyze_session.py --ground``)
        should call ``extract_frame_attention`` separately and run
        ``scout_core.dom_intersect`` locally.

        Args:
            t_spike:    Row index into ``preds[T, 20484]`` (TRIBE TR).
            frame_bytes: JPEG/PNG bytes of the video frame at ``t_spike``.
            manifest:   Parsed ``session_manifest.json`` dict with ``dom_snapshots``.
            capture_h:  Heatmap height (must match Playwright capture resolution).
            capture_w:  Heatmap width  (must match Playwright capture resolution).

        Returns:
            Grounding dict ``{dom_id, tag, bbox, attention_density}`` for the winner,
            or ``{"grounding": None, "reason": "<cause>"}`` if no match found.
        """
        import io

        import numpy as np

        # Extract heatmap (reuses lazy-loaded DINOv2 on the same worker).
        heatmap_bytes = self.extract_frame_attention.local(frame_bytes, capture_h, capture_w)
        heatmap = np.load(io.BytesIO(heatmap_bytes)).astype(np.float32)
        img_h, img_w = heatmap.shape

        # Find nearest DOM snapshot by t_idx.
        snapshots = manifest.get("dom_snapshots", [])
        if not snapshots:
            return {"grounding": None, "reason": "no_dom_snapshots_in_manifest"}

        nearest = min(snapshots, key=lambda s: abs(int(s.get("t_idx", 0)) - t_spike))
        elements = nearest.get("elements", [])
        if not elements:
            return {"grounding": None, "reason": "no_elements_in_snapshot"}

        scroll_y = int(nearest.get("scrollY", 0))
        scroll_x = int(nearest.get("scrollX", 0))

        # Inlined attention-density loop (scout_core not in Modal image).
        best_density = -1.0
        winner: dict | None = None
        for el in elements:
            if not el.get("is_intersecting_viewport", True):
                continue
            bbox = el.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            vx, vy = x - scroll_x, y - scroll_y
            x1, y1 = max(0, vx), max(0, vy)
            x2, y2 = min(img_w, vx + w), min(img_h, vy + h)
            if x2 <= x1 or y2 <= y1:
                continue
            box_area = (x2 - x1) * (y2 - y1)
            if box_area == 0:
                continue
            density = float(heatmap[y1:y2, x1:x2].sum() / box_area)
            if density > best_density:
                best_density = density
                winner = {
                    "dom_id": el.get("dom_id", ""),
                    "tag": el.get("tag", ""),
                    "bbox": [int(b) for b in bbox],
                    "attention_density": round(density, 6),
                }

        return winner or {"grounding": None, "reason": "no_eligible_elements"}

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
    return Path(__file__).resolve().parent / "mrbeast.mp4"


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

    try:
        from export_brain_viewer import export_session_viewer  # noqa: WPS433

        viewer_dir = export_session_viewer(session_id)
    except ModuleNotFoundError as exc:
        if exc.name != "nilearn":
            raise
        viewer_dir = None
        print(
            "3D viewer export skipped: local Python is missing `nilearn`. "
            "Run `python3 -m pip install -r requirements.txt` to enable it."
        )
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
    if viewer_dir is not None:
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
