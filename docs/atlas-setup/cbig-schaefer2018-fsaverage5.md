# CBIG Schaefer 2018 on fsaverage5 (surface `.annot`)

Minimal download for the surface-native NeuroEmo/TribeV2 parcellation path. No FreeSurfer install and no full CBIG clone.

## Required files (400 parcels, 7 Yeo networks)

Save exactly these two files:

```text
scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/label/
  lh.Schaefer2018_400Parcels_7Networks_order.annot
  rh.Schaefer2018_400Parcels_7Networks_order.annot
```

| File | Raw URL |
|------|---------|
| LH | https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_7Networks_order.annot |
| RH | https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_7Networks_order.annot |

Source folder: [CBIG FreeSurfer5.3/fsaverage5/label](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label).

For other resolutions, use `*_<N>Parcels_<7|17>Networks_order.annot` with the same LH/RH naming pattern.

## Vertex-order proof (auto-downloaded)

CBIG `.annot` indices follow **FreeSurfer** fsaverage5 vertex order; TribeV2/Nilearn BOLD uses a **different** vertex order on the same mesh. The builder remaps labels using CBIG `lh.pial` / `rh.pial` (downloaded automatically on first run if missing):

```text
scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/surf/
  lh.pial
  rh.pial
```

| File | Raw URL |
|------|---------|
| LH | https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/surf/lh.pial |
| RH | https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/surf/rh.pial |

## Not required

| Item | Reason |
|------|--------|
| Full CBIG repo | Only `.annot` + optional `surf/*.pial` are read |
| FreeSurfer binaries | Surfaces/labels loaded via `nibabel.freesurfer` |
| Nilearn fsaverage5 meshes | Fetched at runtime by nilearn |
| Volumetric Schaefer NIfTI | Diagnostic fallback only (`--allow-volume-projection-fallback`) |

## Download commands

From the repository root (`TribeV2`):

**PowerShell**

```powershell
$base = "scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/label"
New-Item -ItemType Directory -Force -Path $base | Out-Null
$urls = @{
  "lh.Schaefer2018_400Parcels_7Networks_order.annot" = "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_7Networks_order.annot"
  "rh.Schaefer2018_400Parcels_7Networks_order.annot" = "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_7Networks_order.annot"
}
foreach ($name in $urls.Keys) {
  Invoke-WebRequest -Uri $urls[$name] -OutFile (Join-Path $base $name)
}
```

**Unix**

```bash
BASE=scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/label
mkdir -p "$BASE"
curl -fsSL -o "$BASE/lh.Schaefer2018_400Parcels_7Networks_order.annot" \
  "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_7Networks_order.annot"
curl -fsSL -o "$BASE/rh.Schaefer2018_400Parcels_7Networks_order.annot" \
  "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_7Networks_order.annot"
```

## Build atlas artifacts

Prerequisite: run vertex equivalence proof once if missing:

```bash
python scripts/verify_tribe_vertex_equivalence.py
```

Then:

```bash
.venv\Scripts\activate
python scripts/verify_schaefer_annot_alignment.py \
  --data-dir scout_data/atlases/cbig_schaefer2018

python scripts/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7 \
  --data-dir scout_data/atlases/cbig_schaefer2018
```

Outputs: `configs/vertex_regions.csv`, `configs/parcellation_manifest.yaml`.
