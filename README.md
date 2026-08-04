# mGFD CloudGenerator 2.1 :cloud:

<div align="center">

<img src="docs/logo/logo.png" alt="mGFD CloudGenerator logo" width="680" style="margin: 20px 0;">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/gstinoco/mGFD_CloudGenerator) [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Flask](https://img.shields.io/badge/Flask-3.1.1-000000.svg?logo=flask)](https://flask.palletsprojects.com/) [![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-red.svg?logo=opencv)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/NumPy-2.2.6-013243.svg?logo=numpy)](https://numpy.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.15.2-8CAAE6.svg?logo=scipy)](https://scipy.org/) [![Shapely](https://img.shields.io/badge/Shapely-2.1.1-2E8B57.svg)](https://shapely.readthedocs.io/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.0-11557C.svg)](https://matplotlib.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Web platform for generating 2D unstructured point clouds for meshless mGFD workflows**

*From images to contours to classified point clouds — with visualization and neighbor analysis*

### :link: Quick Links
[![🌐 Live Demo](https://img.shields.io/badge/🌐-Live%20Demo-brightgreen)](https://malla.umich.mx/CloudGenerator/) [![🚀 Quick Start](https://img.shields.io/badge/🚀-Quick%20Start-green)](#rocket-quick-start) [![📦 Install](https://img.shields.io/badge/📦-Install-blue)](#package-installation--setup) [![🧮 Model](https://img.shields.io/badge/🧮-Model-purple)](#books-mathematical-model) [![🗂️ Dataset](https://img.shields.io/badge/🗂️-Dataset-blue)](#file_cabinet-dataset-structure) [![📈 Benchmarks](https://img.shields.io/badge/📈-Benchmarks-purple)](#chart_with_upwards_trend-performance-benchmarks) [![🎬 Visualizations](https://img.shields.io/badge/🎬-Visualizations-purple)](#movie_camera-visualizations) [![👥 Team](https://img.shields.io/badge/👥-Research%20Team-blue)](#scientist-research-team) [![🤝 Contribute](https://img.shields.io/badge/🤝-Contributing-orange)](#handshake-contributing) [![🏭 Partners](https://img.shields.io/badge/🏭-Industry%20Partners-0B1B3A)](#factory-industry-partners-supporting-innovation) [![🙏 Thanks](https://img.shields.io/badge/🙏-Acknowledgments-darkgreen)](#pray-acknowledgments)

</div>

---

## :clipboard: Table of Contents
- [Overview](#star2-overview)
- [Features](#sparkles-features)
- [Installation & Setup](#package-installation--setup)
- [Quick Start](#rocket-quick-start)
- [Usage Guide](#book-usage-guide)
- [Visualizations](#movie_camera-visualizations)
- [API Documentation](#gear-api-documentation)
- [Data Formats](#file_cabinet-data-formats)
- [Project Architecture](#open_file_folder-project-architecture)
- [Mathematical Model](#books-mathematical-model)
- [Dataset Structure](#file_cabinet-dataset-structure)
- [Performance Benchmarks](#chart_with_upwards_trend-performance-benchmarks)
- [Contributing](#handshake-contributing)
- [Research Team](#scientist-research-team)
- [Industry Partners Supporting Innovation](#factory-industry-partners-supporting-innovation)
- [Scientific References](#books-scientific-references)
- [Citation & License](#memo-citation--license)
- [Acknowledgments](#pray-acknowledgments)
- [Contact](#email-contact--support)
- [FAQ](#speech_balloon-faq)

---

## :star2: Overview

**mGFD CloudGenerator 2.1** is a Flask-based web platform that converts **images → contours → classified point clouds** designed for **meshless Generalized Finite Differences (mGFD)** workflows. It provides interactive contour extraction, multi-region management (including interior holes), cloud generation with two distribution strategies, high-quality visualizations (PNG/SVG), and region-aware neighbor computation.

> :globe_with_meridians: **Try it now!** Live demo: **https://malla.umich.mx/CloudGenerator/**

### :wrench: Key Capabilities
- **:art: ContourCreator**: Single-click region detection (Flood Fill), interactive refinement, brush editing, and optional GrabCut.
- **:cloud: CloudGenerator**: Regular (grid-like) and Natural (Poisson disk sampling) distributions, optional contour reduction, multi-region support.
- **:eye: CloudViewer**: Upload a CSV (contours or clouds) and instantly generate PNG/SVG visualizations.
- **:users: NeighborsCalculator**: Compute k-nearest neighbors constrained by region boundaries (cKDTree-based).
- **:floppy_disk: Export & Reproducibility**: Standardized CSV formats + verification on export for integrity.
- **:globe_with_meridians: Internationalization (i18n)**: Full English and Spanish localization seamlessly integrated into a premium user interface.

### :microscope: Typical Applications

| Field | Application | Use Case |
|-------|-------------|----------|
| **Computational Fluid Dynamics** :ocean: | Irregular domains | Node generation for complex geometries |
| **Structural Engineering** :building_construction: | Stress analysis | Point sets for meshless discretization |
| **Heat Transfer** :fire: | Interface problems | Multi-region clouds with internal boundaries |
| **Environmental Modeling** :herb: | Geography-driven domains | Contour-driven discretization from maps |
| **Biomedical Engineering** :microscope: | Organ shapes | Point cloud generation from silhouettes |

---

## :sparkles: Features

### :art: ContourCreator
- **One-click region detection** based on `cv2.floodFill` (fast and interactive).
- **Adaptive tolerance** using local texture statistics for robust segmentation.
- **Positive/negative seed refinement** for add/subtract workflows.
- **Brush editing** to manually correct masks and boundaries.
- **Equidistant Contour Sampling**: advanced node retention algorithm ensuring mathematically uniform spacing along complex boundaries.
- **Export single or multi-region contours** as normalized CSV coordinates.

### :cloud: CloudGenerator
- **Regular Distribution**: uniform grid-style generation inside polygonal regions.
- **Natural Distribution**: Poisson disk sampling for organic point spacing.
- **Adaptive Density**: dynamic bounding-box scaling to guarantee appropriate grid spacing for narrow or intricate subregions.
- **Topological Fallbacks**: guaranteed interior node placement via representative points to prevent mathematically empty sub-domains.
- **Contour reduction** to simplify boundaries before generation (optional).
- **Multi-region support**: main region + interior regions (holes) handled explicitly.
- **Node classification** written to CSV (`boundary`, `interior`, and region-aware classes).

### :eye: CloudViewer
- **CSV upload** and automatic rendering to PNG and SVG.
- **Compatibility** with the standardized CSV formats described below.

### :users: NeighborsCalculator
- **Region-constrained neighbor search** (no cross-region connections).
- **Efficient KD-tree implementation** via `scipy.spatial.cKDTree`.

---

## :package: Installation & Setup

### :computer: System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9+ | 3.10+ |
| **RAM** | 4 GB | 8 GB+ |
| **CPU** | 2 cores | 4+ cores |
| **Storage** | 1 GB | 5 GB+ (if generating large datasets) |
| **OS** | Windows/Linux/macOS | Linux (recommended for deployments) |

### :clipboard: Dependencies

```python
# Web framework
Flask==3.1.1
Werkzeug==3.1.3

# Computer vision / geometry
opencv-python-headless==4.12.0.88
shapely==2.1.1

# Scientific computing
numpy==2.2.6
scipy==1.15.2
matplotlib==3.10.0
```

### Quick Installation

```bash
# Method 1: Direct installation
git clone https://github.com/gstinoco/mGFD_CloudGenerator.git
cd mGFD_CloudGenerator
pip install -r requirements.txt

# Method 2: Virtual environment (recommended)
python -m venv cloudgen_env
source cloudgen_env/bin/activate  # On Windows: cloudgen_env\Scripts\activate
pip install -r requirements.txt
```

### :white_check_mark: Installation Verification

```bash
python -c "import flask, cv2, numpy, scipy, shapely, matplotlib; print(':white_check_mark: Installation successful!')"
python app.py
```

Local server default:
- `http://127.0.0.1:5000/`

---

## :rocket: Quick Start

<table>
  <thead>
    <tr>
      <th align="left" width="170">Step</th>
      <th align="left">What to do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) Run</b></td>
      <td>
        <pre><code>python app.py</code></pre>
      </td>
    </tr>
    <tr>
      <td><b>2) Open</b></td>
      <td>
        Go to <code>http://127.0.0.1:5000/</code>
      </td>
    </tr>
    <tr>
      <td><b>3) Workflow</b></td>
      <td>
        <b>ContourCreator</b> → export contours CSV → <b>CloudGenerator</b> → export cloud CSV + PNG/SVG → optional <b>NeighborsCalculator</b>.
      </td>
    </tr>
  </tbody>
</table>

---

## :book: Usage Guide

<div align="center">

*Practical workflows for ContourCreator, CloudGenerator, CloudViewer, and NeighborsCalculator*

</div>

### :art: ContourCreator (Image → Contours)

<table>
  <thead>
    <tr>
      <th align="left" width="170">Step</th>
      <th align="left">What to do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) Upload</b></td>
      <td>Upload an image (PNG/JPG/JPEG/GIF/BMP). Max request size is <b>16 MB</b>.</td>
    </tr>
    <tr>
      <td><b>2) Detect</b></td>
      <td>Click inside a region to extract it with Flood Fill (adaptive tolerance).</td>
    </tr>
    <tr>
      <td><b>3) Refine</b></td>
      <td>Use positive/negative seeds and brush tools to add/remove pixels from the mask.</td>
    </tr>
    <tr>
      <td><b>4) Export</b></td>
      <td>Export a <b>single region</b> or <b>all regions</b> to CSV with normalized coordinates.</td>
    </tr>
  </tbody>
</table>

### :cloud: CloudGenerator (Contours CSV → Classified Cloud)

<table>
  <thead>
    <tr>
      <th align="left" width="170">Step</th>
      <th align="left">What to do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) Upload CSV</b></td>
      <td>Upload a contour CSV (<code>x,y,region</code> recommended for multi-region).</td>
    </tr>
    <tr>
      <td><b>2) Choose method</b></td>
      <td><b>Regular</b> (grid-like) or <b>Natural</b> (Poisson disk sampling).</td>
    </tr>
    <tr>
      <td><b>3) Options</b></td>
      <td>Enable <b>contour reduction</b> and/or generate points inside interior regions (holes).</td>
    </tr>
    <tr>
      <td><b>4) Outputs</b></td>
      <td>
        <b>CSV:</b> <code>x,y,region,classification</code><br/>
        <b>Images:</b> <code>.png</code> and <code>.svg</code> visualizations
      </td>
    </tr>
  </tbody>
</table>

### :eye: CloudViewer (CSV → Visualization)

<table>
  <thead>
    <tr>
      <th align="left" width="170">Step</th>
      <th align="left">What to do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) Upload CSV</b></td>
      <td>Upload a contour or cloud CSV.</td>
    </tr>
    <tr>
      <td><b>2) Inspect</b></td>
      <td>Get PNG/SVG renderings and point counts for fast validation.</td>
    </tr>
  </tbody>
</table>

### :users: NeighborsCalculator (Cloud CSV → Neighbors)

<table>
  <thead>
    <tr>
      <th align="left" width="170">Step</th>
      <th align="left">What to do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) Upload Cloud CSV</b></td>
      <td>Upload a cloud CSV containing <code>x,y,region,classification</code>.</td>
    </tr>
    <tr>
      <td><b>2) Set k</b></td>
      <td>Select the number of neighbors (<code>nvec</code>, default is 9).</td>
    </tr>
    <tr>
      <td><b>3) Compute</b></td>
      <td>Neighbors are computed within each region only (no cross-region links).</td>
    </tr>
  </tbody>
</table>

---

## :movie_camera: Visualizations

### :framed_picture: Examples Gallery

The repository includes sample inputs and outputs under [static/examples/](static/examples/).

<div align="center">

<table>
  <tr>
    <td align="center">
      <b>Balkhash</b><br/>
      <sub>Input image</sub><br/><br/>
      <img src="static/examples/Balkhash.png" alt="Balkhash input" width="220"><br/><br/>
      <sub>Cloud preview</sub><br/><br/>
      <img src="static/examples/Balkhash_cloud.png" alt="Balkhash cloud" width="220"><br/><br/>
      <a href="static/examples/Balkhash_contours.csv"><code>Balkhash_contours.csv</code></a> ·
      <a href="static/examples/Balkhash_cloud.csv"><code>Balkhash_cloud.csv</code></a>
    </td>
    <td align="center">
      <b>Caspio</b><br/>
      <sub>Input image</sub><br/><br/>
      <img src="static/examples/Caspio.png" alt="Caspio input" width="220"><br/><br/>
      <sub>Cloud preview</sub><br/><br/>
      <img src="static/examples/Caspio_cloud.png" alt="Caspio cloud" width="220"><br/><br/>
      <a href="static/examples/Caspio_contours.csv"><code>Caspio_contours.csv</code></a> ·
      <a href="static/examples/Caspio_cloud.csv"><code>Caspio_cloud.csv</code></a>
    </td>
    <td align="center">
      <b>Titicaca</b><br/>
      <sub>Input image</sub><br/><br/>
      <img src="static/examples/Titicaca.png" alt="Titicaca input" width="220"><br/><br/>
      <sub>Cloud preview</sub><br/><br/>
      <img src="static/examples/Titicaca_cloud.png" alt="Titicaca cloud" width="220"><br/><br/>
      <a href="static/examples/Titicaca_contours.csv"><code>Titicaca_contours.csv</code></a> ·
      <a href="static/examples/Titicaca_cloud.csv"><code>Titicaca_cloud.csv</code></a>
    </td>
  </tr>
</table>

</div>

> Tip: Use these CSV files in CloudViewer/CloudGenerator to reproduce the example outputs.

---

## :gear: API Documentation

The application exposes REST endpoints used by the web UI. All endpoints return JSON (except file downloads).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Upload an image for ContourCreator |
| `/detect_region` | POST | One-click region detection (Flood Fill) |
| `/interactive_segmentation` | POST | Multi-seed segmentation (add/subtract) |
| `/grabcut_segmentation` | POST | Optional GrabCut segmentation |
| `/refine_with_brush` | POST | Apply brush strokes to refine the mask |
| `/export_single_region` | POST | Export one region to CSV (`x,y,region`) |
| `/save_all_coordinates` | POST | Export multiple regions to a single CSV (`x,y,region`) |
| `/upload_csv` | POST | Upload contour CSV for CloudGenerator |
| `/generate_cloud` | POST | Generate cloud (Regular distribution) |
| `/generate_cloud_natural` | POST | Generate cloud (Natural distribution) |
| `/upload_viewer` | POST | Upload CSV for rendering in CloudViewer |
| `/upload_neighbors` | POST | Upload cloud CSV and compute neighbors |
| `/download/<filename>` | GET | Download generated files from the output folder |

---

## :file_cabinet: Data Formats

### :triangular_flag_on_post: Contours CSV (exported by ContourCreator)

Minimal format (single region):

```csv
x,y
0.1234,0.5678
0.1240,0.5681
...
```

Recommended format (multi-region / holes):

```csv
x,y,region
0.1234,0.5678,1
0.1240,0.5681,1
...
```

All coordinates are normalized to the `[0,1] × [0,1]` range (with Y-axis inversion from image coordinates to Cartesian coordinates).

### :cloud: Cloud CSV (exported by CloudGenerator)

```csv
x,y,region,classification
0.1234,0.5678,1,boundary
0.2222,0.3333,1,interior
...
```

---

## :open_file_folder: Project Architecture

```
:package: mGFD_CloudGenerator/
├── app.py                                  # Flask app + API routes
├── requirements.txt                        # Python dependencies
├── vercel.json                             # Vercel deployment config (writes to /tmp)
│
├── contour_modules/                        # ContourCreator (segmentation + contour extraction)
│   └── detection.py
│
├── cloud_modules/                          # Cloud generation pipeline (Regular / Natural)
│   ├── generator.py                        # Orchestrates the full cloud workflow
│   ├── point_generation.py                 # Sampling algorithms (grid / Poisson disk)
│   ├── classification.py                   # Boundary vs interior labeling
│   ├── data_processing.py                  # CSV loaders / validation
│   ├── reduction.py                        # Contour reduction (optional)
│   ├── export.py                           # CSV export utilities
│   ├── visualization.py                    # PNG/SVG rendering
│   └── utils.py
│
├── analysis_modules/                       # Post-processing tools
│   └── neighbors.py                        # Region-constrained kNN neighbor computation
│
├── templates/                              # Web UI (Jinja2)
├── static/                                 # Frontend assets + examples gallery
│   ├── css/
│   ├── js/
│   ├── images/
│   └── examples/
│
├── logs/                                   # local rotating logs (not used on Vercel)
│
└── docs/                                   # Project branding + team images
    ├── logo/
    └── team/
```

Runtime directories created automatically on startup (local mode):
- `uploads/` (temporary uploads)
- `output/` (generated CSV/PNG/SVG)

---

## :books: Mathematical Model

mGFD CloudGenerator is a preprocessing tool, but several steps are driven by simple, explicit mathematical mappings and geometric algorithms.

### :triangular_ruler: Coordinate System & Normalization

ContourCreator works in **normalized coordinates** and exports points to the Cartesian-like convention used by mGFD workflows.

| Concept | UI / Image Space | Exported Space |
|--------|-------------------|----------------|
| Range | $[0,1] \\times [0,1]$ (normalized canvas) | $[0,1] \\times [0,1]$ |
| Origin | top-left | bottom-left |
| Y axis | downwards | upwards (inverted) |

The export transformation applied to each point is:

$$x' = \\mathrm{clip}(x \\cdot s_x, 0, 1), \\quad y' = \\mathrm{clip}((1 - y) \\cdot s_y, 0, 1)$$

Where $(s_x, s_y)$ are scaling factors derived from the source image size:

| Scaling method | $s_x$ | $s_y$ | Notes |
|---------------|------:|------:|------|
| preserve_aspect_ratio (default) | $w/\\max(w,h)$ | $h/\\max(w,h)$ | Preserves aspect ratio in a unit square |
| stretch | $1$ | $1$ | Stretches to fill the unit square |
| custom | user-defined | user-defined | Direct overrides |

### :cloud: Point-Cloud Generation (Regular vs Natural)

Given a set of polygonal contours (optionally multiple regions), CloudGenerator produces boundary and interior nodes:

| Method | Strategy | Typical use |
|--------|----------|-------------|
| Regular Distribution | grid-like sampling inside the polygon | structured clouds, fast generation |
| Natural Distribution | Poisson disk sampling (minimum-distance constraint) | more organic spacing, avoids grid artifacts |

Nodes are exported with metadata:
- `region`: integer region ID (Region 1 is the outer domain; Regions 2+ are interior holes when enabled).
- `classification`: `boundary` or `interior` (used by downstream mGFD pipelines).

### :compass: Neighbor Model (Region-Constrained kNN)

Neighbor computation is performed **per region** using a KD-tree search. This prevents cross-region connectivity that would be invalid for disconnected domains or holes.

---

## :file_cabinet: Dataset Structure

This repository is organized as a web application plus reusable Python modules. When running locally, files are generated in a small set of runtime folders:

```
uploads/                               # temporary uploaded images/CSVs
output/                                # generated CSV/PNG/SVG artifacts
logs/                                  # local rotating logs (if not in Vercel)
static/examples/                       # versioned example inputs/outputs
```

### :package: Output Artifacts

Typical filenames (written under `output/`):
- Contour exports: `<region_name>_<timestamp>.csv`
- Multi-region exports: `<base>_all_regions_<timestamp>.csv`
- Regular clouds: `<input>_cloud_<timestamp>.csv` + PNG/SVG visualizations
- Natural clouds: `<input>_cloud_natural_<timestamp>.csv` + PNG/SVG visualizations

### :hourglass_flowing_sand: Automatic Cleanup

The server periodically removes old files from `uploads/` and `output/` to avoid disk growth in long-running deployments.

---

## :chart_with_upwards_trend: Performance Benchmarks

The application is designed for interactive workloads and scales well for typical mGFD preprocessing sizes. Exact runtimes depend on hardware, point counts, and geometry complexity.

### :stopwatch: Scaling Overview

| Stage | Core operation | Typical scaling |
|-------|----------------|-----------------|
| Segmentation | Flood Fill / interactive refinement | ~ proportional to affected pixels |
| Cloud generation (Regular) | grid sampling + polygon checks | ~ proportional to generated points |
| Cloud generation (Natural) | Poisson disk sampling | ~ proportional to generated points (higher constant) |
| Neighbors | KD-tree kNN per region | ~ $O(n \\log n)$ per region |
| Visualization/Export | CSV + PNG/SVG | ~ proportional to points |

---

## :handshake: Contributing

<div align="center">

### :star2: Contribute to the Project
*Bug reports, feature requests, and pull requests are welcome*

[![Issues](https://img.shields.io/github/issues/gstinoco/mGFD_CloudGenerator?style=flat-square)](https://github.com/gstinoco/mGFD_CloudGenerator/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/gstinoco/mGFD_CloudGenerator?style=flat-square)](https://github.com/gstinoco/mGFD_CloudGenerator/pulls)

</div>

### :bug: Bug Reports
1. **Search existing issues**: Check if the bug has already been reported
2. **Create a detailed report**: Include steps to reproduce and expected vs actual behavior
3. **Provide context**: Operating system, Python version, browser, and relevant parameters (image size, regions, method)

### :bulb: Feature Requests
1. **Describe the feature**: Clear and concise description of the proposed functionality
2. **Justify the need**: Explain how it benefits research, reproducibility, or usability
3. **Provide examples**: Use cases, expected inputs/outputs, and acceptance criteria

### :computer: Code Contributions

```bash
git clone https://github.com/gstinoco/mGFD_CloudGenerator.git
cd mGFD_CloudGenerator

python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
pip install -r requirements.txt

git checkout -b feature/your-feature-name
```

---

## :scientist: Research Team

<div align="center">

### :star2: Meet the Team
*Researchers and graduate students advancing meshless computational methods*

</div>

### :busts_in_silhouette: Main Researchers

<table align="center">
  <thead>
    <tr>
      <th align="center" width="120">Photo</th>
      <th align="left">Researcher</th>
      <th align="left">Affiliation</th>
      <th align="left">Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/gtinoco.webp" alt="Dr. Gerardo Tinoco Guerrero" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Dr. Gerardo Tinoco Guerrero</b> :mexico:<br/>
        <sub>Numerical Methods &amp; Computational Mathematics</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a><br/>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:gerardo.tinoco@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a><br/>
        <a href="https://orcid.org/0000-0003-3119-770X"><img alt="ORCID 0000-0003-3119-770X" src="https://img.shields.io/badge/ORCID-0000--0003--3119--770X-green"></a><br/>
        <a href="https://www.researchgate.net/profile/Gerardo-Tinoco-Guerrero"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/dmota.webp" alt="Dr. Francisco Javier Domínguez Mota" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Dr. Francisco Javier Domínguez Mota</b> :mexico:<br/>
        <sub>Applied Mathematics &amp; Finite Difference Methods</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a><br/>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:francisco.mota@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a><br/>
        <a href="https://orcid.org/0000-0001-6837-172X"><img alt="ORCID 0000-0001-6837-172X" src="https://img.shields.io/badge/ORCID-0000--0001--6837--172X-green"></a><br/>
        <a href="https://www.researchgate.net/profile/Francisco-Dominguez-Mota"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/jagt.webp" alt="Dr. José Alberto Guzmán Torres" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Dr. José Alberto Guzmán Torres</b> :mexico:<br/>
        <sub>Engineering Applications &amp; Artificial Intelligence</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a><br/>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:jose.alberto.guzman@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a><br/>
        <a href="https://orcid.org/0000-0002-9309-9390"><img alt="ORCID 0000-0002-9309-9390" src="https://img.shields.io/badge/ORCID-0000--0002--9309--9390-green"></a><br/>
        <a href="https://www.researchgate.net/profile/Jose-Guzman-Torres"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/harias.webp" alt="Dr. Heriberto Árias Rojas" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Dr. Heriberto Árias Rojas</b> :mexico:<br/>
        <sub>Engineering Applications</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a><br/>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:heriberto.arias@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a><br/>
        <a href="https://orcid.org/0000-0002-7641-8310"><img alt="ORCID 0000-0002-7641-8310" src="https://img.shields.io/badge/ORCID-0000--0002--7641--8310-green"></a><br/>
        <a href="https://www.researchgate.net/profile/Heriberto-Arias-Rojas"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
  </tbody>
</table>

### :mortar_board: Ph.D. Research Students

<table align="center">
  <thead>
    <tr>
      <th align="center" width="120">Photo</th>
      <th align="left">Student</th>
      <th align="left">Institution</th>
      <th align="left">Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/gpj.webp" alt="Gabriela Pedraza-Jiménez" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Gabriela Pedraza-Jiménez</b><br/>
        <img alt="Ph.D. Research Student" src="https://img.shields.io/badge/Ph.D.-Research%20Student-2E8B57?style=flat-square">
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:2220157h@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/eci.webp" alt="Eli Chagolla-Inzunza" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Eli Chagolla-Inzunza</b><br/>
        <img alt="Ph.D. Research Student" src="https://img.shields.io/badge/Ph.D.-Research%20Student-2E8B57?style=flat-square">
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:1137626b@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
      </td>
    </tr>
  </tbody>
</table>

### :mortar_board: M.Sc. Research Students

<table align="center">
  <thead>
    <tr>
      <th align="center" width="120">Photo</th>
      <th align="left">Student</th>
      <th align="left">Institution</th>
      <th align="left">Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/jlgf.webp" alt="Jorge L. González-Figueroa" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Jorge L. González-Figueroa</b><br/>
        <img alt="M.Sc. Research Student" src="https://img.shields.io/badge/M.Sc.-Research%20Student-green?style=flat-square">
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:1718717h@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/cnmb.webp" alt="Christopher N. Magaña-Barocio" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Christopher N. Magaña-Barocio</b><br/>
        <img alt="M.Sc. Research Student" src="https://img.shields.io/badge/M.Sc.-Research%20Student-green?style=flat-square">
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:1339846k@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
      </td>
    </tr>
  </tbody>
</table>

### :mortar_board: Undergraduate Research Students

<table align="center">
  <thead>
    <tr>
      <th align="center" width="120">Photo</th>
      <th align="left">Student</th>
      <th align="left">Institution</th>
      <th align="left">Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="120">
        <img src="docs/team/mgfl.webp" alt="Maria Goretti Fraga Lopez" width="96" height="96" style="border-radius: 50%;">
      </td>
      <td>
        <b>Maria Goretti Fraga-Lopez</b><br/>
        <img alt="Undergraduate Research Student" src="https://img.shields.io/badge/Undergraduate-Research%20Student-green?style=flat-square">
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:1702174b@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
      </td>
    </tr>
  </tbody>
</table>

---

## :factory: Industry Partners Supporting Innovation

<div align="center">

### :star2: Industry Partners Supporting Innovation
*Collaboration between academia and industry to accelerate real-world impact*

</div>

<div align="center">

<table align="center" width="70%">
<tr>
<td align="center">

### :factory: **SIIIA MATH**
#### *Soluciones de Ingeniería, México*

<div align="center">

[![Website](https://img.shields.io/badge/🌐-Visit%20Website-blue?style=for-the-badge)](http://www.siiia.com.mx)
[![Type](https://img.shields.io/badge/📊-R%26D%20Company-orange?style=flat-square)](http://www.siiia.com.mx)
[![Location](https://img.shields.io/badge/📍-Morelia,%20Mexico-green?style=flat-square)](http://www.siiia.com.mx)

</div>

**🎯 Focus areas:**
- Mathematical modeling & simulation
- AI/ML engineering solutions
- Technology transfer and applied R&amp;D

<div align="center">

[![Contact](https://img.shields.io/badge/📧-Partnership%20Contact-0B1B3A?style=for-the-badge)](mailto:gtinoco@siiia.com.mx)

</div>

</td>
</tr>
</table>

</div>

---

## :books: Scientific References

### :books: Core Publications (GFD / mGFD Background)

1. **Tinoco-Guerrero, G.**, Domínguez-Mota, F. J., Guzmán-Torres, J. A., & Tinoco-Ruiz, J. G. (2022). *"Numerical Solution of Diffusion Equation using a Method of Lines and Generalized Finite Differences."* **Revista Internacional de Métodos Numéricos para Cálculo y Diseño en Ingeniería**, 38(2). [DOI: 10.23967/j.rimni.2022.06.003](http://dx.doi.org/10.23967/j.rimni.2022.06.003)

### :trophy: Project Highlights

- **Contour-to-cloud pipeline**: interactive image-based contour extraction and multi-region management
- **Cloud generation methods**: Regular (grid-like) and Natural (Poisson disk sampling) distributions
- **Region-aware analysis**: neighbor computation constrained by region labels for disconnected domains and holes

---

## :memo: Citation & License

If you use this software in your research, please cite:

```bibtex
@software{tinoco2025mGFD_cloudgenerator,
  title={mGFD CloudGenerator 2.1: Web platform for generating 2D unstructured point clouds},
  author={Tinoco-Guerrero, Gerardo and 
          Domínguez-Mota, Francisco Javier and 
          Guzmán-Torres, José Alberto and
          Arias-Rojas, Heriberto},
  year={2025},
  institution={Universidad Michoacana de San Nicolás de Hidalgo},
  organization={SIIIA MATH: Soluciones en ingeniería},
  url={https://github.com/gstinoco/mGFD_CloudGenerator},
  version={2.1},
  note={Web-based preprocessing tool for meshless mGFD workflows: image-to-contour extraction, multi-region handling, point-cloud generation (regular/Poisson), node classification, and region-constrained neighbor analysis}
}
```

### :page_facing_up: License

This project is licensed under the **MIT License** - see the full license text below:

```
MIT License

Copyright (c) 2025 Gerardo Tinoco-Guerrero, Francisco Javier Domínguez-Mota, 
                   José Alberto Guzmán-Torres, Heriberto Árias Rojas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Academic Use:** This software is developed for research and educational purposes. Commercial use is permitted under the MIT License terms.

---

## :pray: Acknowledgments

<div align="center">

### :heart: Special Thanks
*We extend our gratitude to the institutions and partners supporting this research and open-source development*

</div>

### :classical_building: Institutional Support

<table align="center" width="100%" cellspacing="14">
  <tr>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🎓 Universidad Michoacana de San Nicolás de Hidalgo (UMSNH)</b><br/>
          <sub>Academic institution, Mexico</sub><br/><br/>
          <a href="http://www.umich.mx"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-darkred?style=flat-square"></a>
          <img alt="Type: University" src="https://img.shields.io/badge/🏷️%20Type-University-1A3A6B?style=flat-square">
          <img alt="Support: Infrastructure" src="https://img.shields.io/badge/🤝%20Support-Infrastructure-2E8B57?style=flat-square">
        </div>
        <br/>
        <b>Key support</b>
        <ul>
          <li>Academic foundation and research infrastructure</li>
          <li>Scientific training and supervision environment</li>
        </ul>
      </div>
    </td>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🏛️ Secretariat of Science, Humanities, Technology and Innovation(SECIHTI)</b><br/>
          <sub> State Secretariat, Mexico</sub><br/><br/>
          <a href="https://secihti.mx/"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-darkgreen?style=flat-square"></a>
          <img alt="Type: Government" src="https://img.shields.io/badge/🏷️%20Type-Government-2D6A4F?style=flat-square">
          <img alt="Support: Funding and Innovation" src="https://img.shields.io/badge/🤝%20Support-Funding%20%26%20Innovation-40916C?style=flat-square">
        </div>
        <br/>
        <b>Key support</b>
        <ul>
          <li>Support for science and technology initiatives</li>
          <li>Funding and innovation promotion</li>
        </ul>
      </div>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🌿 Centre Internacional de Mètodes Numèrics en Enginyeria (CIMNE)</b><br/>
          <sub>Industry, Spain</sub><br/><br/>
          <a href="https://aulas.cimne.com/aula/aula-morelia/"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-orange?style=flat-square"></a>
          <img alt="Type: Research Center" src="https://img.shields.io/badge/🏷️%20Type-Research%20Center-EE9B00?style=flat-square">
          <img alt="Support: Collaboration" src="https://img.shields.io/badge/🤝%20Support-Collaboration-CA6702?style=flat-square">
        </div>
        <br/>
        <b>Key support</b>
        <ul>
          <li>International collaboration in numerical methods</li>
          <li>Computational engineering research environment</li>
        </ul>
      </div>
    </td>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🏭 SIIIA MATH: Soluciones en Ingeniería</b><br/>
          <sub>Industry, México</sub><br/><br/>
          <a href="http://www.siiia.com.mx"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-blue?style=flat-square"></a>
          <img alt="Type: Industry Partner" src="https://img.shields.io/badge/🏷️%20Type-Industry%20Partner-0B1B3A?style=flat-square">
          <img alt="Support: Technology Transfer" src="https://img.shields.io/badge/🤝%20Support-Technology%20Transfer-1D3557?style=flat-square">
        </div>
        <br/>
        <b>Key support</b>
        <ul>
          <li>Industry-driven applied research and development</li>
          <li>Technology transfer and practical engineering impact</li>
        </ul>
      </div>
    </td>
  </tr>
</table>

### :building_with_garden: Research Centers & Collaborations

<div align="center">

<table align="center" width="100%" cellspacing="14">
  <tr>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🌿 Aula CIMNE-Morelia</b><br/>
          <sub>Research collaboration space</sub><br/><br/>
          <a href="https://aulas.cimne.com/aula/aula-morelia/"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-orange?style=flat-square"></a>
          <img alt="Area: Numerical Methods" src="https://img.shields.io/badge/🧮%20Area-Numerical%20Methods-EE9B00?style=flat-square">
          <img alt="Collaboration: Applied Computing" src="https://img.shields.io/badge/🤝%20Collaboration-Applied%20Computing-CA6702?style=flat-square">
        </div>
        <br/>
        <b>Collaboration highlights</b>
        <ul>
          <li>Numerical methods and computational engineering environment</li>
          <li>Academic–industry collaboration and training activities</li>
        </ul>
      </div>
    </td>
    <td width="50%" valign="top">
      <div style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
        <div align="center">
          <b>🎓 UMSNH</b><br/>
          <sub>Academic collaboration</sub><br/><br/>
          <a href="http://www.umich.mx"><img alt="Website" src="https://img.shields.io/badge/🌐-Website-darkred?style=flat-square"></a>
          <img alt="Type: University" src="https://img.shields.io/badge/🏷️%20Type-University-1A3A6B?style=flat-square">
          <img alt="Support: Research Infrastructure" src="https://img.shields.io/badge/🤝%20Support-Research%20Infrastructure-2E8B57?style=flat-square">
        </div>
        <br/>
        <b>Collaboration highlights</b>
        <ul>
          <li>Institutional infrastructure supporting research and training</li>
          <li>Graduate formation and supervision for scientific computing</li>
        </ul>
      </div>
    </td>
  </tr>
</table>

</div>

### :computer: Technology Communities

<div align="center">

| :package: Framework | :busts_in_silhouette: Community | :star: Contribution |
|:---:|:---:|:---:|
| [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv)](https://opencv.org/) | **OpenCV Community** | Computer vision and image processing |
| [![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/) | **Flask Development Team** | Web framework |
| [![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat-square&logo=numpy)](https://numpy.org/) | **NumPy Community** | Array computing foundation |
| [![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-8CAAE6?style=flat-square&logo=scipy)](https://scipy.org/) | **SciPy Community** | Numerical algorithms |
| [![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)](https://matplotlib.org/) | **Matplotlib Community** | Scientific visualization |
| [![Shapely](https://img.shields.io/badge/Shapely-Geometry-2E8B57?style=flat-square)](https://shapely.readthedocs.io/) | **Shapely Development Team** | Computational geometry |

</div>

---

## :email: Contact & Support

<div align="center">

*Contact channels, technical support, and collaboration opportunities*

[![Issues](https://img.shields.io/badge/🧩-GitHub%20Issues-24292f?style=flat-square&logo=github)](https://github.com/gstinoco/mGFD_CloudGenerator/issues)
[![Email](https://img.shields.io/badge/📧-Email%20Support-blue?style=flat-square)](mailto:gerardo.tinoco@umich.mx)

</div>

<table align="center" width="100%" cellspacing="14">
  <tr>
    <td valign="top" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
      <div align="center">
        <b>Primary Contact</b><br/>
        <sub>Research group coordination</sub>
      </div>
      <br/>
      <b>Dr. Gerardo Tinoco Guerrero</b><br/>
      <sub>Morelia, Michoacán, México</sub>
      <br/><br/>
      <div align="center">
        <a href="mailto:gerardo.tinoco@umich.mx"><img alt="Email" src="https://img.shields.io/badge/📧-Email-blue?style=flat-square"></a>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/🏢%20Company-SIIIA%20MATH-0B1B3A?style=flat-square"></a>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/🎓%20University-UMSNH-1A3A6B?style=flat-square"></a>
      </div>
    </td>
  </tr>
  <tr>
    <td valign="top" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
      <div align="center">
        <b>Technical Support</b><br/>
        <sub>Bug reports, questions, and collaboration requests</sub>
      </div>
      <br/>
      <div align="center">
        <a href="https://github.com/gstinoco/mGFD_CloudGenerator/issues"><img alt="Open an Issue" src="https://img.shields.io/badge/🧩-Open%20Issue-24292f?style=flat-square&logo=github"></a>
        <a href="mailto:gerardo.tinoco@umich.mx"><img alt="Send Email" src="https://img.shields.io/badge/📧-Send%20Email-blue?style=flat-square"></a>
        <a href="mailto:gerardo.tinoco@umich.mx?subject=mGFD%20CloudGenerator%20Collaboration"><img alt="Request Collaboration" src="https://img.shields.io/badge/🤝-Request%20Collaboration-2E8B57?style=flat-square"></a>
      </div>
      <br/>
      <ul>
        <li><b>Issues</b> for bugs and feature requests</li>
        <li><b>Email</b> for technical inquiries</li>
        <li><b>Collaboration</b> for partnerships and joint projects</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td valign="top" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
      <div align="center">
        <b>Collaboration Opportunities</b><br/>
        <sub>Research and engineering partnerships</sub>
      </div>
      <br/>
      <table width="100%">
        <tr>
          <td width="50%"><b>🧮 Meshless Methods</b><br/><sub>mGFD discretizations, boundary handling, point cloud quality</sub></td>
          <td width="50%"><b>📐 Computational Geometry</b><br/><sub>polygon processing, hole handling, robust point-in-region tests</sub></td>
        </tr>
        <tr>
          <td width="50%"><b>🖼️ Computer Vision</b><br/><sub>segmentation workflows, contour extraction from images</sub></td>
          <td width="50%"><b>🌐 Scientific Web Tools</b><br/><sub>reproducible preprocessing platforms for simulation pipelines</sub></td>
        </tr>
        <tr>
          <td width="50%"><b>🌊 CFD / Engineering</b><br/><sub>node generation for complex domains and multi-region problems</sub></td>
          <td width="50%"></td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td valign="top" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
      <div align="center">
        <b>Student Opportunities</b><br/>
        <sub>Projects and training in scientific computing</sub>
      </div>
      <br/>
      <ul>
        <li><b>Graduate Programs</b>: research opportunities with the team</li>
        <li><b>Undergraduate Projects</b>: thesis topics in computational engineering</li>
        <li><b>Internships</b>: scientific computing, numerical methods, and applied modeling</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td valign="top" style="border: 1px solid #d0d7de; border-radius: 12px; padding: 16px;">
      <div align="center">
        <b>Institutional Affiliations</b>
      </div>
      <br/>
      <div align="center">
        <a href="http://www.siiia.com.mx"><img alt="SIIIA MATH" src="https://img.shields.io/badge/🏢-SIIIA%20MATH-0B1B3A?style=flat-square"></a>
        <a href="http://www.umich.mx"><img alt="UMSNH" src="https://img.shields.io/badge/🎓-UMSNH-1A3A6B?style=flat-square"></a>
        <img alt="Research Group" src="https://img.shields.io/badge/🔬-Numerical%20Methods%20%26%20Scientific%20Computing-555?style=flat-square">
      </div>
    </td>
  </tr>
</table>

---

## :speech_balloon: FAQ

<details>
  <summary><b>Which image formats are supported?</b></summary>
  <br/>
  PNG, JPG/JPEG, GIF, and BMP. Maximum request size is <b>16 MB</b>.
</details>

<details>
  <summary><b>Where are outputs saved when running locally?</b></summary>
  <br/>
  Generated files are written to <code>output/</code>. Uploaded files are stored in <code>uploads/</code>. Both folders are created automatically on startup.
</details>

<details>
  <summary><b>What is the expected CSV format for CloudGenerator?</b></summary>
  <br/>
  Contours: <code>x,y,region</code> (region is optional, but recommended for multi-region). Clouds: <code>x,y,region,classification</code>.
</details>

<details>
  <summary><b>Can I use this in commercial projects?</b></summary>
  <br/>
  Yes. The project is released under the MIT License.
</details>

<details>
  <summary><b>How should I cite this work?</b></summary>
  <br/>
  Use the BibTeX entry in the Citation section and the referenced DOI in Scientific References.
</details>

---

<div align="center">

*Advancing meshless methods through open-source collaboration*

[![GitHub stars](https://img.shields.io/github/stars/gstinoco/mGFD_CloudGenerator?style=social)](https://github.com/gstinoco/mGFD_CloudGenerator/stargazers) [![GitHub forks](https://img.shields.io/github/forks/gstinoco/mGFD_CloudGenerator?style=social)](https://github.com/gstinoco/mGFD_CloudGenerator/network/members) [![GitHub watchers](https://img.shields.io/github/watchers/gstinoco/mGFD_CloudGenerator?style=social)](https://github.com/gstinoco/mGFD_CloudGenerator/watchers)

<br/>

<b>If this project helps your research, please consider giving it a star.</b>

</div>
