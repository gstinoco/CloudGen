# Skin Diffusion Simulator with GFD :dna:

<div align="center">

<img src="docs/logo/logo.png" alt="Skin Diffusion Simulator with GFD logo" width="680" style="margin: 20px 0;">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/gstinoco/Skin-Diffusion-GFD) [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) [![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange.svg)](https://numpy.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-red.svg)](https://matplotlib.org/) [![Numba](https://img.shields.io/badge/Numba-0.54+-purple.svg)](https://numba.pydata.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Computational Framework for Skin Diffusion Modeling**

*Advanced numerical simulation using Generalized Finite Differences for biomedical applications*

### :link: Quick Links
[![🚀 Quick Start](https://img.shields.io/badge/🚀-Quick%20Start-green)](#rocket-quick-start) [![📦 Install](https://img.shields.io/badge/📦-Install-blue)](#package-installation--setup) [![🧮 Model](https://img.shields.io/badge/🧮-Model-purple)](#books-mathematical-model) [![🎬 Visualizations](https://img.shields.io/badge/🎬-Visualizations-purple)](#movie_camera-visualizations) [![👥 Team](https://img.shields.io/badge/👥-Research%20Team-blue)](#scientist-research-team) [![🤝 Contribute](https://img.shields.io/badge/🤝-Contributing-orange)](#handshake-contributing) [![🏭 Partners](https://img.shields.io/badge/🏭-Industry%20Partners-0B1B3A)](#factory-industry-partners-supporting-innovation) [![🙏 Thanks](https://img.shields.io/badge/🙏-Acknowledgments-darkgreen)](#pray-acknowledgments)

</div>

---

## :clipboard: Table of Contents
- [Overview](#star2-overview)
- [Features](#sparkles-features)
- [Installation & Setup](#package-installation--setup)
- [Quick Start](#rocket-quick-start)
- [Usage Guide](#book-usage-guide)
- [Visualizations](#movie_camera-visualizations)
- [Project Architecture](#open_file_folder-project-architecture)
- [Mathematical Model](#books-mathematical-model)
- [Dataset Structure](#file_cabinet-dataset-structure)
- [Performance Benchmarks](#chart_with_upwards_trend-performance-benchmarks)
- [Contributing](#handshake-contributing)
- [Research Team](#scientist-research-team)
- [Industry Partners Supporting Innovation](#factory-industry-partners-supporting-innovation)
- [Citation & License](#memo-citation--license)
- [Acknowledgments](#pray-acknowledgments)
- [Contact](#email-contact--support)
- [FAQ](#speech_balloon-faq)

---

## :star2: Overview

This repository presents a **state-of-the-art computational framework** for modeling substance diffusion in biological tissues using the Generalized Finite Differences (GFD) method. The simulator provides high-performance solutions for studying drug delivery, cosmetic penetration, and biomedical transport phenomena in skin layers.

### :wrench: Key Capabilities
- **:dna: Biological Modeling**: 2D transient diffusion equation solver for skin tissue simulation
- **:zap: High-Performance Computing**: Numba JIT compilation for maximum computational efficiency
- **:dart: Scientific Accuracy**: GFD method implementation for irregular mesh handling
- **:arrows_counterclockwise: Automated Dataset Generation**: Parallel processing for large-scale parameter studies
- **:bar_chart: Advanced Visualization**: Comprehensive plotting and analysis tools

### :microscope: Applications

| Field | Application | Use Case |
|-------|-------------|----------|
| **Pharmacology** :pill: | Drug Delivery | Transdermal absorption studies, dosage optimization |
| **Cosmetics** :lipstick: | Skin Penetration | Formulation analysis, safety assessment |
| **Dermatology** :hospital: | Clinical Research | Pathological transport, treatment efficacy |
| **Biomedical Engineering** :gear: | Device Design | Patch development, delivery system optimization |
| **Machine Learning** :robot: | AI Training | Dataset generation, pattern recognition |

---

## :sparkles: Features

### :abacus: Numerical Modeling
- **2D Transient Diffusion Solver**: GFD implementation with 9-point stencil
- **Flexible Boundary Conditions**: Mixed Dirichlet-Neumann conditions for realistic modeling
- **Adaptive Time Stepping**: CFL condition enforcement for numerical stability
- **Irregular Mesh Support**: GFD method handles complex geometries

### :zap: High-Performance Computing
- **Numba JIT Compilation**: Just-in-time optimization for critical functions
- **Vectorized Operations**: NumPy-based efficient array computations
- **Memory Optimization**: Efficient data structures for large-scale simulations
- **Parallel Processing**: Multi-core support for dataset generation

### :bar_chart: Data Generation & Analysis
- **Large-Scale Dataset Creation**: Configurable parameter sweeps (e.g., 90,000 simulations per region mesh for a 100×900 sweep)
- **Parameter Space Exploration**: Systematic variation of diffusion coefficients and initial conditions
- **Automated Data Management**: Hierarchical organization and compression
- **Scientific Visualization**: Advanced plotting with Matplotlib

### :dart: Biomedical Applications
- **Skin Layer Modeling**: Realistic tissue geometry representation
- **Substance Transport**: Drug, cosmetic, and chemical diffusion simulation
- **Clinical Validation**: Framework for experimental data comparison
- **Predictive Modeling**: Machine learning dataset preparation

---

## :package: Installation & Setup

### :computer: System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.9+ |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 5 GB | 25 GB+ (for datasets) |
| **OS** | Windows/Linux/macOS | Linux (optimal performance) |

### :clipboard: Dependencies

```python
# Core scientific computing
numpy >= 1.20.0           # Numerical computations
scipy >= 1.7.0            # Scientific algorithms
matplotlib >= 3.4.0       # Scientific plotting
numba >= 0.54.0           # JIT compilation
tqdm >= 4.62.0            # Progress bars
```

### Quick Installation

```bash
# Method 1: Direct installation
git clone https://github.com/gstinoco/Skin-Diffusion-GFD.git
cd Skin-Diffusion-GFD
pip install -r requirements.txt

# Method 2: Virtual environment (recommended)
python -m venv skin_diffusion_env
source skin_diffusion_env/bin/activate  # On Windows: skin_diffusion_env\Scripts\activate
pip install -r requirements.txt

# Method 3: Conda environment
conda create -n skin_gfd python=3.9
conda activate skin_gfd
pip install -r requirements.txt
```

### :white_check_mark: Installation Verification

```bash
# Test installation
python -c "import numpy, scipy, matplotlib, numba, tqdm; print(':white_check_mark: Installation successful!')"

# Run quick demo
python GFD_skin.py
```

---

## :rocket: Quick Start

### :zap: Single Simulation (Recommended)
```bash
# Run basic skin diffusion simulation
python GFD_skin.py
```

### :construction: Dataset Generation
```bash
# Generate complete training datasets
python create_dataset.py
```

### :wrench: Advanced Usage Examples
```bash
# Select a region mesh (.mat) from region/ (e.g., skin061, skin224, skin256)
# For single simulations, change the region file inside GFD_skin.py main()

# Parameter exploration
# Modify diffusion coefficient (nu) and initial concentration (ci) in main()

# Memory optimization for large datasets
# Adjust batch sizes in create_dataset.py
```

### :dna: Available Mesh Configurations
```bash
# Standard resolution meshes
region/skin061.mat        # 61×61 nodes, ~3.7K points
region/skin224.mat        # 224×224 nodes, ~50K points
region/skin256.mat        # 256×256 nodes, ~65K points
region/skin*.mat          # More mesh resolutions available

# Original skin layer
region/skin_base.png      # Base for the geometries
```

---

## :book: Usage Guide

<div align="center">

*Practical workflows for running simulations and generating datasets*

</div>

### :zap: Single Simulation

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
        <pre><code>python GFD_skin.py</code></pre>
      </td>
    </tr>
    <tr>
      <td><b>2) Outputs</b></td>
      <td>
        <b>Images:</b> <code>skin.png</code>, <code>skin2.png</code> (saved in the project root)
      </td>
    </tr>
    <tr>
      <td><b>3) Customize</b></td>
      <td>
        Edit <code>main()</code> in <code>GFD_skin.py</code> to set the region mesh, diffusion coefficient (<code>nu</code>) and inlet concentration (<code>c_i</code>).
      </td>
    </tr>
  </tbody>
</table>

### :construction: Dataset Generation (Interactive)

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
        <pre><code>python create_dataset.py</code></pre>
      </td>
    </tr>
    <tr>
      <td><b>2) Select mesh</b></td>
      <td>
        Choose a region from <code>region/</code> (e.g., <code>skin061</code>, <code>skin224</code>, <code>skin256</code>).
      </td>
    </tr>
    <tr>
      <td><b>3) Configure sweep</b></td>
      <td>
        The script requests a range for <code>nu</code> as integer multipliers of <code>1e-8</code>, plus <code>c_i</code> range, time steps, workers, and optional compression.
      </td>
    </tr>
    <tr>
      <td><b>4) Outputs</b></td>
      <td>
        <b>Images:</b> <code>Dataset/&lt;region&gt;/ci_###/nu_XXXXXXXX.png</code><br/>
        <b>Optional:</b> <code>Dataset/&lt;region&gt;/ci_###.zip</code>
      </td>
    </tr>
  </tbody>
</table>

---

## :movie_camera: Visualizations

### :framed_picture: Sample Visualizations Gallery

#### :microscope: Comparative Diffusion Analysis

**Concentration Initial = 100** | **Different Diffusion Coefficients**

<div align="center">

<table>
  <tr>
    <td align="center">
      <b>Low Diffusion</b><br/>
      <sub>$\nu = 1 \times 10^{-8}$</sub><br/><br/>
      <img src="docs/visualizations/comparison_nu_1e8_ci100.png" alt="Low Diffusion" width="240">
    </td>
    <td align="center">
      <b>Medium Diffusion</b><br/>
      <sub>$\nu = 4.5 \times 10^{-6}$</sub><br/><br/>
      <img src="docs/visualizations/comparison_nu_450e8_ci100.png" alt="Medium Diffusion" width="240">
    </td>
    <td align="center">
      <b>High Diffusion</b><br/>
      <sub>$\nu = 9 \times 10^{-6}$</sub><br/><br/>
      <img src="docs/visualizations/comparison_nu_900e8_ci100.png" alt="High Diffusion" width="240">
    </td>
  </tr>
</table>

</div>

> :bar_chart: **Dataset Scale**: A full sweep is 100 initial conditions × 900 diffusion coefficients (= 90,000 simulations per region mesh)

---

## :open_file_folder: Project Architecture

### Core Components

```
:package: GFD-Skin-ML/
├── GFD_skin.py                             # Main simulator module
│   ├── difusion_skin_jit()                 # JIT-optimized solver
│   ├── difusion_skin()                     # Vectorized solver
│   ├── Gammas()                            # GFD coefficient calculator
│   └── main()                              # Workflow orchestrator
│
├── create_dataset.py                       # Automated dataset generation
│   ├── Parallel processing support         # Multi-core optimization
│   ├── Parameter space exploration         # Systematic variation
│   ├── Automated data management           # File organization
│   └── Memory optimization                 # Efficient resource usage
│
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
│
├── region/                                 # Computational mesh library
│   ├── skin224.mat                         # Standard resolution mesh ($224 \times 224$)
│   ├── skin256.mat                         # High resolution mesh ($256 \times 256$)
│   ├── skin_base.png                       # Geometry visualization
│   └── red files/                          # Mesh generation files
│
├── Dataset/                                # Generated simulation datasets
│   └── <region>/                           # e.g., skin061, skin224, skin256
│       ├── ci_001/                         # Initial concentration folder
│       │   ├── nu_0.00000001.png           # One simulation (nu formatted with 8 decimals)
│       │   └── ...                         # More diffusion coefficients
│       ├── ci_002/
│       ├── ...
│       └── ci_001.zip                      # Optional compressed archive (one per ci_###)
│
└── docs/                                   # Documentation and examples
    └── visualizations/                     # Sample visualization gallery
        ├── comparison_nu_1e8_ci100.png     # Low diffusion coefficient example
        ├── comparison_nu_450e8_ci100.png   # Medium diffusion coefficient example
        └── comparison_nu_900e8_ci100.png   # High diffusion coefficient example
```
---

## :books: Mathematical Model

The simulator solves the **2D transient diffusion equation**:

$$\frac{\partial u}{\partial t} = \nu \nabla^2 u$$

**Where:**
- $u(x,y,t)$: Concentration field [mg/L]
- $\nu$: Diffusion coefficient [m²/s]
- $\nabla^2$: Laplacian operator
- $t$: Time [s]

### :abacus: Numerical Methods

| Component | Method | Description |
|-----------|--------|-------------|
| **Spatial Discretization** | Generalized Finite Differences (GFD) | 9-point stencil for irregular meshes |
| **Temporal Integration** | Explicit Euler | First-order time stepping |
| **Boundary Conditions** | Mixed Dirichlet-Neumann | Inlet concentration + zero-flux boundaries |
| **Stability** | CFL Condition | Automatic time step adjustment |

### :dart: Boundary Conditions

- **Inlet (Dirichlet)**: $u = c_i$ (prescribed concentration)
- **Boundaries (Neumann)**: $\frac{\partial u}{\partial n} = 0$ (zero flux)
- **Initial Condition**: $u(x,y,0) = 0$ (clean tissue)

---

## :file_cabinet: Dataset Structure

The generated datasets follow a hierarchical organization:

```
Dataset/
└── <region>/                        # Selected mesh (e.g., skin061, skin224, skin256)
    ├── ci_001/                      # Initial concentration folder
    │   ├── nu_0.00000001.png        # One simulation (nu formatted with 8 decimals)
    │   ├── nu_0.00000002.png
    │   └── ...                      # More diffusion coefficients
    ├── ci_002/
    ├── ...
    ├── ci_100/
    ├── ci_001.zip                   # Optional compressed archive
    ├── ci_002.zip
    └── ...
```

### :bar_chart: Data Volume

- **Total Images**: $N_{c_i} \times N_{\nu}$ per region (e.g., $100 \times 900 = 90{,}000$ images/region)
- **Storage**: Varies by mesh size and compression (typically several GB per full sweep)
- **Parameters**: Initial concentration ($c_i$) and diffusion coefficient ($\nu$)
- **Resolution**: Defined by the selected region mesh (e.g., $61 \times 61$, $224 \times 224$, $256 \times 256$ nodes)

### :dart: Dataset Applications

| Use Case | Dataset Type | Description |
|----------|--------------|-------------|
| **Machine Learning Training** :robot: | Complete Dataset | Large-scale image dataset per region (e.g., 90,000 images/region for a 100×900 sweep) |
| **Parameter Studies** :chart_with_upwards_trend: | Subset Analysis | Specific parameter ranges |
| **Validation** :white_check_mark: | Test Sets | Independent validation data |
| **Benchmarking** :trophy: | Reference Solutions | Standard test cases |

---

## :chart_with_upwards_trend: Performance Benchmarks

### :stopwatch: Execution Times

| Mesh Size | Nodes | Time Steps | JIT Solver | Vectorized Solver | Memory Usage |
|-----------|-------|------------|------------|-------------------|--------------|
| $224 \times 224$   | 50,176| 1,000      | ~4.5s      | ~12.3s           | ~2.1 GB       |
| $224 \times 224$   | 50,176| 10,000     | ~45s       | ~123s            | ~2.1 GB       |
| $256 \times 256$   | 65,536| 1,000      | ~6.5s      | ~18.7s           | ~2.8 GB       |
| $256 \times 256$   | 65,536| 10,000     | ~65s       | ~187s            | ~2.8 GB       |

*Benchmarks: Intel i7-8700K @ 3.70GHz, 32GB RAM, Python 3.9*

### :rocket: Performance Optimizations

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| **Numba JIT** | 3-4x | Just-in-time compilation of critical loops |
| **Vectorization** | 2-3x | NumPy array operations |
| **Memory Layout** | 1.5x | Contiguous array storage |
| **Parallel Processing** | Nx | Multi-core dataset generation |

### :bar_chart: Scalability Analysis

```text
# Performance scaling with problem size
Nodes vs Time: O(N log N)     # Near-linear scaling
Memory vs Nodes: O(N)         # Linear memory usage
Parallel Efficiency: 85-90%   # Multi-core performance
```

---

## :handshake: Contributing

<div align="center">

### :star2: Contribute to the Project
*Bug reports, feature requests, and pull requests are welcome*

[![Issues](https://img.shields.io/github/issues/gstinoco/Skin-Diffusion-GFD?style=flat-square)](https://github.com/gstinoco/Skin-Diffusion-GFD/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/gstinoco/Skin-Diffusion-GFD?style=flat-square)](https://github.com/gstinoco/Skin-Diffusion-GFD/pulls)

</div>

### :bug: Bug Reports
1. **Search existing issues**: Check if the bug has already been reported
2. **Create a detailed report**: Include steps to reproduce and expected vs actual behavior
3. **Provide context**: Operating system, Python version, and relevant parameters (mesh, $\nu$, $c_i$)

### :bulb: Feature Requests
1. **Describe the feature**: Clear and concise description of the proposed functionality
2. **Justify the need**: Explain how it benefits research or reproducibility
3. **Provide examples**: Use cases, expected inputs/outputs, and acceptance criteria

### :computer: Code Contributions

```bash
git clone https://github.com/yourusername/Skin-Diffusion-GFD.git
cd Skin-Diffusion-GFD

python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
pip install -r requirements.txt

git checkout -b feature/your-feature-name
```

---

## :scientist: Research Team

<div align="center">

### :star2: Meet the Team
*Researchers and graduate students advancing computational diffusion modeling*

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
        <sub>Engineering applications and Artificial Intelligence</sub>
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
        <sub>Engineering applications</sub>
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

</div>

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

### :books: Core Publications

1. **Tinoco-Guerrero, G.**, Domínguez-Mota, F. J., Guzmán-Torres, J. A., & Tinoco-Ruiz, J. G. (2022). *"Numerical Solution of Diffusion Equation using a Method of Lines and Generalized Finite Differences."* **Revista Internacional de Métodos Numéricos para Cálculo y Diseño en Ingeniería**, 38(2). [DOI: 10.23967/j.rimni.2022.06.003](http://dx.doi.org/10.23967/j.rimni.2022.06.003)

### :trophy: Research Achievements

- **Large-Scale Simulation Dataset**: Configurable sweeps (e.g., 90,000 simulations per region mesh for a 100×900 sweep)
- **High-Performance Implementation**: 3-4x speedup with Numba JIT optimization
- **Open Source Framework**: MIT licensed for academic and commercial use
- **Cross-Platform Compatibility**: Windows, Linux, macOS support

---

## :memo: Citation & License

If you use this software in your research, please cite:

```bibtex
@software{gfd_skin_simulator_2025,
  title={Skin Diffusion Simulator with GFD: High-Performance Computational Framework 
         for Biomedical Transport Modeling},
  author={Tinoco-Guerrero, Gerardo and 
          Domínguez-Mota, Francisco Javier and 
          Guzmán-Torres, José Alberto and
          Arias-Rojas, Heriberto},
  year={2025},
  institution={Universidad Michoacana de San Nicolás de Hidalgo},
  organization={SIIIA MATH: Soluciones en ingeniería},
  url={https://github.com/gstinoco/Skin-Diffusion-GFD},
  note={Advanced computational framework for skin diffusion modeling using 
        Generalized Finite Differences method}
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
| [![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat-square&logo=numpy)](https://numpy.org/) | **NumPy Community** | Array computing foundation |
| [![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-8CAAE6?style=flat-square&logo=scipy)](https://scipy.org/) | **SciPy Community** | Numerical algorithms and I/O |
| [![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)](https://matplotlib.org/) | **Matplotlib Community** | Scientific visualization |
| [![Numba](https://img.shields.io/badge/Numba-JIT%20Acceleration-00A3E0?style=flat-square)](https://numba.pydata.org/) | **Numba Community** | High-performance Python |
| [![tqdm](https://img.shields.io/badge/tqdm-Progress%20Bars-FFC107?style=flat-square)](https://tqdm.github.io/) | **tqdm Community** | Progress monitoring |

</div>

---

## :email: Contact & Support

<div align="center">

*Contact channels, technical support, and collaboration opportunities*

[![Issues](https://img.shields.io/badge/🧩-GitHub%20Issues-24292f?style=flat-square&logo=github)](https://github.com/gstinoco/Skin-Diffusion-GFD/issues)
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
        <a href="https://github.com/gstinoco/Skin-Diffusion-GFD/issues"><img alt="Open an Issue" src="https://img.shields.io/badge/🧩-Open%20Issue-24292f?style=flat-square&logo=github"></a>
        <a href="mailto:gerardo.tinoco@umich.mx"><img alt="Send Email" src="https://img.shields.io/badge/📧-Send%20Email-blue?style=flat-square"></a>
        <a href="mailto:gerardo.tinoco@umich.mx?subject=Skin-Diffusion-GFD%20Collaboration"><img alt="Request Collaboration" src="https://img.shields.io/badge/🤝-Request%20Collaboration-2E8B57?style=flat-square"></a>
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
          <td width="50%"><b>🩺 Biomedical Engineering</b><br/><sub>Transdermal delivery systems, medical device design</sub></td>
          <td width="50%"><b>🤖 Machine Learning</b><br/><sub>Diffusion pattern analysis, predictive modeling</sub></td>
        </tr>
        <tr>
          <td width="50%"><b>🧮 Numerical Methods</b><br/><sub>Discretization techniques, solver optimization</sub></td>
          <td width="50%"><b>🧪 Clinical Research</b><br/><sub>Validation with experimental data, clinical applications</sub></td>
        </tr>
        <tr>
          <td width="50%"><b>💊 Pharmaceutical Research</b><br/><sub>Drug delivery optimization, formulation studies</sub></td>
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
        <li><b>Undergraduate Projects</b>: thesis topics in computational biology</li>
        <li><b>Internships</b>: scientific computing and applied modeling</li>
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
  <summary><b>Which meshes are available?</b></summary>
  <br/>
  Check <code>region/</code> for <code>skin*.mat</code> files (e.g., <code>skin061</code>, <code>skin224</code>, <code>skin256</code>).
</details>

<details>
  <summary><b>Where do results go?</b></summary>
  <br/>
  <code>python GFD_skin.py</code> saves <code>skin.png</code> and <code>skin2.png</code> in the project root. Dataset generation saves outputs in <code>Dataset/&lt;region&gt;/</code>.
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

*Advancing biomedical diffusion modeling through open-source collaboration*

[![GitHub stars](https://img.shields.io/github/stars/gstinoco/Skin-Diffusion-GFD?style=social)](https://github.com/gstinoco/Skin-Diffusion-GFD/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gstinoco/Skin-Diffusion-GFD?style=social)](https://github.com/gstinoco/Skin-Diffusion-GFD/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/gstinoco/Skin-Diffusion-GFD?style=social)](https://github.com/gstinoco/Skin-Diffusion-GFD/watchers)

<br/>

<b>If this project helps your research, please consider giving it a star.</b>

</div>
