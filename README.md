# Multimodal Video Understanding for Predicting and Optimizing Short-Form Content Engagement

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-purple)

**A research-oriented system for automated highlight detection and engagement prediction in video content**

[Problem](#problem-formulation) • [Methodology](#methodology) • [Architecture](#system-architecture) • [Installation](#installation) • [Usage](#usage) • [Research](#research-contributions)

### 🎬 [Watch Video Demo](https://drive.google.com/file/d/1wrS7uFJkV_PpwZuZO-V7l6q_E7Gn2i4l/view?usp=sharing)

</div>

---

## Abstract

This project investigates **multimodal video understanding** for predicting and optimizing short-form content engagement. Given a long-form video, the system automatically identifies and extracts segments that are likely to be highly engaging on platforms such as YouTube Shorts, TikTok, and Instagram Reels.

The system combines **textual analysis** (from subtitles), **audio features** (energy, speech patterns), and **visual features** (motion, scene changes) through a configurable scoring function. We provide a formal **baseline system** and systematic **ablation studies** to quantify the contribution of each modality.

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Baseline vs Enhanced System](#baseline-vs-enhanced-system)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Research Contributions](#research-contributions)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Formulation

### The Challenge

Short-form video platforms prioritize content that is:
- **Semantically coherent** – tells a complete micro-story
- **Emotionally engaging** – triggers viewer response  
- **Temporally well-structured** – good pacing and flow
- **Contextually meaningful** – understandable without full context

Manually identifying such moments in long videos is:
1. Time-consuming (hours of footage → minutes of highlights)
2. Subjective (what's "engaging" varies by audience)
3. Inconsistent (human fatigue affects quality)

### Formal Problem Statement

Given a video $V$ of duration $T$ seconds, we seek to:

1. **Segment** the video into candidate clips $\{S_1, S_2, ..., S_n\}$ where each segment $S_i = (t_{start}, t_{end})$ satisfies duration constraints (60-70 seconds for short-form)

2. **Extract features** for each segment across modalities:
   - Text features $f_t(S_i)$ from subtitles/transcription
   - Audio features $f_a(S_i)$ from the audio track
   - Visual features $f_v(S_i)$ from video frames

3. **Score** each segment using an engagement function:
   $$E(S_i) = w_t \cdot f_t(S_i) + w_a \cdot f_a(S_i) + w_v \cdot f_v(S_i)$$

4. **Rank** segments by engagement score and return top-$k$

### Research Questions

- **RQ1**: How much does each modality contribute to engagement prediction?
- **RQ2**: Are multimodal features superior to unimodal baselines?
- **RQ3**: What feature combinations are most predictive of engagement?

---

## Methodology

### 1. Temporal Segmentation

We implement multiple segmentation strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Pause-Based** | Detects speech pauses and silence gaps | Natural dialogue breaks |
| **Fixed-Window** | Sliding window with overlap | Uniform coverage |
| **Semantic Boundary** | Topic/sentence-level splits | Content-aware cuts |
| **Hybrid** | Combines pause + semantic | Best quality |

**Key Insight**: Pause-based segmentation respects natural speech patterns, leading to more coherent clips than arbitrary time-based cutting.

### 2. Multimodal Feature Extraction

#### Text Features (from subtitles)
```
- Word count, sentence count
- Sentiment polarity (VADER)
- Question/exclamation density
- Speech rate (words per second)
- Keyword presence
```

#### Audio Features (via FFmpeg)
```
- RMS energy (loudness)
- Silence ratio
- Volume dynamics (max - min dB)
- Spectral centroid
```

#### Visual Features (via FFmpeg + OpenCV)

**Signal-Level Features** (basic pixel statistics):
```
- Motion intensity (raw frame difference)
- Scene change count/rate (threshold-based)
- Brightness (mean luminance)
- Color variance
```

**Classical Computer Vision Features** (OpenCV):
```
- Contrast (std of grayscale intensities)
- Edge density (Canny edge detection ratio)
- Edge intensity (Sobel gradient magnitude)
- Motion magnitude (thresholded frame differencing)
- Histogram diff (chi-square distance)
- Scene boundaries (histogram-based cut detection)
```

> **Research Note**: We explicitly separate signal-level features from CV features to enable ablation studies measuring the contribution of classical CV techniques.

### 3. Engagement Scoring

We implement three scoring strategies:

#### Rule-Based Scoring (Baseline)
Hand-crafted rules combining feature thresholds:
```python
score = (
    0.4 * normalize(sentiment) +
    0.3 * normalize(energy) +
    0.2 * normalize(motion) +
    0.1 * keyword_bonus
)
```

#### Normalized Scoring
Statistical normalization across segments:
```python
z_score = (feature - mean) / std
score = sigmoid(weighted_sum(z_scores))
```

#### Learned Scoring (Future)
Train a regressor on human-annotated engagement labels.

### 4. Ablation Studies

Systematic experiments removing modalities to quantify contributions:

| Mode | Text | Audio | Visual | CV | Purpose |
|------|------|-------|--------|-----|---------|
| `full_multimodal` | ✓ | ✓ | ✓ | ✓ | Complete system (reference) |
| `full_no_cv` | ✓ | ✓ | ✓ | ✗ | Measure CV contribution |
| `text_only` | ✓ | ✗ | ✗ | ✗ | Text baseline |
| `audio_only` | ✗ | ✓ | ✗ | ✗ | Audio baseline |
| `visual_only` | ✗ | ✗ | ✓ | ✓ | Visual with CV |
| `visual_signal_only` | ✗ | ✗ | ✓ | ✗ | Visual signal-level only |
| `text_audio` | ✓ | ✓ | ✗ | ✗ | No visual features |

**Metrics**:
- Spearman's ρ (rank correlation with full system)
- Kendall's τ (concordance)
- Top-K agreement (overlap in top selections)

---

## Classical Computer Vision Components

This section documents the **foundational computer vision techniques** implemented in the visual feature extraction module. These are classical CV algorithms commonly taught in introductory vision courses.

### Why Classical CV?

While deep learning approaches (CLIP, ViT) offer powerful semantic understanding, classical CV techniques provide:

1. **Interpretability** — Features have clear geometric/statistical meaning
2. **Efficiency** — No GPU required, runs on CPU
3. **Foundation** — Core concepts underlying modern methods
4. **Ablation** — Measurable contribution to engagement scoring

### Implemented Techniques

#### 1. Contrast (Image Texture Measure)

**Concept**: Standard deviation of grayscale pixel intensities within a frame.

```
contrast = std(grayscale_pixels) / 127.5
```

**CV Foundation**: Contrast measures intra-frame intensity variation, a fundamental image quality metric. High contrast indicates rich texture and sharp edges; low contrast suggests flat, uniform regions.

**Engagement relevance**: Visually complex scenes (high contrast) may be more engaging than flat scenes.

#### 2. Canny Edge Detection

**Concept**: Multi-stage algorithm to detect edges (intensity discontinuities).

```python
edges = cv2.Canny(gray, low_threshold=50, high_threshold=150)
edge_density = edge_pixels / total_pixels
```

**CV Foundation**: 
1. Gaussian smoothing to reduce noise
2. Sobel operators for gradient computation (Gx, Gy)
3. Non-maximum suppression to thin edges
4. Hysteresis thresholding (strong/weak edge linking)

**Features extracted**:
- `edge_density` — Ratio of edge pixels (visual complexity)
- `edge_intensity` — Mean gradient magnitude (edge strength)

**Engagement relevance**: Action sequences and detailed scenes have higher edge density.

#### 3. Temporal Frame Differencing

**Concept**: Detect motion by computing absolute difference between consecutive frames.

```python
motion = |frame[t] - frame[t-1]|
significant_motion = motion > threshold  # Filter noise
motion_magnitude = sqrt(coverage * strength)
```

**CV Foundation**: Temporal derivative estimation for motion detection. Simpler than optical flow but effective for measuring activity level.

**Key improvement over raw differencing**:
- Thresholding (>10 pixels) filters compression artifacts
- Separate coverage (how much moves) and strength (how fast)

**Limitations** (documented for research honesty):
- Cannot distinguish object motion from camera motion
- Sensitive to lighting changes
- Does not capture motion direction

#### 4. Histogram-Based Scene Detection

**Concept**: Detect shot boundaries by comparing color histograms of consecutive frames.

```python
hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
chi_square = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
if chi_square > threshold:
    scene_boundaries.append(timestamp)
```

**CV Foundation**: 
- Histograms summarize global color distribution
- Chi-square distance measures distribution divergence
- More robust to camera motion than pixel differencing

**Features extracted**:
- `histogram_diff_mean` — Average chi-square distance
- `scene_boundaries` — List of detected cut timestamps

### CV Feature Ablation

To measure the contribution of CV features vs. signal-level features:

| Ablation Mode | Visual Features |
|---------------|-----------------|
| `full_multimodal` | Signal-level + CV features |
| `full_no_cv` | Signal-level only (CV zeroed) |
| `visual_only` | CV features enabled |
| `visual_signal_only` | Signal-level only |

**Expected hypothesis**: `full_multimodal` > `full_no_cv` because CV features capture structural information (edges, motion patterns, scene structure) that raw pixel statistics miss.

### JSON Output Structure

Visual features are exposed with clear separation:

```json
{
  "visual_features": {
    "signal_level": {
      "motion_intensity": 0.25,
      "brightness_mean": 0.6,
      "color_variance": 0.3
    },
    "computer_vision": {
      "contrast": 0.45,
      "edge_density": 0.15,
      "edge_intensity": 0.35,
      "motion_magnitude": 0.28,
      "histogram_diff_mean": 0.12,
      "scene_boundaries": [10.5, 25.0]
    }
  }
}
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (HTML/JS)                       │
│  ┌─────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Upload  │  │ Video Player│  │ Analysis  │  │   Results    │  │
│  │  Zone   │  │             │  │  Controls │  │    Panel     │  │
│  └────┬────┘  └──────┬──────┘  └─────┬─────┘  └──────┬───────┘  │
└───────┼──────────────┼───────────────┼───────────────┼──────────┘
        │              │               │               │
        ▼              ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      REST API (Flask)                           │
│  /upload  /cut  /api/pipeline/run  /api/pipeline/ablation       │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PIPELINE                            │
│                                                                 │
│  ┌──────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
│  │ Video    │   │ Transcription│   │  Segment Detection     │   │
│  │ Ingest   │──▶│  (Whisper)  │──▶│  (Pause/Semantic/LLM)  │   │
│  └──────────┘   └─────────────┘   └───────────┬─────────────┘   │
│                                               │                 │
│  ┌──────────────────────────────────────────┐│                  │
│  │         Feature Extraction               ││                  │
│  │  ┌──────┐   ┌───────┐   ┌────────┐      │◀┘                 │
│  │  │ Text │   │ Audio │   │ Visual │      │                    │
│  │  └──┬───┘   └───┬───┘   └───┬────┘      │                    │
│  │     └───────────┼───────────┘           │                    │
│  └─────────────────┼───────────────────────┘                    │
│                    ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Scoring & Ranking                                         │ │
│  │  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐ │ │
│  │  │ Normalize  │─▶│   Weight    │─▶│   Rank by Score     │ │ │
│  │  │  Features  │  │   & Score   │  │   (Top-K)            │ │ │
│  │  └────────────┘  └─────────────┘  └──────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    │                                            │
│                    ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Output: AnalysisResult (JSON)                             │ │
│  │  - Ranked segments with scores                             │ │
│  │  - Feature vectors per segment                             │ │
│  │  - Provenance metadata                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Input | Output | Cacheable |
|-------|-------|--------|-----------|
| `video_ingest` | Video file | VideoMetadata | No |
| `transcription` | Video file | SubtitleData | Yes |
| `segment_detection` | Subtitles | Segment candidates | No |
| `feature_extraction` | Segments + Video | SegmentFeatures | No |
| `scoring` | Features | Scored segments | No |
| `output` | Scores | AnalysisResult | No |

---

## Baseline vs Enhanced System

### Baseline System (v1.0)

The original system used:
- Single modality (text only via subtitles)
- LLM-based segment detection (Gemini API)
- No formal feature extraction
- No reproducible scoring function

```
Video → Whisper → Subtitles → Gemini → Segments → Cut
```

**Limitations**:
- Non-deterministic (LLM outputs vary)
- Black-box reasoning (no explainability)
- Single-modality (ignores audio/visual)
- No quantitative evaluation possible

### Enhanced System (v2.0)

The research-grade system adds:

| Component | Enhancement |
|-----------|-------------|
| **Configuration** | Type-safe dataclasses, environment overrides |
| **Data Models** | Structured schemas with serialization |
| **Segmentation** | Multiple algorithmic strategies |
| **Features** | Real multimodal extraction |
| **Scoring** | Configurable, explainable functions |
| **Logging** | Structured JSON for reproducibility |
| **Ablation** | Systematic modality analysis |
| **Testing** | Comprehensive test suite |

```
Video → Pipeline[6 Stages] → Features[3 Modalities] → Score → Rank
          ↓                        ↓                     ↓
       Logging               Ablation Modes         Explanations
```

---

## Technical Stack

### Backend
| Component | Technology |
|-----------|------------|
| Web Framework | Flask 2.x |
| Video Processing | FFmpeg, MoviePy |
| Computer Vision | OpenCV (cv2) |
| Speech-to-Text | faster-whisper (CTranslate2) |
| LLM (optional) | Google Gemini |
| Data Models | Python dataclasses |
| Logging | Structured JSON (JSONL) |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Vanilla HTML5/CSS3/JS |
| Styling | Custom CSS (dark theme) |
| Fonts | Outfit, JetBrains Mono |
| Icons | Emoji-based |

### Development
| Tool | Purpose |
|------|---------|
| pytest | Testing |
| mypy | Type checking |
| black | Formatting |

---

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (must be in PATH)
- 4GB+ RAM (for Whisper model)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/movie-shorts.git
cd movie-shorts

# Create virtual environment
cd backend
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure Gemini API for legacy mode
echo "GOOGLE_API_KEY=your_key_here" > app/.env

# Run server
python -m app.app
```

### Verify Installation

```bash
# Health check
curl http://localhost:5000/health

# Expected response:
# {"status": "healthy", "version": "2.0.0", "experiment": "default"}
```

---

## Usage

### Web Interface

1. Open **http://localhost:5000** in your browser
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Select an ablation mode (or use Full Multimodal)
4. Click **"Run Analysis"**
5. View ranked segments with scores
6. Preview or cut selected segments

### API Usage

```python
import requests

# Run full pipeline
response = requests.post(
    "http://localhost:5000/api/pipeline/run",
    json={
        "filename": "video.mp4",
        "ablation_mode": "full_multimodal",
        "use_cache": True
    }
)

result = response.json()
print(f"Found {result['segment_count']} segments")
for seg in result['segments']:
    print(f"  {seg['start_seconds']:.1f}-{seg['end_seconds']:.1f}: {seg['score']['total_score']:.3f}")
```

### CLI Scripts

```bash
# Run baseline analysis
python scripts/run_baseline.py --video uploads/video.mp4

# Run ablation study
python scripts/run_ablation.py --video uploads/video.mp4 --output-dir results/

# Visualize segments
python scripts/visualize_segments.py --video uploads/video.mp4
```

---

## Project Structure

```
movie-shorts/
├── backend/
│   ├── app/
│   │   ├── ablation/           # Ablation study framework
│   │   │   ├── analyzer.py     # Statistical analysis
│   │   │   ├── runner.py       # Experiment orchestration
│   │   │   └── report.py       # Report generation
│   │   │
│   │   ├── baseline/           # Baseline system formalization
│   │   │   ├── specification.py
│   │   │   └── runner.py
│   │   │
│   │   ├── features/           # Multimodal feature extraction
│   │   │   ├── text_features.py
│   │   │   ├── audio_features.py
│   │   │   ├── visual_features.py
│   │   │   └── extractor.py    # Unified interface
│   │   │
│   │   ├── models/             # Data models and schemas
│   │   │   └── schemas.py      # VideoMetadata, Segment, etc.
│   │   │
│   │   ├── pipeline/           # Processing pipeline
│   │   │   ├── stages.py       # Individual stage implementations
│   │   │   ├── pipeline.py     # Pipeline orchestration
│   │   │   └── context.py      # Shared state
│   │   │
│   │   ├── scoring/            # Engagement scoring
│   │   │   ├── strategies.py   # Scoring algorithms
│   │   │   ├── normalizers.py  # Feature normalization
│   │   │   ├── ranker.py       # Segment ranking
│   │   │   └── scorer.py       # Main scorer interface
│   │   │
│   │   ├── segmentation/       # Temporal segmentation
│   │   │   ├── strategies.py   # Segmentation algorithms
│   │   │   ├── boundaries.py   # Boundary detection
│   │   │   └── segmenter.py    # Unified interface
│   │   │
│   │   ├── routes/             # API endpoints
│   │   │   ├── pipeline_routes.py
│   │   │   ├── video_routes.py
│   │   │   └── subtitle_routes.py
│   │   │
│   │   ├── services/           # Business logic
│   │   │   ├── subtitle_service.py
│   │   │   └── analysis_service.py
│   │   │
│   │   ├── tests/              # Test suite
│   │   │   ├── test_pipeline.py
│   │   │   ├── test_features.py
│   │   │   ├── test_scoring.py
│   │   │   └── ...
│   │   │
│   │   ├── config.py           # Configuration system
│   │   ├── logging_config.py   # Structured logging
│   │   └── app.py              # Flask application
│   │
│   ├── scripts/                # CLI tools
│   │   ├── run_baseline.py
│   │   ├── run_ablation.py
│   │   └── visualize_segments.py
│   │
│   └── requirements.txt
│
└── README.md
```

---

## API Reference

### Pipeline Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipeline/run` | POST | Run full analysis pipeline |
| `/api/pipeline/ablation` | POST | Run ablation study |
| `/api/pipeline/stages` | GET | List pipeline stages |
| `/api/pipeline/result/<id>` | GET | Get saved result |
| `/api/pipeline/results` | GET | List all results |

### Video Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload video file |
| `/cut` | POST | Cut video segment |
| `/uploads/<filename>` | GET | Serve video |
| `/cuts/<filename>` | GET | Serve cut |

### Subtitle Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/subtitles/check/<filename>` | GET | Check for subtitles |
| `/api/subtitles/get/<filename>` | GET | Get subtitles as JSON |
| `/api/subtitles/extract/<filename>` | GET | Download SRT file |

---

## Research Contributions

This project contributes:

1. **Modular Pipeline Architecture**  
   A composable, stage-based system for video analysis research.

2. **Multimodal Feature Extraction**  
   Unified extraction of text, audio, and visual features.

3. **Ablation Framework**  
   Systematic tools for modality contribution analysis.

4. **Reproducible Experiments**  
   Structured logging and configuration for reproducibility.

5. **Baseline Formalization**  
   Clear specification of inputs, outputs, and methodology.

---

## Limitations and Future Work

### Current Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **No ground truth** | No human-annotated engagement labels | Cannot train supervised models |
| **Rule-based scoring** | Hand-crafted weights, not learned | May not generalize |
| **Single language** | Primarily English support | Limited applicability |
| **No semantic vision** | Classical CV only (no deep learning) | Missing object/scene understanding |
| **No real-time** | Batch processing only | Not suitable for live streams |

### Future Work

#### Short-term
- [ ] Add CLIP/ViT visual embeddings
- [ ] Implement learned scoring with user feedback
- [ ] Add language detection and multilingual support
- [ ] Improve feature caching for large videos

#### Medium-term
- [ ] Create human-annotated engagement dataset
- [ ] Train regression model for engagement prediction
- [ ] Add face detection and expression analysis
- [ ] Implement A/B testing framework

#### Long-term
- [ ] End-to-end neural engagement predictor
- [ ] Cross-platform performance correlation
- [ ] Real-time segment streaming
- [ ] Reinforcement learning from engagement metrics

### Known Issues

1. Large videos (>1 hour) may timeout on feature extraction
2. Whisper transcription can be slow on CPU
3. Visual features require significant disk I/O

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest backend/app/tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style

- Use Black for formatting
- Add type hints where possible
- Write docstrings for public functions
- Include tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **OpenAI** for Whisper speech-to-text
- **Google** for Gemini multimodal reasoning
- **FFmpeg** and **MoviePy** for video processing
- **OpenCV** for classical computer vision algorithms
- The open-source community for foundational tools


---

<div align="center">
Made with ❤️ for video research
</div>
