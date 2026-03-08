# Multimodal Video Understanding for Predicting and Optimizing Short-Form Content Engagement

<div align="center">

![Version](https://img.shields.io/badge/version-2.2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-purple)

**A research-oriented system for automated highlight detection and engagement prediction in video content**

[Problem](#problem-formulation) • [Methodology](#methodology) • [Architecture](#system-architecture) • [Installation](#installation) • [Usage](#usage) • [Research](#research-contributions)

### [Watch Video Demo](https://drive.google.com/file/d/1wrS7uFJkV_PpwZuZO-V7l6q_E7Gn2i4l/view?usp=sharing)

</div>

---

## Abstract

This project investigates **multimodal video understanding** for predicting and optimizing short-form content engagement. Given a long-form video, the system automatically identifies and extracts segments that are likely to be highly engaging on platforms such as YouTube Shorts, TikTok, and Instagram Reels.

The system combines **textual analysis** (from subtitles), **audio features** (energy, speech patterns), **visual features** (motion, scene changes), and **deep learning models** (CLIP, Wav2Vec2, Sentence Transformers, face emotion detection) through a configurable scoring function. We provide a formal **baseline system** and systematic **ablation studies** to quantify the contribution of each modality.

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Deep Learning Features](#deep-learning-features)
- [Classical Computer Vision Components](#classical-computer-vision-components)
- [Performance Optimizations](#performance-optimizations)
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
- **Semantically coherent** -- tells a complete micro-story
- **Emotionally engaging** -- triggers viewer response  
- **Temporally well-structured** -- good pacing and flow
- **Contextually meaningful** -- understandable without full context

Manually identifying such moments in long videos is:
1. Time-consuming (hours of footage to minutes of highlights)
2. Subjective (what's "engaging" varies by audience)
3. Inconsistent (human fatigue affects quality)

### Formal Problem Statement

Given a video \(V\) of duration \(T\) seconds, we seek to:

1. **Segment** the video into candidate clips \(\{S_1, S_2, \ldots, S_n\}\) where each segment \(S_i = (t_{\text{start}}, t_{\text{end}})\) satisfies duration constraints (60-70 seconds for short-form)

2. **Extract features** for each segment across modalities:
   - Text features \(f_t(S_i)\) from subtitles/transcription
   - Audio features \(f_a(S_i)\) from the audio track
   - Visual features \(f_v(S_i)\) from video frames
   - Deep features \(f_d(S_i)\) from pretrained neural networks

3. **Score** each segment using an engagement function:
   \[E(S_i) = w_t \cdot f_t(S_i) + w_a \cdot f_a(S_i) + w_v \cdot f_v(S_i)\]

4. **Rank** segments by engagement score and return top-\(k\)

### Research Questions

- **RQ1**: How much does each modality contribute to engagement prediction?
- **RQ2**: Are multimodal features superior to unimodal baselines?
- **RQ3**: What feature combinations are most predictive of engagement?
- **RQ4**: Do deep learning features improve over classical signal-level features on CPU?

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

#### Deep Learning Features (CPU-friendly)

```
- CLIP ViT-B/32 visual semantics (scene understanding)
- Wav2Vec2 audio emotion (speech emotion recognition)
- Sentence Transformers text embeddings (semantic density)
- Face emotion detection via MediaPipe (facial expressions)
```

> **Research Note**: We explicitly separate signal-level features, classical CV features, and deep learning features to enable ablation studies measuring the contribution of each category.

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
| `full_multimodal` | Y | Y | Y | Y | Complete system (reference) |
| `full_no_cv` | Y | Y | Y | N | Measure CV contribution |
| `text_only` | Y | N | N | N | Text baseline |
| `audio_only` | N | Y | N | N | Audio baseline |
| `visual_only` | N | N | Y | Y | Visual with CV |
| `visual_signal_only` | N | N | Y | N | Visual signal-level only |
| `text_audio` | Y | Y | N | N | No visual features |

**Metrics**:
- Spearman's rho (rank correlation with full system)
- Kendall's tau (concordance)
- Top-K agreement (overlap in top selections)

---

## Deep Learning Features

### Overview

The system includes four CPU-friendly deep learning models that provide semantic understanding beyond classical signal-level features. All models use **lazy loading** (loaded on first use, kept in memory) and run entirely on CPU.

| Model | Task | Size | What it captures |
|-------|------|------|------------------|
| **CLIP ViT-B/32** | Visual semantics | ~600 MB | Scene understanding, object richness |
| **Wav2Vec2** | Audio emotion | ~1.3 GB | Speech emotion (angry, happy, sad, etc.) |
| **all-MiniLM-L6-v2** | Text embeddings | ~90 MB | Semantic density, topic coherence |
| **MediaPipe** | Face detection | ~5 MB | Face count, presence detection |

### CLIP Visual Semantics

Uses OpenAI's CLIP (Contrastive Language-Image Pre-training) to map video frames into a shared text-image embedding space.

**Features extracted:**
- `clip_embedding_mean` -- Mean CLIP embedding across sampled frames
- `clip_semantic_variance` -- Embedding variance (visual diversity within segment)
- `semantic_scene_change_rate` -- Rate of large embedding shifts between frames
- `object_richness_score` -- Embedding norm as proxy for visual complexity

**Adaptive sampling**: Frames are sampled uniformly across the segment, capped at 15 frames regardless of segment length, to keep inference tractable on CPU.

### Wav2Vec2 Audio Emotion

Uses a fine-tuned Wav2Vec2 model for speech emotion recognition.

**Features extracted:**
- `audio_emotion_label` -- Dominant detected emotion (angry, happy, sad, neutral, etc.)
- `audio_emotion_confidence` -- Confidence score
- `audio_emotion_valence` -- Positive/negative emotional valence (-1 to 1)
- `audio_excitement_score` -- Composite excitement measure from high-arousal emotions

Audio is capped at 30 seconds per segment to limit compute.

### Sentence Transformer Text Embeddings

Uses `all-MiniLM-L6-v2` to produce 384-dimensional dense vectors capturing semantic meaning.

**Features extracted:**
- `text_semantic_density` -- Embedding norm as information density proxy
- `text_embedding_variance` -- Variance across sentence embeddings (topic diversity)
- `text_coherence_score` -- Average cosine similarity between consecutive sentences

### Face Emotion Detection

Uses MediaPipe for face detection (with optional FER for emotion classification).

**Features extracted:**
- `face_emotion_label` -- Dominant facial emotion
- `face_emotion_confidence` -- Detection confidence
- `face_count_mean` -- Average number of faces per frame
- `face_emotion_diversity` -- Number of distinct emotions detected

### JSON Output Structure

Deep features are nested under `deep_features` in the segment output:

```json
{
  "deep_features": {
    "visual_deep_features": {
      "clip_semantic_variance": 0.0012,
      "semantic_scene_change_rate": 0.15,
      "object_richness_score": 0.98
    },
    "audio_deep_features": {
      "audio_emotion_label": "happy",
      "audio_emotion_confidence": 0.85,
      "audio_emotion_valence": 0.8,
      "audio_excitement_score": 0.72
    },
    "text_deep_features": {
      "text_semantic_density": 4.21,
      "text_embedding_variance": 0.003,
      "text_coherence_score": 0.89
    },
    "face_deep_features": {
      "face_emotion_label": "happy",
      "face_emotion_confidence": 0.75,
      "face_count_mean": 1.5,
      "face_emotion_diversity": 2
    }
  }
}
```

---

## Classical Computer Vision Components

This section documents the **foundational computer vision techniques** implemented in the visual feature extraction module.

### Why Classical CV?

While deep learning approaches (CLIP, ViT) offer powerful semantic understanding, classical CV techniques provide:

1. **Interpretability** -- Features have clear geometric/statistical meaning
2. **Efficiency** -- No GPU required, minimal compute
3. **Foundation** -- Core concepts underlying modern methods
4. **Ablation** -- Measurable contribution to engagement scoring

### Implemented Techniques

#### 1. Contrast (Image Texture Measure)

```
contrast = std(grayscale_pixels) / 127.5
```

Measures intra-frame intensity variation. High contrast indicates rich texture and sharp edges.

#### 2. Canny Edge Detection

```python
edges = cv2.Canny(gray, low_threshold=50, high_threshold=150)
edge_density = edge_pixels / total_pixels
```

Multi-stage algorithm: Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis thresholding.

#### 3. Temporal Frame Differencing

```python
motion = |frame[t] - frame[t-1]|
significant_motion = motion > threshold
motion_magnitude = sqrt(coverage * strength)
```

Temporal derivative estimation for motion detection. Thresholding (>10 pixels) filters compression artifacts.

#### 4. Histogram-Based Scene Detection

```python
hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
chi_square = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
```

Detects shot boundaries by comparing color histogram distributions. More robust to camera motion than pixel differencing.

---

## Performance Optimizations

The system is designed to handle videos of **1 hour or more** efficiently on CPU hardware.

### Parallel Deep Feature Extraction

All four deep learning extractors (CLIP, Wav2Vec2, Sentence Transformers, Face Detection) run **concurrently** via `ThreadPoolExecutor`. Since they operate on different data (video frames, audio, text, face crops), they can overlap their I/O and compute.

### Parallel Segment Processing

When multiple segments are detected, basic features (text, audio, visual, CV) are extracted in parallel across CPU cores using `ProcessPoolExecutor`. Deep learning features are applied in the main process afterwards to avoid loading heavy models in every worker.

```
max_workers = min(os.cpu_count(), 8)
```

### Adaptive Frame Sampling

For CLIP and face detection, frames are sampled **uniformly across the segment** with adaptive caps:
- **CLIP**: Max 15 frames per segment (regardless of duration)
- **Face detection**: Max 10 frames per segment
- Sampling interval automatically increases for longer segments

### Disk-Based Feature Caching

Feature extraction results are cached to disk. When you re-analyze the same video with the same settings, features are loaded from cache instantly instead of being recomputed.

### Model Lazy Loading

All deep learning models are loaded on first use and kept in memory as singletons. The first run downloads and loads models (one-time cost), but subsequent runs start in seconds.

### FFmpeg Auto-Resolution

FFmpeg is automatically resolved from:
1. System PATH
2. `imageio_ffmpeg` bundled binary (installed with moviepy)

No manual FFmpeg installation required if moviepy is installed.

---

## System Architecture

```
+-------------------------------------------------------------------+
|                        FRONTEND (HTML/JS)                         |
|  +----------+ +--------------+ +------------+ +----------------+ |
|  | Upload   | | Video Player | | Analysis   | | Results Panel  | |
|  | Zone     | |              | | Controls   | | + Deep Features| |
|  +----+-----+ +------+-------+ +-----+------+ +-------+--------+ |
+-------+---------------+---------------+----------------+----------+
        |                |               |                |
        v                v               v                v
+-------------------------------------------------------------------+
|                      REST API (Flask)                              |
|  /upload  /cut  /api/pipeline/run  /api/pipeline/ablation         |
+-------------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------------+
|                    RESEARCH PIPELINE                               |
|                                                                   |
|  +----------+  +-------------+  +--------------------------+      |
|  | Video    |  | Transcription|  | Segment Detection       |      |
|  | Ingest   +->| (Whisper)   +->| (Pause/Semantic/LLM)    |      |
|  +----------+  +-------------+  +------------+-------------+      |
|                                              |                    |
|  +-------------------------------------------v-----------------+ |
|  |            Feature Extraction (Parallel)                     | |
|  | +-------+ +-------+ +--------+ +---------------------------+| |
|  | | Text  | | Audio | | Visual | | Deep Learning             || |
|  | |       | |       | | + CV   | | CLIP | Wav2Vec2 | MiniLM  || |
|  | +---+---+ +---+---+ +---+----+ +--+---+----+----+----+----+| |
|  |     +----------+---------+--------+--------+---------+      | |
|  +-------------------------------------------+------------------+ |
|                                              v                    |
|  +---------------------------------------------------------+     |
|  |  Scoring & Ranking                                       |     |
|  |  Normalize -> Weight & Score -> Rank (Top-K)             |     |
|  +---------------------------------------------------------+     |
|                    |                                              |
|                    v                                              |
|  +---------------------------------------------------------+     |
|  |  Output: AnalysisResult (JSON, cached to disk)           |     |
|  +---------------------------------------------------------+     |
+-------------------------------------------------------------------+
```

### Pipeline Stages

| Stage | Input | Output | Cacheable |
|-------|-------|--------|-----------|
| `video_ingest` | Video file | VideoMetadata | No |
| `transcription` | Video file | SubtitleData | Yes |
| `segment_detection` | Subtitles | Segment candidates | No |
| `feature_extraction` | Segments + Video | SegmentFeatures | Yes |
| `scoring` | Features | Scored segments | No |
| `output` | Scores | AnalysisResult | No |

---

## Baseline vs Enhanced System

### Baseline System (v1.0)

```
Video -> Whisper -> Subtitles -> Gemini -> Segments -> Cut
```

- Single modality (text only), LLM-based, non-deterministic, no evaluation.

### Enhanced System (v2.2)

```
Video -> Pipeline[6 Stages] -> Features[4 Categories] -> Score -> Rank
           |                         |                      |
        Caching              Ablation Modes           Explanations
```

| Component | v1.0 | v2.2 |
|-----------|------|------|
| Modalities | Text only | Text + Audio + Visual + Deep Learning |
| Segmentation | LLM-based | Algorithmic (pause/semantic/hybrid) |
| Features | None (black-box) | 30+ explainable features |
| Vision | None | Classical CV + CLIP ViT-B/32 |
| Audio | None | Signal-level + Wav2Vec2 emotion |
| Text | Subtitles only | VADER + Sentence Transformer embeddings |
| Face Analysis | None | MediaPipe face detection |
| Processing | Sequential | Parallel (multi-process + multi-thread) |
| Caching | None | Transcription + feature disk cache |
| Scoring | LLM opinion | Configurable weighted functions |
| Evaluation | None | Ablation framework |

---

## Technical Stack

### Backend
| Component | Technology |
|-----------|------------|
| Web Framework | Flask 2.x |
| Video Processing | FFmpeg (auto-resolved), MoviePy |
| Computer Vision | OpenCV (cv2) |
| Visual Semantics | CLIP ViT-B/32 (transformers) |
| Audio Emotion | Wav2Vec2 (transformers) |
| Text Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Face Detection | MediaPipe |
| Speech-to-Text | faster-whisper (CTranslate2) |
| LLM (optional) | Google Gemini |
| Parallelism | ProcessPoolExecutor + ThreadPoolExecutor |
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
- 8GB+ RAM (for deep learning models)
- FFmpeg (auto-resolved from moviepy if not on PATH)

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
python run.py
```

### First Run

On the first analysis, deep learning models will be downloaded from HuggingFace Hub and cached locally (~2 GB total). Subsequent runs load from cache in seconds.

| Model | Download Size | First Load | Cached Load |
|-------|--------------|------------|-------------|
| CLIP ViT-B/32 | ~600 MB | ~2 min | ~3s |
| Wav2Vec2 Emotion | ~1.3 GB | ~5 min | ~5s |
| all-MiniLM-L6-v2 | ~90 MB | ~30s | ~1s |
| MediaPipe | ~5 MB | instant | instant |

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
5. View ranked segments with scores and deep learning features
6. Click **"View detailed features"** to see all extracted metrics
7. Preview or cut selected segments

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
    score = seg['score']['total_score']
    deep = seg['features'].get('deep_features', {})
    emotion = deep.get('audio_deep_features', {}).get('audio_emotion_label', 'N/A')
    print(f"  {seg['start_seconds']:.1f}-{seg['end_seconds']:.1f}: "
          f"score={score:.3f}, emotion={emotion}")
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
│   │   ├── deep_features/      # Deep learning feature extraction
│   │   │   ├── __init__.py     # DeepFeatureExtractor (parallel orchestrator)
│   │   │   ├── clip_features.py       # CLIP ViT-B/32 visual semantics
│   │   │   ├── audio_emotion.py       # Wav2Vec2 speech emotion
│   │   │   ├── text_embeddings.py     # Sentence Transformer embeddings
│   │   │   └── face_emotion.py        # MediaPipe face detection
│   │   │
│   │   ├── features/           # Classical feature extraction
│   │   │   ├── text_features.py
│   │   │   ├── audio_features.py
│   │   │   ├── visual_features.py     # Signal-level + OpenCV CV features
│   │   │   └── extractor.py    # Unified interface + parallel processing
│   │   │
│   │   ├── models/             # Data models and schemas
│   │   │   └── schemas.py      # VideoMetadata, Segment, DeepFeatures, etc.
│   │   │
│   │   ├── pipeline/           # Processing pipeline
│   │   │   ├── stages.py       # Individual stage implementations
│   │   │   ├── pipeline.py     # Pipeline orchestration
│   │   │   ├── base.py         # Stage base class with caching
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
│   │   ├── utils/              # Shared utilities
│   │   │   └── video_utils.py  # FFmpeg auto-resolution, video helpers
│   │   │
│   │   ├── tests/              # Test suite
│   │   │   ├── test_pipeline.py
│   │   │   ├── test_features.py
│   │   │   ├── test_scoring.py
│   │   │   └── ...
│   │   │
│   │   ├── static/             # Frontend
│   │   │   └── index.html      # Single-page app (dark theme UI)
│   │   │
│   │   ├── config.py           # Configuration system
│   │   ├── logging_config.py   # Structured logging
│   │   └── app.py              # Flask application
│   │
│   ├── run.py                  # Server entry point
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
   A composable, stage-based system for video analysis research with disk caching.

2. **Multimodal Feature Extraction**  
   Unified extraction of text, audio, visual, and deep learning features (30+ features total).

3. **CPU-Friendly Deep Learning**  
   CLIP, Wav2Vec2, Sentence Transformers, and face detection running efficiently on CPU with parallel execution and adaptive sampling.

4. **Ablation Framework**  
   Systematic tools for modality contribution analysis.

5. **Reproducible Experiments**  
   Structured logging, disk caching, and configuration for reproducibility.

6. **Parallel Processing**  
   Multi-process segment extraction + multi-thread deep feature extraction for handling long videos.

---

## Limitations and Future Work

### Current Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **No ground truth** | No human-annotated engagement labels | Cannot train supervised models |
| **Rule-based scoring** | Hand-crafted weights, not learned | May not generalize |
| **Single language** | Primarily English support | Limited applicability |
| **CPU inference** | Deep models on CPU can be slow for very long videos | Processing time scales linearly |
| **No real-time** | Batch processing only | Not suitable for live streams |

### Future Work

#### Short-term
- [ ] Implement learned scoring with user feedback
- [ ] Add language detection and multilingual support
- [ ] Integrate deep features into engagement scoring weights

#### Medium-term
- [ ] Create human-annotated engagement dataset
- [ ] Train regression model for engagement prediction
- [ ] Add GPU support for faster deep feature extraction
- [ ] Implement A/B testing framework

#### Long-term
- [ ] End-to-end neural engagement predictor
- [ ] Cross-platform performance correlation
- [ ] Real-time segment streaming
- [ ] Reinforcement learning from engagement metrics

### Completed Milestones

- [x] Add CLIP/ViT visual embeddings
- [x] Add face detection and expression analysis
- [x] CPU parallel processing (multi-process + multi-thread)
- [x] Deep learning feature extraction (4 models)
- [x] Disk-based feature caching
- [x] FFmpeg auto-resolution (no manual install needed)
- [x] Frontend deep learning feature display

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

- **OpenAI** for Whisper speech-to-text and CLIP visual model
- **Google** for Gemini multimodal reasoning and MediaPipe
- **Hugging Face** for transformers and model hosting
- **FFmpeg** and **MoviePy** for video processing
- **OpenCV** for classical computer vision algorithms
- The open-source community for foundational tools

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{movie_shorts_2025,
  title = {Multimodal Video Understanding for Short-Form Content Engagement},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/movie-shorts}
}
```

---

<div align="center">
Made with care for video research
</div>
