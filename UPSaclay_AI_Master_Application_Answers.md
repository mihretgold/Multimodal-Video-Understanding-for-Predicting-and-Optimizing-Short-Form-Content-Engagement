# UPSaclay AI Master – Application Answers (Movie-Shorts / Multimodal Video Engagement)

**Applicant:** Mihret Bekele  
**Email:** mihretagegnehu2@gmail.com  
**Print this document as PDF before submitting, then attach the PDF to your application.**

---

## A. Code & Repositories

### Link to one public repo (GitHub/GitLab) where you implemented an ML component

- **Repo URL:** https://github.com/mihretgold/Multimodal-Video-Understanding-for-Predicting-and-Optimizing-Short-Form-Content-Engagement  
- **Backend (core):** https://github.com/mihretgold/Multimodal-Video-Understanding-for-Predicting-and-Optimizing-Short-Form-Content-Engagement/tree/main/backend  
- **File path of core model/scoring code:** `backend/app/scoring/strategies.py` (engagement scoring), `backend/app/features/extractor.py` (multimodal feature extraction), `backend/app/features/text_features.py` (text/sentiment features)  
- **Your username:** mihretgold  
- **Three commit SHAs you authored:** `bde6f5c`, `5697377`, `b616b8f` (verify in your repo history; replace with your own SHAs if different)

### Briefly describe your role (≤50 words)

Designed and implemented a multimodal video analysis pipeline for short-form content highlight detection. Developed feature extraction modules for text, audio, and visual modalities, implemented engagement scoring and ranking algorithms, and built an ablation framework to evaluate modality contributions.

### Paste a 10–20 line snippet from your code (model training or data preprocessing; no boilerplate)

```python
# backend/app/features/text_features.py – sentiment and text feature extraction
def _analyze_sentiment(self, words: List[str]) -> float:
    if not words:
        return 0.0
    positive_count = 0
    negative_count = 0
    negate_next = False
    intensity = 1.0
    for i, word in enumerate(words):
        if word in NEGATORS:
            negate_next = True
            continue
        if word in INTENSIFIERS:
            intensity = 1.5
            continue
        if word in POSITIVE_WORDS:
            if negate_next:
                negative_count += intensity
            else:
                positive_count += intensity
        elif word in NEGATIVE_WORDS:
            if negate_next:
                positive_count += intensity
            else:
                negative_count += intensity
        negate_next = False
        intensity = 1.0
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    score = (positive_count - negative_count) / total
    return max(-1.0, min(1.0, score))
```

### Which lines are yours? Why are they written that way? (≤60 words)

The full `_analyze_sentiment` block is mine. Lexicon-based sentiment with explicit handling of negators (e.g. “not good”) and intensifiers (“very good”) avoids heavy dependencies and keeps the pipeline interpretable and fast. The score is normalized to [-1, 1] so it can be combined with other modalities in the engagement score.

### Show the exact command you used to run the training. Include environment and requirements path.

There is no classical “training” step; the pipeline uses rule-based engagement scoring. To run the full pipeline and ablation (equivalent to evaluation):

```bash
cd backend
pip install -r requirements.txt
python scripts/run_ablation.py --video uploads/video.mp4 --modes text_only audio_only visual_only full_multimodal --output report.json
```

**Environment:** Python 3.8+, pip; optional CUDA for faster-whisper.  
**Requirements path:** `backend/requirements.txt`

---

## B. Data & Reproducibility

### What dataset did you use most recently?

Inputs are **user-uploaded long-form videos** (variable count and duration), not a fixed public dataset. No fixed #samples or #features; segment count depends on video length and segmentation strategy (pause-based, fixed-window, or semantic). Source: local uploads; no external license. Train/val/test: not applicable (unsupervised/rule-based scoring). **Data cleaning:** validated file existence and segment time bounds (start < end, duration within 60–70 s for short-form), and skipped empty or corrupt segments before feature extraction.

### Reproducibility: how did you set seeds and control nondeterminism?

**Seeds:** `random_seed: int = 42` in `backend/app/config.py` (ResearchConfig); used for any stochastic steps and logged in experiment JSON. **Libraries:** config is designed so that any NumPy/Python RNG use can be seeded from this value; Whisper (faster-whisper) and FFmpeg/OpenCV frame extraction are deterministic for the same input. **Remaining nondeterminism:** optional LLM-based segmentation (e.g. Gemini) is non-deterministic; when used, runs can differ. Ablation and scoring with algorithmic segmentation are reproducible.

---

## C. Modeling Decisions

### Task type, model family, why that choice, and most impactful hyperparameter (≤100 words)

**Task:** Ranking/regression of video segments by engagement score (no ground-truth labels). **“Model”:** Hand-crafted engagement function E(S) = w_t·f_t(S) + w_a·f_a(S) + w_v·f_v(S) with rule-based feature combination (no neural net). **Alternatives considered:** (1) Learned regressor (e.g. XGBoost on features)—rejected due to no labels; (2) deep multimodal (e.g. CLIP + audio encoder)—rejected for interpretability and CPU-only deployment. **Choice:** Rule-based scoring so we can run ablations and explain contributions without labels. **Most impactful hyperparameter:** Modality weights (e.g. text=0.4, audio=0.3, visual=0.3). Tuned manually; search range [0.2–0.5] per modality; 0.4 for text gave best balance with text-heavy content.

### Supervision signal: supervised / weakly / semi / self-supervised? Labels or pretext task?

**Unsupervised / rule-based.** No learned model; no labels. “Supervision” is the hand-crafted engagement formula over extracted features (text sentiment/keywords, audio energy/silence, visual motion/scene changes). Future work would use human-annotated or platform engagement labels (views, likes) for supervised learned scoring.

---

## D. Evaluation & Error Analysis

### Primary metric you optimized and one trade-off it introduced (≤60 words)

**Primary metric:** Spearman’s ρ (rank correlation) between ablation modes and the full multimodal ranking; also Top-K agreement (overlap of top segments). **Trade-off:** Optimizing rank correlation favors consistency with the full system, not necessarily true engagement (we have no engagement labels). So we report both correlation and Top-1/Top-3 agreement, and document that the “reference” is the full multimodal system, not human or platform metrics.

### One concrete failure mode from error analysis, cause, and fix attempt (≤100 words)

**Failure:** Segments with very high scene-change rate were over-scored and ranked too high, producing jarring “best” clips. **Cause:** Scene-change count was normalized and weighted (0.3) without capping; rapid cuts inflated the visual score. **Fix:** In `RuleBasedScoring._score_visual` we added a penalty when scene_norm > 0.7: `if scene_norm > 0.7: scene_norm = 1.0 - (scene_norm - 0.7) / 0.3`, so excessive scene changes reduce the score. This improved subjective quality of top-ranked segments in internal tests.

### Final validation log line (loss/metric), checkpoint filename or SHA, and overfitting (≤60 words)

No trained model or validation loss. **Output:** Ablation report JSON with per-mode metrics (e.g. `rank_correlation`, `top_1_agreement`, `segment_count`). Results saved under `backend/app/experiments/` (e.g. `ablation_ablation_study.json`). **Overfitting:** N/A (rule-based). No train/val split; we guard against overfitting to a single video by running ablations on multiple videos and reporting Spearman and Top-K agreement across modes.

---

## E. Compute & Systems

### Hardware used (CPU/GPU, RAM/VRAM)

CPU-only for deployment (e.g. Render); development on a typical laptop (multi-core CPU, 8–16 GB RAM). Optional GPU for faster-whisper (not required). No discrete GPU or VRAM for the scoring/feature code.

### Longest single run time and how you monitored it (≤50 words)

Longest runs: full pipeline + ablation on ~1 h videos, roughly 10–20 minutes depending on resolution and Whisper model. Monitored via structured JSON logs (backend `logging_config`), wall-clock time per stage in pipeline context, and ablation report fields `execution_time_seconds` per mode.

### Did you profile bottlenecks? One profiler excerpt and what you changed (≤80 words)

No formal PyTorch/tensor profiler (no deep learning training). Bottlenecks identified via logging and experience: (1) Whisper transcription (CPU-bound), (2) visual feature extraction (FFmpeg + OpenCV frame decode). **Change:** Reduced `visual_sample_fps` in config to sample fewer frames per segment; added optional feature caching in `FeatureExtractor` to avoid re-extracting for the same segment when re-running ablations. This cut repeated run time on the same video.

---

## F. MLOps & Engineering Hygiene

### How did you track experiments? One experiment ID/run URL or table row and what decision it informed (≤60 words)

Experiments tracked via **structured JSON logs** and **saved result/ablation JSON files** under `backend/app/experiments/` and `backend/app/logs/`. No W&B/MLflow. Example: `ablation_ablation_study.json` contains per-mode `execution_time_seconds`, `segment_count`, and top-segment scores. A run comparing `text_only` vs `full_multimodal` showed higher Top-1 agreement when visual features were included, which supported keeping all three modalities in the default config.

### Testing: one unit/integration test for data or model code (what it checks, where it lives) (≤60 words)

**Unit test:** `backend/app/tests/test_features.py` – e.g. `test_text_sentiment_positive` and `test_text_sentiment_negative` call `TextFeatureExtractor().extract(text)` and assert sign and rough magnitude of `sentiment_score`, and that word/sentence counts match expectations. This checks that the text feature and sentiment pipeline behaves correctly and regresses when we change lexicons or normalization.

---

## G. Teamwork & Contribution

### Describe a merge request/PR you opened: link, title, and main reviewer comment you addressed (≤60 words)

Example from the repo history: **PR “TEAM-2-be-update-arch”** (merge commit `5563c82`) – added routes and architecture updates. If you have a specific PR link (e.g. on GitHub), paste it here. **Reviewer comment to address:** e.g. “Add tests for the new pipeline route” → added or extended tests in `backend/app/tests/test_pipeline.py` for the new endpoint. (Replace with your actual PR link and reviewer feedback.)

### If you worked in a team, what would break without your contribution? (≤50 words)

Without my contribution: the **multimodal feature extractor** (`features/extractor.py`) and **ablation runner** (`ablation/runner.py`) would be missing, so the pipeline would not support text+audio+visual features or systematic modality ablations. The **engagement scoring logic** (`scoring/strategies.py`) and **normalization** would also be absent, so no unified E(S) or comparable scores across modalities.

---

## H. Responsible & Legal AI

### One dataset bias or limitation and how you measured or mitigated it (≤80 words)

**Bias/limitation:** (1) **Language:** Sentiment and keyword lexicons are English-only; non-English subtitles get weak or misleading text features. (2) **Content:** Training-free rules favor “high energy” and “high motion”; calm or slow-paced but engaging content can be under-scored. **Measurement:** Inspected top-ranked segments on a few non-English and calm-content videos. **Mitigation:** Documented English-only and energy/motion bias in the README; recommended language detection and multilingual lexicons (or learned scoring with diverse labels) as future work.

### Licensing: license of code/models you used and compatibility with your repo (≤50 words)

**Our repo:** MIT (see README and LICENSE). **Dependencies:** FFmpeg (LGPL), OpenCV (Apache 2.0), faster-whisper (MIT), Flask, NumPy, etc. (permissive). **Models:** Whisper weights used via faster-whisper (MIT). No proprietary datasets. MIT is compatible with these; we do not ship FFmpeg binary, only assume it is installed. No license conflict.

---

## I. Math & Understanding

### Exact loss function minimized (symbolically) and one regularization term (≤60 words)

No learned loss: the system minimizes no differentiable objective. Engagement is defined as **E(S) = w_t·f_t(S) + w_a·f_a(S) + w_v·f_v(S)** with f_* normalized to [0,1] and w_t + w_a + w_v = 1. **No regularization term.** For the optional learned scoring (placeholder): would minimize MSE or ranking loss on engagement labels (e.g. normalized views); L2 on weights would be the natural regularizer.

### If you used cross-validation or early stopping: patience/folds and selection criterion (≤40 words)

No cross-validation or early stopping: no trained model. Ablation “modes” are different feature subsets (e.g. text_only, full_multimodal); we compare them with Spearman ρ and Top-K agreement on the same videos, without a train/val split.

---

## Deployment-oriented detail (M2 applicants)

**Dockerfile snippet** (Hugging Face Spaces / containerized deployment):

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=tiny
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app.app:create_app()"]
```

**API route** (Flask): `POST /api/pipeline/run` – accepts `filename`, `ablation_mode`, `use_cache`; runs the full pipeline and returns `result_id`, `segments`, `processing_time_seconds`. See `backend/app/routes/pipeline_routes.py`.

---

## Integrity Check

**I confirm the repositories and logs referenced above are my own work or clearly indicate collaborators. I understand that committees may check commit histories and contact supervisors.**

*(Sign or initial here when you submit.)*

---

**REMEMBER: Print this document as PDF before submitting, then attach the PDF to your application.**
