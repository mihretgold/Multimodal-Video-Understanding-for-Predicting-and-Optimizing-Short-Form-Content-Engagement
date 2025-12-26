# Multimodal Video Understanding for Predicting and Optimizing Short-Form Content Engagement

## Overview

**Multimodal Video Understanding for Predicting and Optimizing Short-Form Content Engagement** is a research-oriented web application that explores how **textual, audio, and temporal information** extracted from videos can be used to automatically identify and extract **highly engaging short-form segments** from long-form video content.

The project investigates the intersection of **visual AI, multimedia signal processing, and multimodal learning**, with a focus on understanding *what makes a video segment engaging* and *how AI systems can assist in content optimization* for platforms such as YouTube Shorts, TikTok, and Instagram Reels.

Rather than relying on naive clipping or fixed heuristics, the system is designed to evolve toward **content-aware and multimodal video understanding**.

---

## Research Motivation

Short-form video platforms prioritize clips that are:
- semantically coherent,
- emotionally engaging,
- temporally well-structured, and
- contextually meaningful.

Manually identifying such moments in long videos is time-consuming and subjective. This project explores how **AI-driven multimodal analysis** can assist in discovering high-impact segments by reasoning over:

- **Textual content** (dialogue, keywords, semantic importance),
- **Temporal structure** (pacing, continuity),
- **Audio presence** (speech vs silence),
- **Narrative flow** across scene boundaries.

The long-term goal is to build a system that supports **automatic highlight detection and engagement-aware video summarization** using scalable AI techniques.

---

## Core Features

### Video Processing
- Upload long-form videos in common formats (mp4, mov, mkv, webm)
- Video preprocessing and trimming using FFmpeg

### Multimodal Analysis (Current Focus: Text + Time)
- Automatic speech-to-text transcription using Whisper
- Extraction and alignment of subtitle timestamps
- Subtitle-driven semantic analysis for identifying moments of interest

### Intelligent Temporal Segmentation
- Identification of candidate short-form segments (≈60–70 seconds)
- Preservation of semantic continuity across cuts
- Multiple segment suggestions per video

### Subtitle-Aware Reasoning
- Uses subtitle density, timing, and content to guide segmentation
- Enables content-aware cutting rather than purely duration-based trimming

### Export
- Download short-form optimized video clips
- Output suitable for major short-video platforms

---

## System Pipeline

The system follows a modular multimodal workflow:

1. **Video Ingestion**
   - Upload and validation of video files
   - Extraction of metadata and duration

2. **Subtitle Acquisition**
   - Extraction of embedded subtitles (when available)
   - Automatic transcription using Whisper for videos without captions

3. **Temporal Alignment**
   - Mapping subtitle timestamps to the video timeline
   - Preparation for segment-level analysis

4. **Content Analysis**
   - Semantic analysis of subtitles
   - Detection of candidate engagement peaks

5. **Segment Selection**
   - Identification of high-impact segments
   - Optimization for short-form narrative coherence

6. **Video Cutting**
   - Precise trimming using FFmpeg and MoviePy
   - Audio-visual synchronization preservation

7. **Preview and Export**
   - User review of AI-suggested segments
   - Download of final clips

---

## Technical Stack

### Frontend
- Next.js 14
- TypeScript
- React
- Tailwind CSS

### Backend
- Python
- Flask
- FFmpeg
- Whisper (OpenAI)
- MoviePy

---

## Planned Research Extensions

This project is intentionally designed to support future research-oriented extensions, including:

- Multimodal feature fusion (text + audio + visual signals)
- Learned engagement prediction models
- Scene boundary detection
- Emotion and sentiment-aware segmentation
- Quantitative evaluation against human-selected highlights
- Dataset creation for short-form video research

---

## Installation

### Prerequisites
- Node.js 18+
- Python 3.8+
- FFmpeg

### Setup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-shorts.git
cd movie-shorts
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the `backend/app` directory with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Start the backend server:
```bash
cd backend/app
python app.py
```

5. Open the application in your browser:
```
http://localhost:5000
```

## Requirements

- Python 3.8+
- FFmpeg
- Google API Key for Gemini
- Modern web browser with JavaScript enabled

## Project Structure

```
movie-shorts/
├── backend/
│   ├── app/
│   │   ├── static/
│   │   │   └── index.html
│   │   ├── uploads/
│   │   ├── cuts/
│   │   ├── subtitles/
│   │   ├── app.py
│   │   └── .env
│   └── requirements.txt
├── Images/
│   ├── design.png
│   ├── s1.png
│   ├── s2.png
│   ├── s3.png
│   ├── s4.png
│   ├── s5.png
│   ├── s6.png
│   ├── s7.png
│   └── s8.png
└── README.md
```
## Usage Workflow

1. Upload a long-form video file through the application interface.  
2. Extract existing subtitles or generate new subtitles using automatic speech recognition.  
3. Run multimodal analysis combining visual, audio, and textual signals to identify high-engagement segments.  
4. Review AI-recommended short-form segments with associated timestamps and engagement rationale.  
5. Select preferred segments and generate optimized short-form clips suitable for platforms like YouTube Shorts, Instagram Reels, and TikTok.  
6. Download or share the generated clips directly.

## Tech Stack

### Frontend
- Next.js 14
- TypeScript
- React
- Tailwind CSS

### Backend
- Python
- Flask
- FFmpeg
- MoviePy
- Whisper (for speech-to-text)
- Gemini (for multimodal and semantic analysis)

## Prerequisites

- Node.js 18 or higher
- Python 3.8 or higher
- FFmpeg
- A modern web browser with JavaScript enabled
- Google API Key (for Gemini)

## Installation

1. Clone the repository.
2. Install backend dependencies using the provided requirements file.
3. Configure environment variables for API keys.
4. Start the backend server.
5. Open the application in your browser and begin uploading videos.


## Contributing

Contributions are welcome and encouraged.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear descriptions.
4. Push to your branch.
5. Open a pull request for review.

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for Whisper
- Google Gemini for multimodal reasoning
- FFmpeg and MoviePy for video processing
- The open-source community for foundational tools and libraries


