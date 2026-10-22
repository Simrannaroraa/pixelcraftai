# PixelCraft AI — Intelligent Image Editing & Natural-Language Search Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Zero Paid Credits](https://img.shields.io/badge/Budget-%E2%82%B90%20Zero%20Paid%20Credits-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> **PixelCraft AI** is an AI-first image editing and natural-language retrieval platform built from the ground up to demonstrate production-grade full-stack Python, Generative AI architecture, and computer vision—running entirely on **₹0 budget with zero paid external API credits**.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [User Interface & Screenshots Instructions](#3-user-interface--screenshots-instructions)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture Diagram](#5-system-architecture-diagram)
6. [Project Directory Structure](#6-project-directory-structure)
7. [Local Installation](#7-local-installation)
8. [Virtual Environment Setup](#8-virtual-environment-setup)
9. [How to Run the Application](#9-how-to-run-the-application)
10. [API Configuration](#10-api-configuration)
11. [Free-Mode Guarantees & Constraints](#11-free-mode-guarantees--constraints)
12. [Image Editing Limitations & AI Honesty](#12-image-editing-limitations--ai-honesty)
13. [Embedding Pipeline & Retrieval Details](#13-embedding-pipeline--retrieval-details)
14. [Database & Version History Design](#14-database--version-history-design)
15. [Automated Testing Suite](#15-automated-testing-suite)
16. [Troubleshooting Guide](#16-troubleshooting-guide)
17. [Deployment Instructions](#17-deployment-instructions)
18. [Future Improvements](#18-future-improvements)

---

## 1. Project Overview

PixelCraft AI solves the challenge of organizing, understanding, and editing images through natural-language prompts without locking users into expensive SaaS subscriptions or paid cloud API tokens.

Users can upload images, automatically analyze visual composition, execute complex edits using conversational commands (e.g., *"Remove the background"*, *"Change the background to a snowy mountain"*, *"Make it brighter and warmer"*, *"Apply vintage 70s filter"*, or *"Crop to square 1:1"*), and instantly search across their entire library using semantic concepts (e.g., *"golden sunset on beach"*, *"snowy mountains"*).

Every edit is non-destructive, recorded in an ACID SQLite database with full version branching, rollback, and side-by-side before/after comparison previews.

---

## 2. Key Features

- **🖼️ Smart Image Library**:
  - Secure upload validation (verifies file extension and actual binary header using PIL `verify()`).
  - Automatic heuristic visual scene analysis (extracts dominant color temperature, brightness, contrast dispersion, and tags).
  - Responsive thumbnail gallery with file metadata and one-click actions.
- **🎨 Natural-Language Image Editing**:
  - Conversational command interpretation (e.g., *"Remove the background"*, *"Change the background to a snowy mountain"*, *"Make the image brighter and warmer"*).
  - High-precision local AI background removal using the deep learning **U2-Net** neural network via `rembg`.
  - Realistic procedural background compositing (snowy alpine mountains, sunset gradients, dark studio vignettes, or custom images).
  - Color science adjustments (warmth/temperature, tint balance, saturation, sepia, black & white).
  - Lighting & tone enhancements (brightness, contrast, sharpness, exposure).
  - Artistic stylization (vintage retro, cinematic teal & orange, cyberpunk neon, pencil sketch line art, film noir).
  - Aspect ratio cropping (1:1 square, 16:9 cinematic, 4:3 standard, 9:16 story) and rotation.
- **📜 Non-Destructive Version History**:
  - The original image file remains **strictly preserved and immutable** on disk.
  - Every successful edit creates an incremental version with unique UUID, parent lineage, method tags, and edit prompts.
  - Interactive version timeline with rollback, preview, and download.
- **🔍 Natural-Language Semantic Search**:
  - Vector embedding retrieval using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
  - L2 normalized cosine similarity ranking over image captions, tags, and edit prompts.
  - Real-time search index synchronization when images are uploaded, edited, recaptioned, or deleted.
  - Automatic, zero-dependency **Scikit-learn TF-IDF fallback** if the dense neural model is offline.
  - Transparent UI indicator badges displaying the active retrieval engine.
- **⚙️ Self-Diagnostic Dashboard**:
  - Live inspection of active embedding engines, segmentation models, disk footprint, and database records.
  - One-click search index rebuilder.

---

## 3. User Interface & Screenshots Instructions

The application uses a custom-styled dark SaaS interface built with Streamlit and glassmorphism CSS:

### Recommended Portfolio Screenshots:
1. **Image Library Grid**: Capture `🖼️ Image Library` showing uploaded cards with badges, metadata, and tags.
2. **Side-by-Side Editor**: In `🎨 Image Editor`, run *"Change the background to a snowy mountain"* or *"Remove the background"*, select **Before & After**, and capture the split preview.
3. **Version History Timeline**: Capture the bottom timeline showing Version 1, Version 2, and the original image with rollback buttons.
4. **Natural-Language Search**: In `🔍 Smart Search`, search for *"snowy mountain"* or *"sunset"* to show matching results with percentage match bars and the active engine badge.
5. **System Diagnostics**: In `⚙️ System & Diagnostics`, capture the model status and database health cards.

---

## 4. Technology Stack

| Layer | Technologies Used | Rationale |
| :--- | :--- | :--- |
| **User Interface** | Streamlit 1.40+, Custom CSS | Responsive, interactive, rapid data-app presentation |
| **Image Processing** | Pillow 10+, NumPy, SciPy | Fast, vectorized, local pixel transformations |
| **Local AI Segmentation** | `rembg` (U2-Net), `onnxruntime` | High-quality background isolation with ₹0 external cost |
| **Semantic Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | High-efficiency 384-d dense sentence embeddings |
| **Search Fallback** | `scikit-learn` (`TfidfVectorizer`) | Lightweight keyword retrieval fallback with 0 GPU/model needs |
| **Vector Similarity** | NumPy Matrix Dot Product | Instant cosine similarity over normalized vector arrays |
| **Persistence** | SQLite3, Local Filesystem | Zero-config, robust ACID transactions, foreign keys |
| **Quality Assurance** | Pytest 8+, Pytest-Pythonpath | Automated unit & integration tests covering all services |

---

## 5. System Architecture Diagram

```mermaid
graph TD
    User([👤 User / Browser]) -->|HTTP / WebSocket| UI[🖥️ Streamlit Presentation Layer (app.py)]

    subgraph Service Tier
        UI -->|Upload & Validate| StorageSvc[Storage Service (services/storage.py)]
        UI -->|Natural-Language Commands| EditSvc[Editing Service (services/editing.py)]
        UI -->|Version History & Rollback| VerSvc[Versioning Service (services/versioning.py)]
        UI -->|Natural-Language Query| SearchSvc[Search Service (services/search.py)]
        StorageSvc -->|Visual Profiling| CaptionSvc[Captioning Service (services/captioning.py)]
        EditSvc -->|Log Version Record| VerSvc
        CaptionSvc -->|Trigger Embedding Sync| SearchSvc
        VerSvc -->|Trigger Embedding Sync| SearchSvc
    end

    subgraph Local AI & Processing Engines
        EditSvc -->|Offline Neural Segmentation| RembgAI[✂️ rembg (U2-Net ONNX)]
        EditSvc -->|Vectorized Filters| PillowEngine[🎨 Pillow & NumPy Vectorized Filters]
        EditSvc -.->|Optional Cloud Inpainting| CloudAI[☁️ Optional External Vision/GenAI APIs]
        SearchSvc -->|Primary Dense Retrieval| MiniLMAI[🧠 sentence-transformers (all-MiniLM-L6-v2)]
        SearchSvc -->|Zero-Dependency Fallback| TfidfEngine[📊 Scikit-Learn TfidfVectorizer]
    end

    subgraph Persistent Storage
        StorageSvc -->|Atomic Write| UploadsDir[📁 data/uploads/ (Originals Preserved)]
        VerSvc -->|Atomic Write| VersionsDir[📁 data/versions/ (Edited Increments)]
        VerSvc -->|ACID Schema| SQLiteDB[(🗄️ SQLite: images, versions tables)]
        SearchSvc -->|Store Dense Embeddings| VectorStore[(🗄️ SQLite: embeddings table - BLOBs)]
    end
```

---

## 6. Project Directory Structure

```text
ai-image-editor/
├── app.py                     # Main Streamlit web application & UI router
├── requirements.txt           # Verified Python dependencies
├── pytest.ini                 # Pytest configuration and pythonpath
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules for data, db, and secrets
├── prompts/
│   ├── __init__.py
│   └── templates.py           # Reusable, documented prompt templates
├── services/
│   ├── __init__.py
│   ├── database.py            # SQLite schema, migrations, and CRUD operations
│   ├── storage.py             # File validation, size limits, and disk I/O
│   ├── captioning.py          # Local visual analysis and scene profiling
│   ├── editing.py             # Local transformations, rembg, and NL parser
│   ├── versioning.py          # Non-destructive version lineage and rollback
│   ├── embeddings.py          # MiniLM embeddings and Scikit-learn TF-IDF
│   └── search.py              # Semantic vector search and index synchronization
├── data/
│   ├── uploads/               # Saved original images (never modified)
│   ├── versions/              # Saved edited versions
│   └── thumbnails/            # Cached preview assets
├── docs/
│   └── architecture.md        # Deep architectural design documentation
└── tests/
    ├── __init__.py
    ├── test_database.py       # SQLite CRUD, cascades, and transactions
    ├── test_storage.py        # File validation, headers, and deletion
    ├── test_editing.py        # NL parser, transforms, and generative honesty
    ├── test_versioning.py     # Version creation and original preservation
    └── test_embeddings_search.py # Vectors, TF-IDF, and sync lifecycle
```

---

## 7. Local Installation

Ensure Python 3.11 or higher is installed:
```bash
python3 --version
```

Clone or navigate into the repository:
```bash
cd ai-image-editor
```

---

## 8. Virtual Environment Setup

Create and activate an isolated virtual environment:

### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 9. How to Run the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will start and automatically open in your browser at:
`http://localhost:8501`

---

## 10. API Configuration

PixelCraft AI **does not require any external API keys** for its complete baseline workflow.

If you wish to experiment with optional cloud generative services:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your optional keys:
   ```env
   MAX_UPLOAD_SIZE_MB=15.0
   OPENAI_API_KEY=your_optional_openai_key_here
   GEMINI_API_KEY=your_optional_gemini_key_here
   ```
If left blank, PixelCraft AI transparently remains in 100% free, offline mode.

---

## 11. Free-Mode Guarantees & Constraints

- **Budget:** Exactly ₹0.
- **Zero Paid Subscriptions:** No mandatory cloud billing or third-party metered usage.
- **Secrets Safety:** No API keys are hardcoded or committed to version control.
- **Offline Integrity:** All core features (background removal, editing, versioning, captioning, search) execute locally on CPU/GPU.

---

## 12. Image Editing Limitations & AI Honesty

In accordance with strict engineering ethics:
1. **Honest AI Disclosure**:
   - Background removal is executed using **U2-Net** deep learning models (`rembg`).
   - Tone, warmth, and color adjustments are applied via vectorized **Pillow & NumPy** routines.
   - The UI clearly labels the exact method used for every edit.
2. **Generative Limits**:
   - The zero-cost baseline does not hallucinate new semantic subjects (e.g., if a user prompts *"Add a tree in the background"* or *"Add a sports car"*).
   - Rather than pretending a conventional filter performed generative object synthesis, the platform explicitly explains that adding arbitrary new objects requires a generative text-to-image diffusion API (e.g. OpenAI DALL-E or Stability AI).
   - The original image file remains 100% pristine and unaltered.

---

## 13. Embedding Pipeline & Retrieval Details

PixelCraft AI implements a two-tier retrieval pipeline:

### 1. Primary: Dense Semantic Embeddings (`all-MiniLM-L6-v2`)
- Computes 384-dimensional dense vectors using mean pooling over transformer token representations.
- Vectors are L2 normalized ($\|\mathbf{v}\|_2 = 1.0$), ensuring that cosine similarity is equivalent to a fast matrix dot product.
- Vectors are serialized into raw 32-bit float byte buffers (1,536 bytes) and stored as SQLite `BLOB` records.

### 2. Secondary Fallback: Scikit-learn TF-IDF Cosine Similarity
- If model download or initialization fails (e.g. in restricted air-gapped environments), PixelCraft AI seamlessly falls back to `TfidfVectorizer` with sublinear term-frequency scaling and unigram + bigram support.
- The UI visibly indicates when the fallback mode is active.

---

## 14. Database & Version History Design

### SQLite Schema:
- **`images`**: Stores original file metadata, dimensions, file size, generated captions, and searchable tags.
- **`versions`**: Stores version sequence number, parent version ID, file path, edit prompt, applied parameters (JSON), and timestamp. Uses `FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE`.
- **`embeddings`**: Stores serialized 384-d vector BLOBs, model identifier, and indexed text snippet.

### Integrity Features:
- Foreign key constraints enabled (`PRAGMA foreign_keys = ON;`).
- Atomic transactions with rollback protection on errors.
- Deleting an image automatically cascades to remove its versions and embedding vectors, while cleaning up all associated files from disk.

---

## 15. Automated Testing Suite

The repository includes a comprehensive `pytest` suite with 19 passing tests covering all critical paths:

```bash
pytest tests/ -v
```

### Test Coverage Highlights:
- ✅ `test_storage.py`: Valid image acceptance, corrupt file detection, empty file handling, extension filtering, and safe deletion.
- ✅ `test_database.py`: CRUD operations, metadata updating, version sequence numbering, and cascade deletion.
- ✅ `test_editing.py`: Natural-language instruction parser, lighting enhancement, grayscale conversion, aspect ratio cropping, and honest generative refusal.
- ✅ `test_versioning.py`: Strict preservation of original image bytes across edits, version record persistence, and failed edit rollback without orphaned records.
- ✅ `test_embeddings_search.py`: Vector normalization, cosine similarity calculation, TF-IDF fallback search, and index synchronization lifecycle.

---

## 16. Troubleshooting Guide

- **Issue: `ModuleNotFoundError: No module named 'rembg'` or `onnxruntime`**
  - *Fix:* Ensure dependencies are installed via `pip install rembg onnxruntime`. If onnxruntime has platform issues on your OS, the system automatically falls back to Pillow luminosity thresholding.
- **Issue: First-time search is slow**
  - *Explanation:* On the very first search query, `sentence-transformers` downloads the lightweight `all-MiniLM-L6-v2` weights (~90 MB) into your local Hugging Face cache. Subsequent searches execute in milliseconds.
- **Issue: Database locked error**
  - *Fix:* SQLite uses WAL mode and a 30-second timeout. Ensure only one instance of the app is writing simultaneously.

---

## 17. Deployment Instructions

PixelCraft AI is ready for deployment to free hosting platforms such as **Streamlit Community Cloud** or **Hugging Face Spaces**:

### Deploying to Streamlit Community Cloud:
1. Push this repository to GitHub (ensure personal uploaded images and SQLite files are excluded via `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, set branch to `main`, and main file path to `app.py`.
4. Add any optional environment variables in the Streamlit Cloud Settings.
5. Click **Deploy!**

---

## 18. Future Improvements

- Add client-side interactive canvas bounding box selection for localized inpainting.
- Add local quantized Stable Diffusion / LCM (Latent Consistency Model) on supported hardware with Apple Silicon MPS / CUDA for offline generative additions.
- Add multi-image batch processing and zip export.
- Integrate FAISS HNSW indexing for libraries exceeding 100,000 images.

---

## License

MIT License. Designed and engineered for academic and professional portfolio presentation.
