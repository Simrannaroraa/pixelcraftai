# PixelCraft AI — System Architecture & Technical Specifications

PixelCraft AI is an offline-capable, portfolio-grade AI-powered image editing and natural-language search platform built entirely on a **₹0 / zero-paid-credit budget**. The system pairs deep local neural network models (`sentence-transformers/all-MiniLM-L6-v2` and `rembg` U2-Net) with deterministic pixel transformations and an ACID-compliant SQLite versioning engine.

---

## 1. High-Level System Architecture Diagram

```mermaid
graph TD
    User([👤 User / Web Client]) -->|Interacts via Browser| UI[🖥️ Streamlit Presentation Layer (app.py)]

    subgraph Core Application Services
        UI -->|Upload & Validate| UploadSvc[Storage Service (services/storage.py)]
        UI -->|Natural-Language Commands| EditSvc[Editing Engine (services/editing.py)]
        UI -->|Version Queries & Rollback| VerSvc[Versioning Service (services/versioning.py)]
        UI -->|Natural-Language Query| SearchSvc[Search & Retrieval (services/search.py)]
        UploadSvc -->|Visual Profiling| CaptionSvc[Captioning Service (services/captioning.py)]
        EditSvc -->|Log Version Record| VerSvc
        CaptionSvc -->|Trigger Embedding Sync| SearchSvc
        VerSvc -->|Trigger Embedding Sync| SearchSvc
    end

    subgraph Processing Engines
        EditSvc -->|Local AI Segmentation| RembgEngine[✂️ rembg (U2-Net Neural Net)]
        EditSvc -->|Deterministic Transformations| PillowEngine[🎨 Pillow & NumPy Vectorized Filters]
        EditSvc -.->|Optional Cloud Inpainting| CloudAI[☁️ Optional Vision/GenAI APIs]
        SearchSvc -->|Primary Dense Retrieval| MiniLMEngine[🧠 sentence-transformers (all-MiniLM-L6-v2)]
        SearchSvc -->|Zero-Dependency Fallback| TfidfEngine[📊 Scikit-Learn TfidfVectorizer]
    end

    subgraph Persistence Layer (Local File System & SQLite)
        UploadSvc -->|Atomic Write| UploadsDir[📁 data/uploads/ (Originals Untouched)]
        VerSvc -->|Atomic Write| VersionsDir[📁 data/versions/ (Edited Increments)]
        VerSvc -->|ACID Schema| SQLiteDB[(🗄️ SQLite Database: images & versions tables)]
        SearchSvc -->|Store Dense Embeddings| VectorStore[(🗄️ SQLite Database: embeddings table - BLOBs)]
    end
```

---

## 2. Core Architectural Lifecycles

### A. How Uploads Move Through the System
1. **Intake & Verification**: The user uploads a file through the drag-and-drop zone in Streamlit.
2. **Security & Content Integrity**:
   - `storage.validate_image_file()` checks the file extension against an allowlist (`jpg`, `jpeg`, `png`, `webp`) and enforces max file size constraints (default 15 MB).
   - PIL's `Image.open().verify()` inspects the magic byte header to prevent malicious or disguised binaries from being saved.
3. **Atomic Disk Write**: The file is assigned a random UUIDv4 filename (e.g., `3f5c71b1-....png`) and written to `data/uploads/`. Path traversal attacks are structurally impossible.
4. **Local Visual Analysis**:
   - `captioning.analyze_image_locally()` analyzes pixel statistics (aspect ratio, average luminance, contrast standard deviation, warmth bias, and saturation).
   - Generates an objective descriptive caption and extracted tags (e.g., `high-key, bright, daylight, landscape, wide-angle`).
5. **Database Transaction**: Metadata (`id`, `filename`, `file_path`, `width`, `height`, `file_size`, `caption`, `tags`) is committed to the SQLite `images` table.
6. **Search Index Synchronization**: `search.sync_image_index(image_id)` immediately computes a 384-dimensional normalized vector using `all-MiniLM-L6-v2` and persists the vector BLOB to the `embeddings` table.

---

### B. How Image Edits Produce Versions (Non-Destructive Lineage)
1. **Parent Resolution**: The system identifies the current parent image (either the pristine original from `data/uploads/` or a previously selected version from `data/versions/`).
2. **Zero-Destruction Guarantee**: The parent file and original upload are opened read-only. Their bytes on disk are **never modified or overwritten**.
3. **Instruction Interpretation**:
   - The user inputs an instruction (e.g., *"Remove the background"*, *"Change the background to a snowy mountain"*, *"Make it brighter and warmer"*, or *"Apply vintage 70s filter"*).
   - `editing.parse_natural_language_instruction()` parses the instruction into an action and parameter dictionary.
   - If the request asks for impossible generative additions without a configured cloud model (e.g., *"Add a tree"*), the engine declines with a clear, honest explanation rather than fabricating an edit.
4. **Execution**:
   - For background removal: local U2-Net deep neural network (`rembg`) segments the subject cleanly.
   - For background replacement: foreground is composited over high-resolution procedural scenic backdrops (snowy mountain, sunset, dark studio, white) or user-supplied image.
   - For lighting/color/styling: vectorized NumPy arrays and Pillow filters apply exact color science adjustments.
5. **Persistence**:
   - The edited PIL Image is saved to `data/versions/` with a version-specific filename: `v_{image_id}_num{n}_{uuid[:8]}.png`.
   - `versioning.create_new_version()` records `parent_version_id`, `version_number`, `edit_prompt`, `method`, `parameters` (JSON), and dimensions in the `versions` table.
6. **Rollback & Lineage**: The user can preview before/after comparisons, download any historical version, or branch an edit off an earlier version in the timeline.

---

### C. Embedding Generation & Natural-Language Retrieval
1. **Text Serialization**: An image's searchable context is formed by concatenating its caption, scene description, tags, original filename, and latest edit prompt:
   $$\text{SearchableText} = \text{Caption} \parallel \text{Description} \parallel \text{Tags} \parallel \text{Filename} \parallel \text{LatestEdit}$$
2. **Vector Generation**:
   - `sentence-transformers/all-MiniLM-L6-v2` maps the text into a dense 384-dimensional continuous vector space.
   - Numerical validation checks that all values are finite and non-NaN.
   - L2 unit normalization is applied:
     $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
3. **Vector Storage**: The 384 float32 numbers (1,536 bytes) are saved as an SQLite `BLOB` in the `embeddings` table with a foreign key to `images(id)`.
4. **Query Execution**:
   - When a user enters a search query (e.g., *"mountain with snow"* or *"golden sunset"*), the query is embedded into a normalized vector $\mathbf{q}$.
   - Cosine similarity is computed via matrix dot product against all candidate vectors:
     $$\text{sim}(\mathbf{q}, \mathbf{d}_i) = \mathbf{q} \cdot \mathbf{d}_i$$
   - Matches are ranked in descending order and filtered by a user-configurable confidence threshold.

---

### D. Zero-Dependency TF-IDF Fallback
If the sentence embedding model cannot be loaded (due to low memory or offline air-gapped environments):
- The search engine automatically activates `TfidfSearchFallback` using Scikit-learn's `TfidfVectorizer`.
- Computes sublinear term-frequency inverse-document-frequency vectors with bigram support (`ngram_range=(1, 2)`).
- Ranks candidates using sparse cosine similarity.
- The UI displays an explicit indicator badge: `Keyword TF-IDF Search (Scikit-Learn Fallback)` so users always know which engine is running.

---

## 3. Component Offline & Cost Matrix

| Component | Offline Capable? | Cost / Billing | Technology / Provider |
| :--- | :---: | :---: | :--- |
| **Streamlit SaaS UI** | ✅ Yes | ₹0 | Local Python Web Server |
| **Metadata & Version Database** | ✅ Yes | ₹0 | SQLite3 (Embedded ACID DB) |
| **Image Storage & Validation** | ✅ Yes | ₹0 | Local Disk + Pillow verify() |
| **AI Background Removal** | ✅ Yes | ₹0 | U2-Net Neural Net via `rembg` (CPU/GPU) |
| **Lighting & Color Adjustments** | ✅ Yes | ₹0 | Vectorized NumPy + Pillow ImageEnhance |
| **Procedural Backdrops & Compositing** | ✅ Yes | ₹0 | Local Pillow ImageDraw + Alpha Compositing |
| **Heuristic Visual Captioning** | ✅ Yes | ₹0 | Local Color & Luminance Statistics |
| **Semantic Embedding Search** | ✅ Yes | ₹0 | `sentence-transformers/all-MiniLM-L6-v2` |
| **Keyword Search Fallback** | ✅ Yes | ₹0 | Scikit-learn `TfidfVectorizer` |
| **External Generative Inpainting** | ❌ Requires Cloud | Optional ($/credits) | OpenAI / Stability / Replicate (Disabled by default) |
| **External Cloud Vision API** | ❌ Requires Cloud | Optional ($/credits) | OpenAI GPT-4o-mini / Gemini (Disabled by default) |
