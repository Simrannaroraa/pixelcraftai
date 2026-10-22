"""
AI-Powered Image Editing Platform — Streamlit Application.
Full-featured, portfolio-ready SaaS application for image editing,
version history, natural-language search, and local AI transformations.
Runs 100% offline at ₹0 cost without paid external APIs.
"""

import os
import sys
import time
import json
import logging
from io import BytesIO
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image

# Ensure local packages are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from services import database, storage, editing, versioning, captioning, embeddings, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Page Configuration
st.set_page_config(
    page_title="PixelCraft AI — Intelligent Image Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Storage & Database
storage.ensure_storage_dirs()
database.init_db()

# Custom CSS for Modern, Pleasing & Professional SaaS Aesthetics
CUSTOM_CSS = """
<style>
/* Modern Typography & Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

/* Deep Obsidian & Mesh Glow Atmosphere */
.stApp {
    background-color: #070a12;
    background-image:
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.14) 0px, transparent 45%),
        radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 40%),
        radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.08) 0px, transparent 50%),
        radial-gradient(at 85% 60%, rgba(99, 102, 241, 0.06) 0px, transparent 35%);
    background-attachment: fixed;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #090d16;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Top Studio Navigation / Hero Header */
.studio-header {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.studio-brand-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.studio-logo-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.45);
}

.studio-header-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.studio-version-tag {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(168, 85, 247, 0.3) 100%);
    border: 1px solid rgba(99, 102, 241, 0.5);
    color: #c7d2fe;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    -webkit-text-fill-color: #c7d2fe;
}

.studio-header-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}

.studio-telemetry-container {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}

.studio-telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #cbd5e1;
    letter-spacing: 0.02em;
}

/* Pulsing Status Dot */
@keyframes pulse-emerald {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.pulse-dot-emerald {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #34d399;
    animation: pulse-emerald 2.2s infinite;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 0.65rem 0.95rem;
    margin-bottom: 0.45rem;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateX(4px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.28) 0%, rgba(168, 85, 247, 0.18) 100%) !important;
    border-color: rgba(99, 102, 241, 0.7) !important;
    box-shadow: 0 4px 18px -2px rgba(99, 102, 241, 0.35) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label span {
    font-weight: 500;
    color: #f1f5f9;
}

/* Native Streamlit Bordered Container Custom Styling */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.68) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 0.6rem !important;
    box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    margin-bottom: 0.75rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 12px 35px -4px rgba(0, 0, 0, 0.45), 0 0 15px rgba(99, 102, 241, 0.1) !important;
}

/* Stat Cards */
.stat-card {
    position: relative;
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.stat-card.stat-indigo::before { background: linear-gradient(90deg, #6366f1, #818cf8); }
.stat-card.stat-emerald::before { background: linear-gradient(90deg, #10b981, #34d399); }
.stat-card.stat-cyan::before { background: linear-gradient(90deg, #06b6d4, #38bdf8); }
.stat-card.stat-purple::before { background: linear-gradient(90deg, #a855f7, #c084fc); }

.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255, 255, 255, 0.18);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
}

.stat-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.stat-lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 0.3rem;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-semantic {
    background: rgba(99, 102, 241, 0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.badge-local-ai {
    background: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-tfidf {
    background: rgba(245, 158, 11, 0.18);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-version {
    background: rgba(168, 85, 247, 0.18);
    color: #d8b4fe;
    border: 1px solid rgba(168, 85, 247, 0.4);
}

/* Tag Chips */
.tag-chip {
    display: inline-block;
    background: rgba(30, 41, 59, 0.7);
    color: #94a3b8;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: all 0.2s ease;
}

.tag-chip:hover {
    color: #e2e8f0;
    border-color: rgba(99, 102, 241, 0.4);
    background: rgba(99, 102, 241, 0.15);
}

/* Studio Image Canvas (Checkerboard for transparency) */
div[data-testid="stImage"] {
    background-color: #090d16 !important;
    background-image: 
        linear-gradient(45deg, #111827 25%, transparent 25%),
        linear-gradient(-45deg, #111827 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #111827 75%),
        linear-gradient(-45deg, transparent 75%, #111827 75%) !important;
    background-size: 20px 20px !important;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

div[data-testid="stImage"] img {
    border-radius: 8px !important;
}

/* Button Refinement */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.7) !important;
    color: #f1f5f9 !important;
}

div.stButton > button:hover {
    background: rgba(51, 65, 85, 0.85) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-1.5px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
}

button[kind="primary"], .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px 0 rgba(99, 102, 241, 0.4) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6) !important;
}

/* File Uploader Container */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 14px !important;
    background: rgba(15, 23, 42, 0.5) !important;
    padding: 0.8rem !important;
    transition: border-color 0.25s ease, background 0.25s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.75) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Text Input Enhancement */
div[data-baseweb="input"] {
    border-radius: 10px !important;
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}

/* Quick Action Category Card */
.preset-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
    margin-top: 0.8rem;
    margin-bottom: 0.4rem;
}

/* Monospace & Code */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.83rem !important;
    color: #c7d2fe !important;
    background: rgba(30, 41, 59, 0.8) !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== Top Studio Header Bar ====================

def render_top_header():
    engine_info = embeddings.get_active_embedding_engine_info()
    st.markdown(
        f"""
        <div class="studio-header">
            <div class="studio-brand-left">
                <div class="studio-logo-icon">✨</div>
                <div>
                    <div class="studio-header-title">
                        PixelCraft AI <span class="studio-version-tag">PRO STUDIO</span>
                    </div>
                    <div class="studio-header-subtitle">
                        Autonomous Computer Vision Studio • Neural Matting, Color Science & Semantic Retrieval
                    </div>
                </div>
            </div>
            <div class="studio-telemetry-container">
                <span class="studio-telemetry-badge">
                    <span class="pulse-dot-emerald"></span>
                    {"U2-Net Neural Matting" if editing.HAS_REMBG else "Color-Key Matting"}
                </span>
                <span class="studio-telemetry-badge">
                    ⚡ {"MiniLM 384-d Dense" if engine_info["mode"] == "semantic_embedding" else "TF-IDF Fallback"}
                </span>
                <span class="studio-telemetry-badge">
                    🛡️ ₹0 Zero API Credits
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_top_header()


# ==================== Session State Management ====================

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Library"
if "selected_image_id" not in st.session_state:
    st.session_state.selected_image_id = None
if "selected_version_id" not in st.session_state:
    st.session_state.selected_version_id = None
if "editor_instruction" not in st.session_state:
    st.session_state.editor_instruction = ""
if "last_action_message" not in st.session_state:
    st.session_state.last_action_message = None


# ==================== Helper: Load Sample Demo Image ====================

def load_sample_asset(sample_filename: str, display_name: str) -> bool:
    """Load a pre-packaged sample demo photo into the user's library."""
    sample_path = os.path.join(CURRENT_DIR, "data", "samples", sample_filename)
    if not os.path.exists(sample_path):
        st.error(f"Sample asset '{sample_filename}' not found.")
        return False
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    is_valid, err, meta = storage.validate_image_file(file_bytes, sample_filename)
    if not is_valid:
        st.error(f"Sample validation failed: {err}")
        return False

    with st.spinner(f"Ingesting sample '{display_name}' and computing neural embeddings..."):
        img_id, fpath, meta = storage.save_uploaded_image(file_bytes, display_name)
        pil_img = storage.load_image(fpath)
        analysis = captioning.analyze_image_locally(pil_img)

        database.insert_image(
            image_id=img_id,
            filename=os.path.basename(fpath),
            original_name=display_name,
            file_path=fpath,
            file_size=meta["file_size"],
            mime_type=meta["mime_type"],
            width=meta["width"],
            height=meta["height"],
            caption=analysis["caption"],
            description=analysis["description"],
            tags=analysis["tags"]
        )
        search.sync_image_index(img_id)
        st.session_state.selected_image_id = img_id
        st.session_state.selected_version_id = None
        return True


# ==================== Sidebar Navigation & Diagnostics ====================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.8rem;">
            <div style="width:36px; height:36px; background:linear-gradient(135deg, #6366f1, #a855f7); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; box-shadow:0 0 15px rgba(99,102,241,0.5);">✨</div>
            <div>
                <div style="font-size:1.25rem; font-weight:800; letter-spacing:-0.02em; color:#ffffff;">PixelCraft AI</div>
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:500;">Zero-Credit AI Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_options = [
        "🖼️ Image Library",
        "🎨 Image Editor",
        "🔍 Smart Search",
        "⚙️ System & Diagnostics"
    ]
    nav_map = {
        "🖼️ Image Library": "Library",
        "🎨 Image Editor": "Editor",
        "🔍 Smart Search": "Search",
        "⚙️ System & Diagnostics": "Settings"
    }
    inv_nav_map = {v: k for k, v in nav_map.items()}

    current_selection = inv_nav_map.get(st.session_state.active_nav, nav_options[0])
    selected_page_label = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(current_selection),
        label_visibility="collapsed"
    )
    st.session_state.active_nav = nav_map[selected_page_label]

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Engine Telemetry Card
    engine_info = embeddings.get_active_embedding_engine_info()
    with st.container(border=True):
        st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#818cf8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;'>⚡ Studio Telemetry</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-bottom:0.45rem;">
                <span class="badge {'badge-semantic' if engine_info['mode'] == 'semantic_embedding' else 'badge-tfidf'}">
                    {'🟢 Dense MiniLM (384d)' if engine_info['mode'] == 'semantic_embedding' else '🟠 TF-IDF Keyword'}
                </span>
            </div>
            <div>
                <span class="badge {'badge-local-ai' if editing.HAS_REMBG else 'badge-tfidf'}">
                    {'🟢 U2-Net Deep Neural' if editing.HAS_REMBG else '🟡 Color-Key Matting'}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Library Quick Stats
    total_imgs = database.count_images()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0.2rem; font-size:0.82rem; color:#94a3b8;">
            <span>Catalog Images:</span>
            <strong style="color:#ffffff; font-size:0.95rem;">{total_imgs}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.caption("🛡️ **Zero Paid Credits Guarantee**: All core vision and retrieval features execute 100% locally with zero cloud API dependency.")


# ==================== Page 1: Image Library ====================

def render_library():
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <h2 style="font-weight:800; font-size:1.6rem; letter-spacing:-0.02em; margin-bottom:0.2rem; color:#ffffff;">
                    🖼️ Image Library & Catalog
                </h2>
                <div style="font-size:0.88rem; color:#94a3b8;">
                    Upload, profile, and inspect your media catalog with instant local visual analysis.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_images = database.list_images(limit=100)

    # Upload & Sample Hero Section
    with st.expander("📤 Upload New Image or Load Instant Demo Assets", expanded=(len(all_images) == 0)):
        col_up1, col_up2 = st.columns([1.6, 1.0])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Drag and drop or browse from disk",
                type=["jpg", "jpeg", "png", "webp"],
                help="Max size 15 MB. Validated via PIL header checks."
            )
            custom_caption = st.text_input(
                "Optional Custom Caption / Notes",
                placeholder="Leave blank for automatic heuristic scene profiling"
            )
            if uploaded_file is not None:
                if st.button("🚀 Ingest & Profile Image", use_container_width=True, type="primary"):
                    file_bytes = uploaded_file.read()
                    is_valid, err, meta = storage.validate_image_file(file_bytes, uploaded_file.name)
                    if not is_valid:
                        st.error(f"❌ Upload Failed: {err}")
                    else:
                        with st.spinner("Processing image and building search embeddings..."):
                            img_id, fpath, meta = storage.save_uploaded_image(file_bytes, uploaded_file.name)
                            pil_img = storage.load_image(fpath)

                            if custom_caption.strip():
                                caption_text = custom_caption.strip()
                                analysis = captioning.analyze_image_locally(pil_img)
                                desc_text = analysis["description"]
                                tags_text = analysis["tags"]
                            else:
                                analysis = captioning.analyze_image_locally(pil_img)
                                caption_text = analysis["caption"]
                                desc_text = analysis["description"]
                                tags_text = analysis["tags"]

                            database.insert_image(
                                image_id=img_id,
                                filename=os.path.basename(fpath),
                                original_name=uploaded_file.name,
                                file_path=fpath,
                                file_size=meta["file_size"],
                                mime_type=meta["mime_type"],
                                width=meta["width"],
                                height=meta["height"],
                                caption=caption_text,
                                description=desc_text,
                                tags=tags_text
                            )
                            search.sync_image_index(img_id)
                            st.session_state.selected_image_id = img_id
                            st.session_state.selected_version_id = None
                            st.success(f"✅ '{uploaded_file.name}' added to library!")
                            time.sleep(0.4)
                            st.rerun()

        with col_up2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="font-weight:700; font-size:0.85rem; color:#c7d2fe; margin-bottom:0.4rem;">
                        ✨ Instant Demo Presets
                    </div>
                    <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:0.8rem;">
                        Test background removal, procedural compositing, and semantic search with 1 click:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                c_s1, c_s2 = st.columns(2)
                with c_s1:
                    if st.button("📸 Studio Portrait", use_container_width=True, help="Load stylish studio portrait"):
                        if load_sample_asset("sample_portrait.jpg", "Demo Studio Portrait.jpg"):
                            st.success("Loaded portrait sample!")
                            time.sleep(0.4)
                            st.rerun()
                with c_s2:
                    if st.button("🌅 Coastal Sunset", use_container_width=True, help="Load vibrant landscape sunset"):
                        if load_sample_asset("sample_sunset.jpg", "Demo Coastal Sunset.jpg"):
                            st.success("Loaded sunset sample!")
                            time.sleep(0.4)
                            st.rerun()

    if not all_images:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding:2.5rem 1rem;">
                    <div style="font-size:3rem; margin-bottom:0.8rem;">🎨</div>
                    <div style="font-size:1.25rem; font-weight:700; color:#ffffff; margin-bottom:0.4rem;">Your Studio Catalog is Empty</div>
                    <div style="font-size:0.88rem; color:#94a3b8; max-width:480px; margin:0 auto 1.5rem auto;">
                        Upload an image or click one of the <b>Instant Demo Presets</b> above to immediately test neural segmentation and semantic search!
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        return

    # Executive Stat Tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="stat-card stat-indigo">
                <div class="stat-val">{len(all_images)}</div>
                <div class="stat-lbl">Catalog Images</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        all_embs = database.get_all_embeddings()
        st.markdown(
            f"""
            <div class="stat-card stat-emerald">
                <div class="stat-val">{len(all_embs)}</div>
                <div class="stat-lbl">Indexed Embeddings</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        total_kb = sum(img["file_size"] for img in all_images) / 1024
        size_str = f"{total_kb:.1f} KB" if total_kb < 1024 else f"{total_kb/1024:.2f} MB"
        st.markdown(
            f"""
            <div class="stat-card stat-cyan">
                <div class="stat-val">{size_str}</div>
                <div class="stat-lbl">Disk Footprint</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        engine_mode = "MiniLM 384d" if embeddings.get_active_embedding_engine_info()["mode"] == "semantic_embedding" else "TF-IDF"
        st.markdown(
            f"""
            <div class="stat-card stat-purple">
                <div class="stat-val">{engine_mode}</div>
                <div class="stat-lbl">Search Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Image Grid Cards
    cols_per_row = 3
    for i in range(0, len(all_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(all_images):
                img_data = all_images[idx]
                with cols[j]:
                    with st.container(border=True):
                        latest_v = database.get_latest_version(img_data["id"])
                        display_path = latest_v["file_path"] if latest_v else img_data["file_path"]

                        # Image preview
                        if os.path.exists(display_path):
                            st.image(display_path, use_container_width=True)
                        else:
                            st.warning("⚠️ File missing on disk")

                        # Info Header
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem; margin-bottom:0.2rem;">
                                <strong style="color:#ffffff; font-size:0.95rem; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="{img_data['original_name']}">
                                    {img_data['original_name'][:24]}{'...' if len(img_data['original_name']) > 24 else ''}
                                </strong>
                                {'<span class="badge badge-version">v' + str(latest_v["version_number"]) + ' Edit</span>' if latest_v else '<span class="badge" style="background:rgba(255,255,255,0.06); color:#94a3b8;">Original</span>'}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        caption_preview = img_data.get("caption") or "No caption provided"
                        st.caption(f"📝 {caption_preview[:85]}...")

                        # Tags
                        if img_data.get("tags"):
                            tag_html = "".join([f'<span class="tag-chip">#{t.strip()}</span>' for t in img_data["tags"].split(",")[:4]])
                            st.markdown(tag_html, unsafe_allow_html=True)

                        st.markdown(
                            f"<div style='font-size:0.73rem; color:#64748b; font-family:JetBrains Mono, monospace; margin-bottom:0.7rem;'>"
                            f"{img_data['width']}×{img_data['height']} px • {img_data['file_size']/1024:.1f} KB • {img_data['mime_type'].split('/')[-1].upper()}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # Action Toolbar
                        btn_c1, btn_c2, btn_c3 = st.columns([1.5, 1.2, 0.9])
                        with btn_c1:
                            if st.button("🎨 Open Studio", key=f"edit_btn_{img_data['id']}", use_container_width=True, type="primary"):
                                st.session_state.selected_image_id = img_data["id"]
                                st.session_state.selected_version_id = latest_v["id"] if latest_v else None
                                st.session_state.active_nav = "Editor"
                                st.rerun()
                        with btn_c2:
                            if os.path.exists(display_path):
                                with open(display_path, "rb") as f:
                                    st.download_button(
                                        label="💾 Save",
                                        data=f.read(),
                                        file_name=f"pixelcraft_{img_data['original_name']}",
                                        mime="image/png",
                                        key=f"dl_btn_{img_data['id']}",
                                        use_container_width=True
                                    )
                        with btn_c3:
                            if st.button("🗑️", key=f"del_btn_{img_data['id']}", use_container_width=True, help="Delete image"):
                                success, paths = database.delete_image(img_data["id"])
                                for p in paths:
                                    storage.delete_file_safely(p)
                                search.remove_image_from_index(img_data["id"])
                                if st.session_state.selected_image_id == img_data["id"]:
                                    st.session_state.selected_image_id = None
                                st.success("Deleted image.")
                                time.sleep(0.3)
                                st.rerun()


# ==================== Page 2: Image Editor & Version History ====================

def render_editor():
    all_images = database.list_images(limit=100)
    if not all_images:
        st.info("ℹ️ No images found. Please upload an image in the Library first!")
        if st.button("🚀 Go to Library", type="primary"):
            st.session_state.active_nav = "Library"
            st.rerun()
        return

    # Image selector dropdown
    img_options = {img["id"]: f"{img['original_name']} ({img['width']}×{img['height']} px)" for img in all_images}
    default_idx = 0
    if st.session_state.selected_image_id in img_options:
        default_idx = list(img_options.keys()).index(st.session_state.selected_image_id)

    # Workspace Header Bar
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <h2 style="font-weight:800; font-size:1.6rem; letter-spacing:-0.02em; margin-bottom:0.2rem; color:#ffffff;">
                    🎨 AI Image Studio & Version History
                </h2>
                <div style="font-size:0.88rem; color:#94a3b8;">
                    Non-destructive editing with neural segmentation, color science, and instant rollback.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    top_sel_c1, top_sel_c2 = st.columns([2.5, 1.0])
    with top_sel_c1:
        selected_id = st.selectbox(
            "Active Image in Studio:",
            options=list(img_options.keys()),
            index=default_idx,
            format_func=lambda x: img_options[x],
            label_visibility="collapsed"
        )
        st.session_state.selected_image_id = selected_id
    with top_sel_c2:
        if st.button("🔄 Switch / Refresh Active", use_container_width=True):
            st.rerun()

    # Load active version / state
    try:
        active_pil, active_info = versioning.get_active_image_state(
            selected_id, version_id=st.session_state.selected_version_id
        )
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    if active_pil is None:
        st.error("Failed to load image from disk.")
        return

    # Two-column layout: Left = Viewport & Comparisons, Right = Editing Controls & Versions
    view_col, ctrl_col = st.columns([1.25, 1.0])

    with view_col:
        with st.container(border=True):
            # Header with version badge & view mode
            head_c1, head_c2 = st.columns([1.6, 1.4])
            with head_c1:
                ver_num = active_info.get("version_number", 0)
                if ver_num == 0:
                    st.markdown("<span style='font-size:0.9rem; font-weight:700; color:#e2e8f0;'>Active View:</span> <span class='badge' style='background:rgba(255,255,255,0.08); color:#94a3b8;'>Original Upload</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='font-size:0.9rem; font-weight:700; color:#e2e8f0;'>Active View:</span> <span class='badge badge-version'>Version {ver_num} Output</span>", unsafe_allow_html=True)
            with head_c2:
                comparison_mode = st.radio(
                    "Comparison Mode",
                    ["Before & After", "Active Only"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            if comparison_mode == "Before & After":
                before_img, after_img, comp_meta = versioning.get_comparison_pair(
                    selected_id, version_id=st.session_state.selected_version_id
                )
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94a3b8; margin-bottom:4px; text-transform:uppercase;'>⏮️ Before (Parent / Orig)</div>", unsafe_allow_html=True)
                    if before_img:
                        st.image(before_img, use_container_width=True)
                with col_a:
                    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#a855f7; margin-bottom:4px; text-transform:uppercase;'>⏭️ After (v{comp_meta.get('version_number', 0)})</div>", unsafe_allow_html=True)
                    if after_img:
                        st.image(after_img, use_container_width=True)
            else:
                st.image(active_pil, use_container_width=True)

            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#94a3b8; font-family:JetBrains Mono, monospace; margin-top:0.75rem; margin-bottom:0.5rem;">
                    <span>Canvas: <b>{active_pil.width} × {active_pil.height} px</b></span>
                    <span>Method: <code>{active_info.get('method', 'original')}</code></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Download active button
            dl_buf = BytesIO()
            active_pil.save(dl_buf, format="PNG")
            st.download_button(
                label="💾 Download Current Output (PNG)",
                data=dl_buf.getvalue(),
                file_name=f"pixelcraft_{selected_id[:8]}_v{active_info.get('version_number', 0)}.png",
                mime="image/png",
                use_container_width=True
            )

    with ctrl_col:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-weight:700; font-size:1.05rem; color:#ffffff; margin-bottom:0.2rem;">
                    ✨ Natural-Language AI Command
                </div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.8rem;">
                    Type conversational instructions or select a 1-click studio preset below:
                </div>
                """,
                unsafe_allow_html=True
            )

            prompt_input = st.text_input(
                "Editing Instruction",
                value=st.session_state.editor_instruction,
                placeholder="e.g. 'Remove the background' or 'Change background to snowy mountain'",
                key="nl_edit_input",
                label_visibility="collapsed"
            )

            col_run, col_clear = st.columns([2, 1])
            with col_run:
                run_btn = st.button("🪄 Execute AI Edit", type="primary", use_container_width=True)
            with col_clear:
                if st.button("Clear", use_container_width=True):
                    st.session_state.editor_instruction = ""
                    st.rerun()

            # Categorized Presets
            st.markdown('<div class="preset-section-title">🔮 Neural Segmentation & Backdrop</div>', unsafe_allow_html=True)
            p_row1_c1, p_row1_c2 = st.columns(2)
            with p_row1_c1:
                if st.button("✂️ Cutout Subject", use_container_width=True, help="U2-Net AI background isolation"):
                    st.session_state.editor_instruction = "Remove the background"
                    run_btn = True
            with p_row1_c2:
                if st.button("🏔️ Alpine Snow", use_container_width=True, help="Composite onto snowy mountain"):
                    st.session_state.editor_instruction = "Change the background to a snowy mountain"
                    run_btn = True

            p_row2_c1, p_row2_c2 = st.columns(2)
            with p_row2_c1:
                if st.button("🌅 Golden Sunset", use_container_width=True, help="Composite onto sunset gradient"):
                    st.session_state.editor_instruction = "Replace background with sunset"
                    run_btn = True
            with p_row2_c2:
                if st.button("🏢 Dark Cyber Studio", use_container_width=True, help="Dark vignette studio background"):
                    st.session_state.editor_instruction = "Change background to dark studio"
                    run_btn = True

            st.markdown('<div class="preset-section-title">🎨 Color Artistry & Cinematic Styles</div>', unsafe_allow_html=True)
            p_row3_c1, p_row3_c2 = st.columns(2)
            with p_row3_c1:
                if st.button("☀️ Warm Daylight", use_container_width=True):
                    st.session_state.editor_instruction = "Make the image brighter and warmer"
                    run_btn = True
            with p_row3_c2:
                if st.button("🎬 Hollywood Teal & Orange", use_container_width=True):
                    st.session_state.editor_instruction = "Apply cinematic teal and orange filter"
                    run_btn = True

            p_row4_c1, p_row4_c2 = st.columns(2)
            with p_row4_c1:
                if st.button("✏️ Fine Pencil Sketch", use_container_width=True):
                    st.session_state.editor_instruction = "Convert to pencil sketch style"
                    run_btn = True
            with p_row4_c2:
                if st.button("🔲 Film Noir High-Contrast", use_container_width=True):
                    st.session_state.editor_instruction = "Make it black and white noir"
                    run_btn = True

            st.markdown('<div class="preset-section-title">📐 Smart Aspect Ratio Crops</div>', unsafe_allow_html=True)
            p_row5_c1, p_row5_c2 = st.columns(2)
            with p_row5_c1:
                if st.button("📐 1:1 Square (Avatar)", use_container_width=True):
                    st.session_state.editor_instruction = "Crop to square aspect ratio 1:1"
                    run_btn = True
            with p_row5_c2:
                if st.button("📐 16:9 Cinema Wide", use_container_width=True):
                    st.session_state.editor_instruction = "Crop to aspect ratio 16:9"
                    run_btn = True

            # Manual Fine-Tuning Sliders Accordion
            with st.expander("🎛️ Manual Parameters & Color Grading"):
                slider_b = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
                slider_c = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
                slider_s = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
                slider_sat = st.slider("Saturation", 0.0, 2.5, 1.0, 0.1)
                slider_warm = st.slider("Warmth Balance (Cool ↔ Warm)", -100, 100, 0, 5)
                slider_sepia = st.checkbox("Sepia Vintage Effect", value=False)

                if st.button("⚡ Apply Fine-Tuning as New Version", use_container_width=True):
                    with st.spinner("Applying manual parameters..."):
                        tuned = editing.adjust_lighting(active_pil, brightness=slider_b, contrast=slider_c, sharpness=slider_s)
                        tuned = editing.adjust_color(tuned, saturation=slider_sat, warmth=slider_warm, sepia=slider_sepia)

                        next_num = database.get_next_version_number(selected_id)
                        ver_id, fpath, meta = storage.save_version_image(tuned, selected_id, next_num)
                        ver_record = database.insert_version(
                            version_id=ver_id,
                            image_id=selected_id,
                            parent_version_id=active_info.get("version_id"),
                            version_number=next_num,
                            file_path=fpath,
                            edit_prompt=f"Manual Tuning (B:{slider_b}, C:{slider_c}, S:{slider_s}, Sat:{slider_sat}, Warm:{slider_warm})",
                            method="Manual Deterministic Transformation (Pillow/NumPy)",
                            parameters={"brightness": slider_b, "contrast": slider_c, "warmth": slider_warm},
                            width=meta["width"],
                            height=meta["height"],
                            file_size=meta["file_size"]
                        )
                        search.sync_image_index(selected_id)
                        st.session_state.selected_version_id = ver_id
                        st.success("Tuning applied as new version!")
                        time.sleep(0.4)
                        st.rerun()

            # Handle Execution
            if run_btn:
                instruction = st.session_state.editor_instruction.strip()
                if not instruction:
                    st.warning("Please enter or select an editing instruction.")
                else:
                    with st.spinner(f"Executing: '{instruction}'..."):
                        success, msg, ver_record = versioning.create_new_version(
                            image_id=selected_id,
                            instruction=instruction,
                            parent_version_id=active_info.get("version_id")
                        )
                        if success and ver_record:
                            search.sync_image_index(selected_id)
                            st.session_state.selected_version_id = ver_record["id"]
                            st.session_state.last_action_message = f"✅ Applied '{instruction}' as Version {ver_record['version_number']}."
                            st.rerun()
                        else:
                            st.error(f"❌ Edit Could Not Be Applied: {msg}")

    # ==================== Version History Timeline ====================
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom:0.8rem;">
            <div style="font-size:1.15rem; font-weight:700; color:#ffffff;">
                📜 Non-Destructive Version History & Filmstrip Timeline
            </div>
            <div style="font-size:0.82rem; color:#94a3b8;">
                The original upload remains strictly immutable on disk. Every edit forks a new child version with instant rollback.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    history = versioning.get_version_history(selected_id)
    orig_row = database.get_image(selected_id)

    # Timeline cards
    v_cols = st.columns(max(len(history) + 1, 1))

    # Version 0: Original
    with v_cols[0]:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; font-size:0.8rem; font-weight:700; color:#38bdf8; margin-bottom:4px;'>⭐ ORIGINAL UPLOAD</div>", unsafe_allow_html=True)
            if orig_row and os.path.exists(orig_row["file_path"]):
                st.image(orig_row["file_path"], use_container_width=True)
            st.caption(f"{orig_row['created_at'][:16] if orig_row else ''}")
            if st.button("⏮️ Revert / View", key="revert_orig", use_container_width=True):
                st.session_state.selected_version_id = None
                st.rerun()

    # Subsequent versions
    for idx, v in enumerate(history):
        with v_cols[idx + 1]:
            with st.container(border=True):
                st.markdown(f"<div style='text-align:center; font-size:0.8rem; font-weight:700; color:#c084fc; margin-bottom:4px;'>VERSION {v['version_number']}</div>", unsafe_allow_html=True)
                if os.path.exists(v["file_path"]):
                    st.image(v["file_path"], use_container_width=True)
                st.caption(f"_{v['edit_prompt'][:25]}..._")
                st.caption(f"<small>{v['created_at'][:16]}</small>", unsafe_allow_html=True)

                btn_col1, btn_col2 = st.columns([2, 1])
                with btn_col1:
                    if st.button("👁️ Load", key=f"load_v_{v['id']}", use_container_width=True):
                        st.session_state.selected_version_id = v["id"]
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️", key=f"del_v_{v['id']}", help="Delete this version"):
                        success, fpath = database.delete_version(v["id"])
                        storage.delete_file_safely(fpath)
                        search.sync_image_index(selected_id)
                        if st.session_state.selected_version_id == v["id"]:
                            st.session_state.selected_version_id = None
                        st.rerun()


# ==================== Page 3: Natural Language Search ====================

def render_search():
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <h2 style="font-weight:800; font-size:1.6rem; letter-spacing:-0.02em; margin-bottom:0.2rem; color:#ffffff;">
                    🔍 Natural-Language Semantic Search
                </h2>
                <div style="font-size:0.88rem; color:#94a3b8;">
                    Search your media library by conceptual meaning, lighting, colors, and visual tags using 384-dimensional dense vectors.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    engine_info = embeddings.get_active_embedding_engine_info()

    # Search Bar & Threshold
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        query = st.text_input(
            "Natural-Language Search Query",
            placeholder="e.g. 'golden sunset', 'snowy mountains', 'high-contrast portrait', 'vibrant colors'",
            key="search_query_input",
            label_visibility="collapsed"
        )
    with s_col2:
        min_thresh = st.slider("Min Relevance %", min_value=0, max_value=80, value=10, step=5)

    # Suggestion Chips
    st.markdown("<small style='color:#94a3b8; font-weight:600;'>Sample Search Prompts: </small>", unsafe_allow_html=True)
    chip_cols = st.columns(5)
    samples = ["golden sunset", "snowy mountain", "high-contrast", "studio portrait", "vibrant landscape"]
    for i, s in enumerate(samples):
        with chip_cols[i]:
            if st.button(f"🔍 {s}", key=f"sample_q_{i}", use_container_width=True):
                query = s
                st.session_state.search_query_input = s

    st.markdown("<br>", unsafe_allow_html=True)

    if not query.strip():
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding:2rem 1rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.8rem;">💡</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin-bottom:0.4rem;">Type a Search Concept Above</div>
                    <div style="font-size:0.85rem; color:#94a3b8; max-width:440px; margin:0 auto;">
                        Try searching for moods, objects, lighting styles, or colors. Dense semantic embeddings rank results by cosine similarity.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        return

    # Execute Search
    with st.spinner(f"Evaluating dense vector projections for '{query}'..."):
        search_res = search.search_images(query, min_score=(min_thresh / 100.0))

    results = search_res.get("results", [])
    engine_used = search_res.get("engine_used", "Search Engine")

    # Engine badge
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
            <span class="badge badge-semantic">⚡ {engine_used}</span>
            <span style="font-size:0.85rem; color:#94a3b8;"><b>{len(results)}</b> matches identified</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not results:
        st.warning(f"No catalog images matched '{query}' above {min_thresh}% similarity threshold. Try lowering the threshold slider or using alternate keywords.")
        return

    # Results Grid
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(results):
                item = results[idx]
                img_data = item["image"]
                latest_v = item["latest_version"]
                display_path = item["display_path"]
                score_pct = item["score_percent"]

                with cols[j]:
                    with st.container(border=True):
                        if os.path.exists(display_path):
                            st.image(display_path, use_container_width=True)

                        # Relevance Score Bar
                        badge_style = "badge-local-ai" if score_pct >= 70 else ("badge-semantic" if score_pct >= 40 else "badge-tfidf")
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem; margin-bottom:0.3rem;">
                                <strong style="color:#ffffff; font-size:0.95rem;">{img_data['original_name'][:22]}</strong>
                                <span class="badge {badge_style}">{score_pct}% Match</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.progress(score_pct / 100.0)

                        st.caption(f"📝 {img_data.get('caption', '')[:100]}")

                        if img_data.get("tags"):
                            tag_html = "".join([f'<span class="tag-chip">#{t.strip()}</span>' for t in img_data["tags"].split(",")[:4]])
                            st.markdown(tag_html, unsafe_allow_html=True)

                        if latest_v:
                            st.caption(f"Latest Edit: *{latest_v.get('edit_prompt', '')}*")

                        if st.button("🎨 Open in Studio", key=f"open_search_{img_data['id']}", use_container_width=True, type="primary"):
                            st.session_state.selected_image_id = img_data["id"]
                            st.session_state.selected_version_id = latest_v["id"] if latest_v else None
                            st.session_state.active_nav = "Editor"
                            st.rerun()


# ==================== Page 4: System & Diagnostics ====================

def render_settings():
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <h2 style="font-weight:800; font-size:1.6rem; letter-spacing:-0.02em; margin-bottom:0.2rem; color:#ffffff;">
                    ⚙️ System Diagnostics & Index Engine
                </h2>
                <div style="font-size:0.88rem; color:#94a3b8;">
                    Inspect local neural pipelines, disk footprints, SQLite integrity, and trigger full index rebuilds.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("<div style='font-size:1.05rem; font-weight:700; color:#818cf8; margin-bottom:0.8rem;'>🧠 AI Model Pipelines</div>", unsafe_allow_html=True)
            engine_info = embeddings.get_active_embedding_engine_info()
            
            st.markdown(
                f"""
                <table style="width:100%; font-size:0.85rem; border-collapse:collapse; margin-bottom:1rem;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06); padding:0.4rem 0;">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Retrieval Mode:</td>
                        <td style="text-align:right;"><b>{engine_info['mode'].replace('_', ' ').title()}</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Model Checkpoint:</td>
                        <td style="text-align:right;"><code>{engine_info.get('model_name', 'N/A').split('/')[-1]}</code></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Vector Dimensionality:</td>
                        <td style="text-align:right;"><b>{engine_info.get('dimension', 'N/A')} Dense Dimensions</b></td>
                    </tr>
                    <tr>
                        <td style="color:#94a3b8; padding:0.4rem 0;">Cost Per Query:</td>
                        <td style="text-align:right; color:#34d399;"><b>₹0.00 (Zero Credits)</b></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div style='font-size:1.05rem; font-weight:700; color:#34d399; margin-bottom:0.8rem;'>✂️ Neural Segmentation Engine</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Framework:</td>
                        <td style="text-align:right;"><b>{'rembg (U2-Net Deep Neural Network)' if editing.HAS_REMBG else 'Pillow Color-Key Luminosity'}</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Execution Backend:</td>
                        <td style="text-align:right;"><code>{'onnxruntime CPU Multi-Threaded' if editing.HAS_REMBG else 'Pillow CPU'}</code></td>
                    </tr>
                    <tr>
                        <td style="color:#94a3b8; padding:0.4rem 0;">Cost Per Segmentation:</td>
                        <td style="text-align:right; color:#34d399;"><b>₹0.00 (100% Offline)</b></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

    with c2:
        with st.container(border=True):
            st.markdown("<div style='font-size:1.05rem; font-weight:700; color:#38bdf8; margin-bottom:0.8rem;'>🗄️ Database & Storage Footprint</div>", unsafe_allow_html=True)
            db_path = database.DEFAULT_DB_PATH
            img_count = database.count_images()
            all_embs = database.get_all_embeddings()

            def get_dir_size_mb(path):
                total = 0
                if os.path.exists(path):
                    for dirpath, dirnames, filenames in os.walk(path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if os.path.exists(fp):
                                total += os.path.getsize(fp)
                return round(total / (1024 * 1024), 2)

            up_mb = get_dir_size_mb(storage.UPLOADS_DIR)
            ver_mb = get_dir_size_mb(storage.VERSIONS_DIR)

            st.markdown(
                f"""
                <table style="width:100%; font-size:0.85rem; border-collapse:collapse; margin-bottom:1.2rem;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">SQLite Storage DB:</td>
                        <td style="text-align:right;"><code>{os.path.basename(db_path)}</code></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Total Catalog Records:</td>
                        <td style="text-align:right;"><b>{img_count} images</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Persisted Embeddings:</td>
                        <td style="text-align:right;"><b>{len(all_embs)} vector BLOBs</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="color:#94a3b8; padding:0.4rem 0;">Uploads Directory:</td>
                        <td style="text-align:right;"><b>{up_mb} MB</b></td>
                    </tr>
                    <tr>
                        <td style="color:#94a3b8; padding:0.4rem 0;">Versions Directory:</td>
                        <td style="text-align:right;"><b>{ver_mb} MB</b></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;'>🔄 Search Index Rebuilder</div>", unsafe_allow_html=True)
            st.caption("Re-compute all dense embeddings across the catalog.")
            if st.button("Rebuild Entire Search Index Now", use_container_width=True, type="primary"):
                with st.spinner("Recomputing vector embeddings..."):
                    res = search.rebuild_full_index()
                    st.success(f"Indexed {res['semantic_indexed']} of {res['total_images']} images successfully!")
                    time.sleep(0.5)
                    st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🔑 External Generative API Settings (Optional)")
    st.info("External cloud APIs are 100% OPTIONAL and disabled by default. The entire studio runs locally at zero cost.")

    with st.expander("Configure Optional Cloud API Keys (Session Memory Only)"):
        st.caption("If you provide an OpenAI or Gemini API key, it will be used for optional cloud generative inpainting. If left blank, ₹0 local AI is used.")
        opt_openai = st.text_input("OpenAI API Key (Optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        opt_gemini = st.text_input("Gemini API Key (Optional)", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        if st.button("Save API Configuration"):
            if opt_openai:
                os.environ["OPENAI_API_KEY"] = opt_openai
            if opt_gemini:
                os.environ["GEMINI_API_KEY"] = opt_gemini
            st.success("API preferences updated for current session.")


# ==================== Main Router ====================

if st.session_state.active_nav == "Library":
    render_library()
elif st.session_state.active_nav == "Editor":
    render_editor()
elif st.session_state.active_nav == "Search":
    render_search()
elif st.session_state.active_nav == "Settings":
    render_settings()
