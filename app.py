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

# Custom CSS for Modern SaaS Aesthetics
CUSTOM_CSS = """
<style>
/* Modern Glassmorphic Container & Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Gradient */
.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.brand-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}

/* Glass Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-semantic {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.badge-local-ai {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-tfidf {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-version {
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

/* Tag Chips */
.tag-chip {
    display: inline-block;
    background: rgba(51, 65, 85, 0.6);
    color: #cbd5e1;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.72rem;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Quick Action Chips */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}

/* Metric Display Cards */
.metric-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}

.metric-lbl {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Code & System Logs */
code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


# ==================== Sidebar Navigation & Diagnostics ====================

with st.sidebar:
    st.markdown('<div class="brand-title">PixelCraft AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Zero-Cost AI Image Studio</div>', unsafe_allow_html=True)

    nav_options = ["🖼️ Image Library", "🎨 Image Editor", "🔍 Smart Search", "⚙️ System & Diagnostics"]
    
    # Map friendly names
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

    st.markdown("---")

    # Engine Status Indicators
    engine_info = embeddings.get_active_embedding_engine_info()
    st.markdown("#### ⚡ Active Engines")

    if engine_info["mode"] == "semantic_embedding":
        st.markdown(
            '<span class="badge badge-semantic">🟢 Dense Semantic Search</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Model: `{engine_info['model_name'].split('/')[-1]}` ({engine_info['dimension']}d)")
    else:
        st.markdown(
            '<span class="badge badge-tfidf">🟠 Keyword TF-IDF Fallback</span>',
            unsafe_allow_html=True
        )
        st.caption("Scikit-learn cosine similarity fallback")

    if editing.HAS_REMBG:
        st.markdown(
            '<span class="badge badge-local-ai">🟢 Local AI Segmentation</span>',
            unsafe_allow_html=True
        )
        st.caption("U2-Net Deep Neural Network (`rembg`)")
    else:
        st.markdown(
            '<span class="badge badge-tfidf">🟡 Threshold Matting</span>',
            unsafe_allow_html=True
        )
        st.caption("Pillow Color-Key Luminosity")

    st.markdown("---")
    # Quick Stats in Sidebar
    total_imgs = database.count_images()
    st.markdown(f"**Library Images:** {total_imgs}")

    st.markdown("---")
    st.caption("🛡️ **Zero Paid Credits Guarantee**: All primary features run 100% offline using local AI models and Pillow.")


# ==================== Page 1: Image Library ====================

def render_library():
    st.markdown("### 🖼️ Image Library & Catalog")
    st.markdown("Upload, inspect, and organize your images. All files and metadata persist across restarts.")

    # Top upload card
    with st.expander("📤 Upload New Image (JPG, PNG, WEBP)", expanded=(database.count_images() == 0)):
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "webp"],
                help="Max size 15 MB. Validated via PIL header checks."
            )
            custom_caption = st.text_input(
                "Optional Custom Caption / Description",
                placeholder="Leave blank for automatic local scene analysis"
            )
        with col_up2:
            st.markdown("<br>", unsafe_allow_html=True)
            upload_btn = st.button("🚀 Upload & Analyze Image", use_container_width=True, type="primary")

        if upload_btn and uploaded_file is not None:
            file_bytes = uploaded_file.read()
            is_valid, err, meta = storage.validate_image_file(file_bytes, uploaded_file.name)
            if not is_valid:
                st.error(f"❌ Upload Failed: {err}")
            else:
                with st.spinner("Saving image and running local visual analysis..."):
                    img_id, fpath, meta = storage.save_uploaded_image(file_bytes, uploaded_file.name)
                    pil_img = storage.load_image(fpath)

                    # Caption generation
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

                    # Save to DB
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

                    # Sync search index
                    search.sync_image_index(img_id)

                    st.success(f"✅ Image '{uploaded_file.name}' saved and indexed successfully!")
                    st.session_state.selected_image_id = img_id
                    time.sleep(0.5)
                    st.rerun()

    # Library Controls
    all_images = database.list_images(limit=100)
    if not all_images:
        st.info("ℹ️ Your image library is empty. Upload an image above to start editing and searching!")
        return

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(all_images)}</div><div class="metric-lbl">Total Images</div></div>', unsafe_allow_html=True)
    with m2:
        all_embs = database.get_all_embeddings()
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(all_embs)}</div><div class="metric-lbl">Indexed Embeddings</div></div>', unsafe_allow_html=True)
    with m3:
        total_kb = sum(img["file_size"] for img in all_images) / 1024
        st.markdown(f'<div class="metric-box"><div class="metric-val">{total_kb:.1f} KB</div><div class="metric-lbl">Library Size</div></div>', unsafe_allow_html=True)
    with m4:
        engine_mode = "MiniLM 384d" if embeddings.get_active_embedding_engine_info()["mode"] == "semantic_embedding" else "TF-IDF"
        st.markdown(f'<div class="metric-box"><div class="metric-val">{engine_mode}</div><div class="metric-lbl">Search Pipeline</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Image Grid Display
    cols_per_row = 3
    for i in range(0, len(all_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(all_images):
                img_data = all_images[idx]
                with cols[j]:
                    with st.container():
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        # Latest version thumbnail or original
                        latest_v = database.get_latest_version(img_data["id"])
                        display_path = latest_v["file_path"] if latest_v else img_data["file_path"]

                        if os.path.exists(display_path):
                            st.image(display_path, use_container_width=True)
                        else:
                            st.warning("⚠️ File missing on disk")

                        # Info
                        st.markdown(f"**{img_data['original_name']}**")
                        if latest_v:
                            st.markdown(
                                f'<span class="badge badge-version">v{latest_v["version_number"]} Edit</span>',
                                unsafe_allow_html=True
                            )

                        caption_preview = img_data.get("caption") or "No caption provided"
                        st.caption(f"📝 {caption_preview[:100]}...")

                        # Tags
                        if img_data.get("tags"):
                            tag_html = "".join([f'<span class="tag-chip">{t.strip()}</span>' for t in img_data["tags"].split(",")[:4]])
                            st.markdown(tag_html, unsafe_allow_html=True)

                        st.markdown(f"<small style='color:#94a3b8;'>{img_data['width']}×{img_data['height']} px • {img_data['file_size']/1024:.1f} KB</small>", unsafe_allow_html=True)

                        # Actions
                        btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
                        with btn_c1:
                            if st.button("🎨 Edit", key=f"edit_btn_{img_data['id']}", use_container_width=True):
                                st.session_state.selected_image_id = img_data["id"]
                                st.session_state.selected_version_id = latest_v["id"] if latest_v else None
                                st.session_state.active_nav = "Editor"
                                st.rerun()
                        with btn_c2:
                            # Download
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
                            if st.button("🗑️ Del", key=f"del_btn_{img_data['id']}", use_container_width=True):
                                success, paths = database.delete_image(img_data["id"])
                                for p in paths:
                                    storage.delete_file_safely(p)
                                search.remove_image_from_index(img_data["id"])
                                if st.session_state.selected_image_id == img_data["id"]:
                                    st.session_state.selected_image_id = None
                                st.success("Deleted image.")
                                time.sleep(0.3)
                                st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)


# ==================== Page 2: Image Editor & Version History ====================

def render_editor():
    st.markdown("### 🎨 AI Image Studio & Version History")

    all_images = database.list_images(limit=100)
    if not all_images:
        st.info("ℹ️ No images found. Please upload an image in the Library first!")
        if st.button("Go to Library", type="primary"):
            st.session_state.active_nav = "Library"
            st.rerun()
        return

    # Image selector dropdown
    img_options = {img["id"]: f"{img['original_name']} ({img['width']}×{img['height']} px)" for img in all_images}
    default_idx = 0
    if st.session_state.selected_image_id in img_options:
        default_idx = list(img_options.keys()).index(st.session_state.selected_image_id)

    selected_id = st.selectbox(
        "Select Image to Edit:",
        options=list(img_options.keys()),
        index=default_idx,
        format_func=lambda x: img_options[x]
    )
    st.session_state.selected_image_id = selected_id

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
    view_col, ctrl_col = st.columns([1.2, 1.0])

    with view_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Header with version badge
        head_c1, head_c2 = st.columns([2, 1])
        with head_c1:
            ver_num = active_info.get("version_number", 0)
            if ver_num == 0:
                st.markdown("**Active View:** Original Upload")
            else:
                st.markdown(f"**Active View:** <span class='badge badge-version'>Version {ver_num}</span>", unsafe_allow_html=True)
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
                st.caption(f"⏮️ **Before** ({comp_meta.get('version_number', 0) == 0 and 'Original' or 'Parent'})")
                if before_img:
                    st.image(before_img, use_container_width=True)
            with col_a:
                st.caption(f"⏭️ **After** (v{comp_meta.get('version_number', 0)})")
                if after_img:
                    st.image(after_img, use_container_width=True)
        else:
            st.image(active_pil, use_container_width=True)

        st.caption(f"Resolution: {active_pil.width} × {active_pil.height} px | Method: `{active_info.get('method', 'original')}`")

        # Download active button
        dl_buf = BytesIO()
        active_pil.save(dl_buf, format="PNG")
        st.download_button(
            label="📥 Download Current Version (PNG)",
            data=dl_buf.getvalue(),
            file_name=f"pixelcraft_{selected_id[:8]}_v{active_info.get('version_number', 0)}.png",
            mime="image/png",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with ctrl_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ✨ Natural-Language Editing")
        st.caption("Type a prompt or choose a preset action below.")

        prompt_input = st.text_input(
            "Editing Instruction",
            value=st.session_state.editor_instruction,
            placeholder="e.g. 'Remove the background' or 'Change background to a snowy mountain'",
            key="nl_edit_input"
        )

        col_run, col_clear = st.columns([2, 1])
        with col_run:
            run_btn = st.button("🪄 Execute Edit", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.editor_instruction = ""
                st.rerun()

        st.markdown("##### Quick Presets")
        # Preset buttons arranged in clean grid
        p_row1_c1, p_row1_c2 = st.columns(2)
        with p_row1_c1:
            if st.button("✂️ Remove Background", use_container_width=True):
                st.session_state.editor_instruction = "Remove the background"
                run_btn = True
        with p_row1_c2:
            if st.button("🏔️ Snowy Mountain BG", use_container_width=True):
                st.session_state.editor_instruction = "Change the background to a snowy mountain"
                run_btn = True

        p_row2_c1, p_row2_c2 = st.columns(2)
        with p_row2_c1:
            if st.button("🌅 Sunset Glow BG", use_container_width=True):
                st.session_state.editor_instruction = "Replace background with sunset"
                run_btn = True
        with p_row2_c2:
            if st.button("🏢 Dark Studio BG", use_container_width=True):
                st.session_state.editor_instruction = "Change background to dark studio"
                run_btn = True

        p_row3_c1, p_row3_c2 = st.columns(2)
        with p_row3_c1:
            if st.button("☀️ Brighten & Warm", use_container_width=True):
                st.session_state.editor_instruction = "Make the image brighter and warmer"
                run_btn = True
        with p_row3_c2:
            if st.button("🎬 Cinematic Look", use_container_width=True):
                st.session_state.editor_instruction = "Apply cinematic teal and orange filter"
                run_btn = True

        p_row4_c1, p_row4_c2 = st.columns(2)
        with p_row4_c1:
            if st.button("✏️ Pencil Sketch", use_container_width=True):
                st.session_state.editor_instruction = "Convert to pencil sketch style"
                run_btn = True
        with p_row4_c2:
            if st.button("🔲 Classic Noir B&W", use_container_width=True):
                st.session_state.editor_instruction = "Make it black and white noir"
                run_btn = True

        p_row5_c1, p_row5_c2 = st.columns(2)
        with p_row5_c1:
            if st.button("📐 Crop 1:1 Square", use_container_width=True):
                st.session_state.editor_instruction = "Crop to square aspect ratio 1:1"
                run_btn = True
        with p_row5_c2:
            if st.button("📐 Crop 16:9 Cinema", use_container_width=True):
                st.session_state.editor_instruction = "Crop to aspect ratio 16:9"
                run_btn = True

        # Manual Fine-Tuning Sliders Accordion
        with st.expander("🎛️ Manual Sliders & Color Balance"):
            slider_b = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
            slider_c = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
            slider_s = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
            slider_sat = st.slider("Saturation", 0.0, 2.5, 1.0, 0.1)
            slider_warm = st.slider("Warmth (Cool ↔ Warm)", -100, 100, 0, 5)
            slider_sepia = st.checkbox("Sepia Vintage Effect", value=False)

            if st.button("Apply Manual Tuning", use_container_width=True):
                with st.spinner("Applying manual parameters..."):
                    # Apply lighting then color
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

        st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown("---")
    st.markdown("#### 📜 Version History & Rollback Timeline")
    st.caption("Each edit produces a non-destructive version. The original image remains preserved forever.")

    history = versioning.get_version_history(selected_id)
    orig_row = database.get_image(selected_id)

    # Timeline cards
    v_cols = st.columns(max(len(history) + 1, 1))

    # Version 0: Original
    with v_cols[0]:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown("**Original**")
        if orig_row and os.path.exists(orig_row["file_path"]):
            st.image(orig_row["file_path"], use_container_width=True)
        st.caption(f"{orig_row['created_at'][:16] if orig_row else ''}")
        if st.button("⏮️ Revert to Original", key="revert_orig", use_container_width=True):
            st.session_state.selected_version_id = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Subsequent versions
    for idx, v in enumerate(history):
        with v_cols[idx + 1]:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown(f"**Version {v['version_number']}**")
            if os.path.exists(v["file_path"]):
                st.image(v["file_path"], use_container_width=True)
            st.caption(f"_{v['edit_prompt'][:30]}_")
            st.caption(f"<small>{v['created_at'][:16]}</small>", unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns(2)
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

            st.markdown('</div>', unsafe_allow_html=True)


# ==================== Page 3: Natural Language Search ====================

def render_search():
    st.markdown("### 🔍 Natural-Language Image Search")
    st.markdown("Find images using descriptive natural language concepts (e.g. *'beach at golden hour'*, *'mountains with snow'*, *'red car'*, *'portrait with high contrast'*).")

    engine_info = embeddings.get_active_embedding_engine_info()

    # Search Bar & Settings
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        query = st.text_input(
            "Natural-Language Search Query",
            placeholder="e.g. 'golden sunset', 'snowy mountains', 'vibrant colors', 'portrait'",
            key="search_query_input"
        )
    with s_col2:
        min_thresh = st.slider("Minimum Relevance %", min_value=0, max_value=80, value=10, step=5)

    # Suggestion Chips
    st.markdown("<small style='color:#94a3b8;'>Try sample searches: </small>", unsafe_allow_html=True)
    chip_cols = st.columns(5)
    samples = ["sunset", "snowy mountain", "high-contrast", "portrait", "vibrant colors"]
    for i, s in enumerate(samples):
        with chip_cols[i]:
            if st.button(f"🔍 {s}", key=f"sample_q_{i}", use_container_width=True):
                query = s
                st.session_state.search_query_input = s

    st.markdown("<br>", unsafe_allow_html=True)

    if not query.strip():
        st.info("💡 Enter a search prompt above to query the catalog.")
        return

    # Execute Search
    with st.spinner(f"Searching catalog for '{query}'..."):
        search_res = search.search_images(query, min_score=(min_thresh / 100.0))

    results = search_res.get("results", [])
    engine_used = search_res.get("engine_used", "Search Engine")

    # Engine badge
    st.markdown(
        f"**Search Engine:** <span class='badge badge-semantic'>{engine_used}</span> • **Found:** {len(results)} matches",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not results:
        st.warning(f"No images matched query '{query}' above {min_thresh}% relevance threshold. Try broadening your keywords or lowering the threshold.")
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
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    if os.path.exists(display_path):
                        st.image(display_path, use_container_width=True)

                    # Relevance Score Bar
                    st.markdown(
                        f"**{img_data['original_name']}** — <span class='badge badge-semantic'>{score_pct}% Match</span>",
                        unsafe_allow_html=True
                    )
                    st.progress(score_pct / 100.0)

                    st.caption(f"📝 {img_data.get('caption', '')[:120]}")

                    if img_data.get("tags"):
                        tag_html = "".join([f'<span class="tag-chip">{t.strip()}</span>' for t in img_data["tags"].split(",")[:4]])
                        st.markdown(tag_html, unsafe_allow_html=True)

                    if latest_v:
                        st.caption(f"Latest Edit: *{latest_v.get('edit_prompt', '')}*")

                    if st.button("🎨 Open in Studio", key=f"open_search_{img_data['id']}", use_container_width=True, type="primary"):
                        st.session_state.selected_image_id = img_data["id"]
                        st.session_state.selected_version_id = latest_v["id"] if latest_v else None
                        st.session_state.active_nav = "Editor"
                        st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)


# ==================== Page 4: System & Diagnostics ====================

def render_settings():
    st.markdown("### ⚙️ System Status, Index Management & Diagnostics")
    st.markdown("Monitor offline system health, embedding pipelines, SQLite tables, and manage search indexes.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 AI Model Pipelines")
        engine_info = embeddings.get_active_embedding_engine_info()
        st.json(engine_info)

        st.markdown("#### ✂️ Segmentation Engine (`rembg`)")
        st.json({
            "rembg_installed": editing.HAS_REMBG,
            "backend": "onnxruntime U2-Net CPU" if editing.HAS_REMBG else "Fallback Color-Key Luminosity Matting",
            "offline_capable": True,
            "cost_per_inference": "₹0.00 (Zero Credits)"
        })
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🗄️ Database & Storage Diagnostics")
        db_path = database.DEFAULT_DB_PATH
        img_count = database.count_images()
        all_embs = database.get_all_embeddings()

        # Disk size
        def get_dir_size_mb(path):
            total = 0
            if os.path.exists(path):
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total += os.path.getsize(fp)
            return round(total / (1024 * 1024), 2)

        st.json({
            "sqlite_db_path": db_path,
            "total_images_in_db": img_count,
            "persisted_embeddings": len(all_embs),
            "uploads_dir_size_mb": get_dir_size_mb(storage.UPLOADS_DIR),
            "versions_dir_size_mb": get_dir_size_mb(storage.VERSIONS_DIR)
        })

        st.markdown("#### 🔄 Index Rebuilder")
        st.caption("Re-compute all dense embeddings and resynchronize the database catalog.")
        if st.button("Rebuild Entire Search Index Now", use_container_width=True):
            with st.spinner("Rebuilding search index..."):
                res = search.rebuild_full_index()
                st.success(f"Indexed {res['semantic_indexed']} of {res['total_images']} images successfully!")
                time.sleep(0.5)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔑 External Generative API Settings (Optional)")
    st.info("External vision and image-editing APIs are 100% OPTIONAL and disabled by default. The entire application runs natively offline without paid credits.")

    with st.expander("Configure Optional Cloud Keys (Stored in memory / session only)"):
        st.caption("If you provide an OpenAI or Gemini API key, it will be used for optional generative inpainting or cloud vision. If left blank, zero-cost local AI is used.")
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
