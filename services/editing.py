"""
Image Editing Service for AI-Powered Image Editing Platform.
Provides local deterministic transformations, local AI segmentation (rembg),
natural language instruction parsing, and modular external generative adapters.
Zero-cost baseline runs 100% locally with honest AI disclosure.
"""

import os
import re
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

logger = logging.getLogger(__name__)

# Check if rembg is available for local AI background removal
HAS_REMBG = False
try:
    import rembg
    HAS_REMBG = True
except Exception as e:
    logger.info("rembg is not available or failed to load: %s", e)


# ==================== Local Transformations ====================

def adjust_lighting(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    auto_contrast: bool = False
) -> Image.Image:
    """Adjust lighting, brightness, contrast, and sharpness using Pillow."""
    img = image.copy()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    if auto_contrast:
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb = ImageOps.autocontrast(rgb, cutoff=1)
            r2, g2, b2 = rgb.split()
            img = Image.merge("RGBA", (r2, g2, b2, a))
        else:
            img = ImageOps.autocontrast(img, cutoff=1)

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

    return img


def adjust_color(
    image: Image.Image,
    saturation: float = 1.0,
    warmth: int = 0,
    tint: int = 0,
    sepia: bool = False,
    vibrance: float = 1.0
) -> Image.Image:
    """
    Adjust color saturation, warmth/temperature (-100 to +100),
    tint (-100 to +100), and sepia tone.
    """
    img = image.copy()
    has_alpha = img.mode == "RGBA"
    alpha_channel = None
    if has_alpha:
        alpha_channel = img.split()[-1]
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    # Warmth and Tint in NumPy
    if warmth != 0 or tint != 0:
        arr = np.array(img, dtype=np.float32)
        # Warmth: add to Red, subtract from Blue
        w_factor = warmth * 0.4
        arr[:, :, 0] += w_factor
        arr[:, :, 2] -= w_factor

        # Tint: add to Green (tint < 0) or add to Magenta/R+B (tint > 0)
        t_factor = tint * 0.3
        arr[:, :, 1] -= t_factor
        arr[:, :, 0] += t_factor * 0.5
        arr[:, :, 2] += t_factor * 0.5

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Sepia tone
    if sepia:
        arr = np.array(img, dtype=np.float32)
        r = arr[:, :, 0] * 0.393 + arr[:, :, 1] * 0.769 + arr[:, :, 2] * 0.189
        g = arr[:, :, 0] * 0.349 + arr[:, :, 1] * 0.686 + arr[:, :, 2] * 0.168
        b = arr[:, :, 0] * 0.272 + arr[:, :, 1] * 0.534 + arr[:, :, 2] * 0.131
        sepia_arr = np.stack([r, g, b], axis=-1)
        sepia_arr = np.clip(sepia_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(sepia_arr)

    if has_alpha and alpha_channel:
        img = img.convert("RGBA")
        img.putalpha(alpha_channel)

    return img


def remove_background_local(image: Image.Image) -> Tuple[Image.Image, str]:
    """
    Remove background using local AI (rembg U2-Net) or color thresholding fallback.
    Returns (rgba_image, method_description).
    """
    img = image.copy()
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if HAS_REMBG:
        try:
            result = rembg.remove(img)
            return result, "Local AI (U2-Net Neural Network via rembg)"
        except Exception as e:
            logger.warning(f"rembg processing failed: {e}. Falling back to thresholding.")

    # Fallback: Luminosity / edge-based transparency fallback
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    # Estimate background from the 4 corners
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg_color = np.mean(corners, axis=0)

    # Euclidean color distance from estimated background
    diff = np.sqrt(np.sum((arr - bg_color) ** 2, axis=2))
    threshold = 35.0
    alpha = np.clip((diff - threshold) * 8.0, 0, 255).astype(np.uint8)

    # Smooth the alpha mask
    mask_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=1.5))
    result = img.copy()
    result.putalpha(mask_img)
    return result, "Local Color-Key Luminosity Thresholding (Pillow Fallback)"


def generate_backdrop(width: int, height: int, backdrop_type: str) -> Image.Image:
    """Generate high-quality procedural backdrops for background replacement."""
    backdrop_type = backdrop_type.lower().strip()
    backdrop = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(backdrop)

    if "snow" in backdrop_type or "mountain" in backdrop_type:
        # Snowy Mountain gradient: cold sky blue to snowy mist with mountain silhouettes
        for y in range(height):
            ratio = y / max(height - 1, 1)
            # Sky from deep alpine blue (70, 130, 180) to misty snow white (235, 245, 255)
            r = int(70 + (235 - 70) * ratio)
            g = int(130 + (245 - 130) * ratio)
            b = int(180 + (255 - 180) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

        # Draw soft mountain ridges in the lower 40%
        mountain_points = [
            (0, height),
            (0, int(height * 0.65)),
            (int(width * 0.25), int(height * 0.45)),
            (int(width * 0.5), int(height * 0.60)),
            (int(width * 0.75), int(height * 0.40)),
            (width, int(height * 0.55)),
            (width, height)
        ]
        draw.polygon(mountain_points, fill=(180, 205, 225, 220))

        foreground_ridge = [
            (0, height),
            (0, int(height * 0.8)),
            (int(width * 0.4), int(height * 0.65)),
            (int(width * 0.7), int(height * 0.75)),
            (width, int(height * 0.68)),
            (width, height)
        ]
        draw.polygon(foreground_ridge, fill=(240, 248, 255, 240))
        backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=3))

    elif "sunset" in backdrop_type or "golden" in backdrop_type:
        # Warm sunset gradient: deep violet -> orange -> warm yellow
        for y in range(height):
            ratio = y / max(height - 1, 1)
            if ratio < 0.5:
                sub_r = ratio / 0.5
                r = int(75 + (235 - 75) * sub_r)
                g = int(35 + (90 - 35) * sub_r)
                b = int(110 + (30 - 110) * sub_r)
            else:
                sub_r = (ratio - 0.5) / 0.5
                r = int(235 + (255 - 235) * sub_r)
                g = int(90 + (200 - 90) * sub_r)
                b = int(30 + (80 - 30) * sub_r)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    elif "ocean" in backdrop_type or "sea" in backdrop_type or "blue" in backdrop_type:
        # Deep coastal ocean gradient
        for y in range(height):
            ratio = y / max(height - 1, 1)
            r = int(20 + (5 - 20) * ratio)
            g = int(90 + (160 - 90) * ratio)
            b = int(140 + (190 - 140) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    elif "studio" in backdrop_type or "dark" in backdrop_type or "black" in backdrop_type:
        # Dark studio backdrop with subtle radial light vignette
        cx, cy = width / 2, height / 2
        max_dist = np.sqrt(cx**2 + cy**2)
        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        norm_dist = np.clip(dist_from_center / max_dist, 0, 1)

        val = np.clip(60 - norm_dist * 45, 15, 60).astype(np.uint8)
        arr = np.dstack([val, val, val, np.full((height, width), 255, dtype=np.uint8)])
        backdrop = Image.fromarray(arr, mode="RGBA")

    elif "white" in backdrop_type or "light" in backdrop_type:
        backdrop = Image.new("RGBA", (width, height), (250, 250, 252, 255))

    elif "neon" in backdrop_type or "cyber" in backdrop_type:
        # Cyberpunk purple/cyan gradient
        for y in range(height):
            ratio = y / max(height - 1, 1)
            r = int(180 * (1 - ratio) + 15 * ratio)
            g = int(20 * (1 - ratio) + 220 * ratio)
            b = int(220 * (1 - ratio) + 240 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    else:
        # Default smooth neutral modern gray gradient
        for y in range(height):
            ratio = y / max(height - 1, 1)
            v = int(240 - 70 * ratio)
            draw.line([(0, y), (width, y)], fill=(v, v, v, 255))

    return backdrop


def replace_background_local(
    image: Image.Image,
    backdrop_type: str = "white",
    custom_backdrop_img: Optional[Image.Image] = None
) -> Tuple[Image.Image, str]:
    """
    Remove existing background and composite isolated subject over new backdrop.
    Returns (composite_image, method_description).
    """
    # 1. Isolate subject
    isolated_subject, seg_method = remove_background_local(image)
    w, h = isolated_subject.size

    # 2. Prepare backdrop
    if custom_backdrop_img is not None:
        bg = custom_backdrop_img.copy().resize((w, h), Image.Resampling.LANCZOS)
        if bg.mode != "RGBA":
            bg = bg.convert("RGBA")
        method_desc = f"Subject Isolated via {seg_method} + Custom Backdrop Composite"
    else:
        bg = generate_backdrop(w, h, backdrop_type)
        method_desc = f"Subject Isolated via {seg_method} + Procedural '{backdrop_type}' Backdrop Composite"

    # 3. Alpha composite
    composite = Image.alpha_composite(bg, isolated_subject)
    return composite, method_desc


def restyle_image(image: Image.Image, style_name: str) -> Tuple[Image.Image, str]:
    """Apply artistic styling filters."""
    style_name = style_name.lower().strip()
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    if "vintage" in style_name or "retro" in style_name:
        # Warm, slightly low-contrast, amber glow with soft vignette
        img = adjust_color(img, saturation=0.85, warmth=35, tint=10)
        img = adjust_lighting(img, brightness=1.05, contrast=0.92)
        # Add subtle vignette
        w, h = img.size
        cx, cy = w / 2, h / 2
        max_dist = np.sqrt(cx**2 + cy**2)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_dist
        vignette_mask = 1.0 - np.clip(dist**2 * 0.5, 0, 0.45)
        arr = (np.array(img, dtype=np.float32) * vignette_mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr), "Artistic Filter: Vintage 70s Warmth with Vignette"

    elif "cinematic" in style_name or "teal" in style_name:
        # Teal & Orange Hollywood cinema look
        arr = np.array(img, dtype=np.float32)
        luminance = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
        # Shadows get cool teal (more blue/green), highlights get warm orange (more red/green)
        arr[:, :, 0] += (luminance > 0.5) * (luminance - 0.5) * 60  # Red highlights
        arr[:, :, 2] += (luminance <= 0.5) * (0.5 - luminance) * 70  # Blue shadows
        arr[:, :, 1] += (luminance <= 0.5) * (0.5 - luminance) * 30  # Green shadows
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        cinematic = Image.fromarray(arr)
        cinematic = adjust_lighting(cinematic, contrast=1.18, sharpness=1.2)
        return cinematic, "Artistic Filter: Cinematic Teal & Orange"

    elif "neon" in style_name or "cyberpunk" in style_name:
        # Cyberpunk Neon: punchy magenta, electric cyan, high contrast
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.35, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.45, 0, 255)
        neon = Image.fromarray(arr.astype(np.uint8))
        neon = adjust_lighting(neon, brightness=1.1, contrast=1.35, sharpness=1.3)
        return neon, "Artistic Filter: Cyberpunk Neon Glow"

    elif "sketch" in style_name or "pencil" in style_name:
        # High quality pencil sketch via Gaussian dodge blending
        gray = img.convert("L")
        inv = ImageOps.invert(gray)
        blur = inv.filter(ImageFilter.GaussianBlur(radius=15))
        # Color dodge blend: gray / (255 - blur)
        g_arr = np.array(gray, dtype=np.float32)
        b_arr = np.array(blur, dtype=np.float32)
        denom = 255.0 - b_arr
        denom[denom == 0] = 1.0
        sketch = np.clip((g_arr * 255.0) / denom, 0, 255).astype(np.uint8)
        sketch_img = Image.fromarray(sketch).convert("RGB")
        return sketch_img, "Artistic Filter: Pencil Sketch Line Art"

    elif "noir" in style_name or "black and white" in style_name or "monochrome" in style_name:
        # Classic Noir: high contrast monochrome with rich deep blacks
        bw = img.convert("L")
        bw = ImageOps.autocontrast(bw, cutoff=2)
        enhancer = ImageEnhance.Contrast(bw)
        bw = enhancer.enhance(1.4)
        return bw.convert("RGB"), "Artistic Filter: Classic Film Noir Black & White"

    elif "warm" in style_name or "golden" in style_name:
        # Golden hour warmth
        warm = adjust_color(img, saturation=1.15, warmth=40, tint=5)
        warm = adjust_lighting(warm, brightness=1.06, contrast=1.08)
        return warm, "Artistic Filter: Golden Hour Radiance"

    else:
        # Modern Crisp Pop
        crisp = adjust_lighting(img, brightness=1.05, contrast=1.15, sharpness=1.25)
        crisp = adjust_color(crisp, saturation=1.15)
        return crisp, f"Artistic Filter: {style_name.title()}"


def crop_or_transform(
    image: Image.Image,
    aspect_ratio: Optional[str] = None,
    rotate_deg: int = 0,
    flip_h: bool = False,
    flip_v: bool = False
) -> Tuple[Image.Image, str]:
    """Apply geometry transformations: crop aspect ratio, rotate, flip."""
    img = image.copy()
    w, h = img.size
    actions = []

    # Aspect ratio cropping (centered)
    if aspect_ratio:
        aspect_ratio = aspect_ratio.strip()
        target_ratio = None
        if aspect_ratio == "1:1":
            target_ratio = 1.0
        elif aspect_ratio in ["16:9", "16/9"]:
            target_ratio = 16.0 / 9.0
        elif aspect_ratio in ["4:3", "4/3"]:
            target_ratio = 4.0 / 3.0
        elif aspect_ratio in ["9:16", "9/16"]:
            target_ratio = 9.0 / 16.0

        if target_ratio:
            curr_ratio = w / h
            if curr_ratio > target_ratio:
                # Image is too wide
                new_w = int(h * target_ratio)
                left = (w - new_w) // 2
                img = img.crop((left, 0, left + new_w, h))
            else:
                # Image is too tall
                new_h = int(w / target_ratio)
                top = (h - new_h) // 2
                img = img.crop((0, top, w, top + new_h))
            actions.append(f"Aspect Crop ({aspect_ratio})")

    if rotate_deg != 0:
        # Rotate without clipping
        img = img.rotate(-rotate_deg, expand=True, resample=Image.Resampling.BICUBIC)
        actions.append(f"Rotated {rotate_deg}°")

    if flip_h:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        actions.append("Horizontal Flip")

    if flip_v:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        actions.append("Vertical Flip")

    return img, ", ".join(actions) if actions else "No Transform Applied"


def remove_object_local(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[Image.Image, str]:
    """
    Local patch-based inpainting / object removal using neighborhood interpolation.
    Explicitly discloses deterministic local method.
    """
    img = image.copy()
    w, h = img.size

    # Default to center-left bounding box if none provided
    if bbox is None:
        # e.g., left 25% to 45%
        bbox = (int(w * 0.15), int(h * 0.25), int(w * 0.45), int(h * 0.75))

    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

    arr = np.array(img)
    # Estimate replacement patch from surrounding neighborhood border pixels
    border_pixels = []
    if y1 > 0:
        border_pixels.append(arr[max(0, y1 - 8):y1, x1:x2])
    if y2 < h:
        border_pixels.append(arr[y2:min(h, y2 + 8), x1:x2])
    if x1 > 0:
        border_pixels.append(arr[y1:y2, max(0, x1 - 8):x1])
    if x2 < w:
        border_pixels.append(arr[y1:y2, x2:min(w, x2 + 8)])

    if border_pixels:
        all_border = np.concatenate([p.reshape(-1, arr.shape[2]) for p in border_pixels], axis=0)
        mean_color = np.median(all_border, axis=0).astype(arr.dtype)
        # Apply smooth blend
        patch = np.tile(mean_color, (y2 - y1, x2 - x1, 1))
        arr[y1:y2, x1:x2] = patch

    patched_img = Image.fromarray(arr)
    # Smooth patch seams
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

    result = Image.composite(patched_img, img, mask)
    method_desc = "Local Deterministic Neighborhood Inpainting (Patch Blend) — Note: complex generative removal requires external diffusion API"
    return result, method_desc


# ==================== Natural Language Instruction Parser ====================

def parse_natural_language_instruction(prompt: str) -> Dict[str, Any]:
    """
    Parse natural language image editing prompt into structured action and parameters.
    Zero-cost rule-based and regex engine.
    """
    p = prompt.lower().strip()

    # 1. Background Removal
    if ("remove" in p or "cut out" in p or "isolate" in p or "erase" in p) and "background" in p:
        return {
            "action": "background_removal",
            "parameters": {},
            "is_generative": False,
            "explanation": "Isolating foreground subject and making background transparent using local AI (U2-Net)."
        }

    # 2. Background Replacement
    if ("replace" in p or "change" in p or "switch" in p or "put" in p or "set" in p) and "background" in p:
        # Extract target background
        target = "neutral"
        if "snow" in p or "mountain" in p:
            target = "snowy mountain"
        elif "sunset" in p or "golden" in p:
            target = "sunset"
        elif "ocean" in p or "sea" in p or "beach" in p or "water" in p:
            target = "ocean"
        elif "dark" in p or "studio" in p or "black" in p:
            target = "dark studio"
        elif "white" in p or "plain" in p:
            target = "white"
        elif "neon" in p or "cyber" in p:
            target = "neon cyberpunk"
        else:
            match = re.search(r"(?:to|with|into)\s+(?:a\s+|an\s+)?([a-z\s]+)", p)
            if match:
                target = match.group(1).replace("background", "").strip()

        return {
            "action": "background_replacement",
            "parameters": {"backdrop_type": target},
            "is_generative": False,
            "explanation": f"Replacing background with procedural '{target}' backdrop."
        }

    # 3. Object Removal
    if "remove" in p and ("person" in p or "object" in p or "left" in p or "right" in p or "item" in p or "blemish" in p or "spot" in p):
        return {
            "action": "object_removal",
            "parameters": {"target": p},
            "is_generative": False,
            "explanation": "Applying local spatial neighborhood inpainting to remove the specified area."
        }

    # 4. Color / Temperature / Saturation
    if any(k in p for k in ["warm", "warmer", "cool", "cooler", "saturat", "vibrant", "black and white", "b&w", "grayscale", "sepia", "tint", "color"]):
        saturation = 1.0
        warmth = 0
        tint = 0
        sepia = False

        if "black and white" in p or "b&w" in p or "grayscale" in p or "monochrome" in p:
            saturation = 0.0
        elif "vibrant" in p or "hyper" in p or "saturat" in p:
            saturation = 1.45
        elif "desaturat" in p or "muted" in p:
            saturation = 0.65

        if "warmer" in p or "warm" in p:
            warmth = 35
        elif "cooler" in p or "cool" in p or "cold" in p:
            warmth = -35

        if "sepia" in p:
            sepia = True

        return {
            "action": "color_adjustment",
            "parameters": {
                "saturation": saturation,
                "warmth": warmth,
                "tint": tint,
                "sepia": sepia
            },
            "is_generative": False,
            "explanation": f"Adjusting color balance (saturation={saturation}, warmth={warmth}, sepia={sepia})."
        }

    # 5. Lighting / Brightness / Contrast / Sharpness
    if any(k in p for k in ["bright", "brighter", "darker", "contrast", "sharp", "sharpen", "lighting", "exposure", "autocontrast"]):
        brightness = 1.0
        contrast = 1.0
        sharpness = 1.0
        auto = False

        if "brighter" in p or "bright" in p or "light" in p:
            brightness = 1.25
        elif "darker" in p or "dim" in p:
            brightness = 0.8

        if "high contrast" in p or "more contrast" in p:
            contrast = 1.3
        elif "low contrast" in p or "less contrast" in p:
            contrast = 0.8
        elif "contrast" in p:
            contrast = 1.2

        if "sharp" in p:
            sharpness = 1.5

        if "auto" in p:
            auto = True

        return {
            "action": "lighting_enhancement",
            "parameters": {
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "auto_contrast": auto
            },
            "is_generative": False,
            "explanation": f"Enhancing lighting (brightness={brightness}, contrast={contrast}, sharpness={sharpness})."
        }

    # 6. Restyling / Artistic filters
    if any(k in p for k in ["vintage", "retro", "cinematic", "teal", "neon", "cyberpunk", "sketch", "pencil", "noir", "golden hour"]):
        style = "vintage"
        if "cinematic" in p or "teal" in p:
            style = "cinematic"
        elif "neon" in p or "cyberpunk" in p:
            style = "neon"
        elif "sketch" in p or "pencil" in p:
            style = "sketch"
        elif "noir" in p:
            style = "noir"
        elif "golden" in p or "warm" in p:
            style = "golden"

        return {
            "action": "image_restyling",
            "parameters": {"style": style},
            "is_generative": False,
            "explanation": f"Applying artistic filter '{style}'."
        }

    # 7. Aspect ratio / Crop
    if any(k in p for k in ["crop", "square", "1:1", "16:9", "4:3", "9:16", "aspect"]):
        ratio = "1:1"
        if "16:9" in p or "widescreen" in p:
            ratio = "16:9"
        elif "4:3" in p:
            ratio = "4:3"
        elif "9:16" in p or "story" in p or "vertical" in p:
            ratio = "9:16"
        return {
            "action": "crop_transform",
            "parameters": {"aspect_ratio": ratio},
            "is_generative": False,
            "explanation": f"Cropping image to aspect ratio {ratio}."
        }

    # 8. Generative Object Addition / Modification (e.g. "Add a tree in background")
    if "add" in p or "insert" in p or "generate" in p or "draw" in p:
        return {
            "action": "unsupported_generative",
            "parameters": {"prompt": p},
            "is_generative": True,
            "explanation": "Adding arbitrary new objects (such as trees or animals) requires an external generative diffusion API (e.g., OpenAI DALL-E or Stability AI). The zero-cost local engine preserves image integrity and does not hallucinate artificial objects."
        }

    # Default fallback: general enhancement
    return {
        "action": "lighting_enhancement",
        "parameters": {"brightness": 1.1, "contrast": 1.15, "sharpness": 1.2},
        "is_generative": False,
        "explanation": "Interpreted instruction as general clarity and lighting enhancement."
    }


# ==================== Central Execution Engine ====================

def execute_edit(
    image: Image.Image,
    instruction: str,
    custom_params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Image.Image], str, Dict[str, Any], bool]:
    """
    Execute an edit instruction on a PIL Image.
    Returns (edited_image_or_none, method_description, parameters_dict, success_boolean).
    """
    if custom_params:
        parsed = custom_params
    else:
        parsed = parse_natural_language_instruction(instruction)

    action = parsed.get("action", "")
    params = parsed.get("parameters", {})
    is_generative = parsed.get("is_generative", False)

    if action == "unsupported_generative":
        # Disclose truthfully without fabricating an edit
        logger.info("Declining generative addition without external API: %s", instruction)
        return None, parsed.get("explanation", "Generative addition unavailable in offline mode."), params, False

    try:
        if action == "background_removal":
            res, method = remove_background_local(image)
            return res, method, params, True

        elif action == "background_replacement":
            target = params.get("backdrop_type", "white")
            res, method = replace_background_local(image, backdrop_type=target)
            return res, method, params, True

        elif action == "color_adjustment":
            res = adjust_color(
                image,
                saturation=params.get("saturation", 1.0),
                warmth=params.get("warmth", 0),
                tint=params.get("tint", 0),
                sepia=params.get("sepia", False),
                vibrance=params.get("vibrance", 1.0)
            )
            method = "Local Deterministic Color Transformation (NumPy/Pillow)"
            return res, method, params, True

        elif action == "lighting_enhancement":
            res = adjust_lighting(
                image,
                brightness=params.get("brightness", 1.0),
                contrast=params.get("contrast", 1.0),
                sharpness=params.get("sharpness", 1.0),
                auto_contrast=params.get("auto_contrast", False)
            )
            method = "Local Lighting & Tone Enhancement (Pillow/ImageEnhance)"
            return res, method, params, True

        elif action == "image_restyling":
            style = params.get("style", "vintage")
            res, method = restyle_image(image, style)
            return res, method, params, True

        elif action == "object_removal":
            res, method = remove_object_local(image)
            return res, method, params, True

        elif action == "crop_transform":
            res, method = crop_or_transform(
                image,
                aspect_ratio=params.get("aspect_ratio"),
                rotate_deg=params.get("rotate_deg", 0),
                flip_h=params.get("flip_h", False),
                flip_v=params.get("flip_v", False)
            )
            return res, f"Local Geometry Transformation: {method}", params, True

        else:
            # Fallback
            res = adjust_lighting(image, brightness=1.1, contrast=1.1)
            return res, "Local General Enhancement", params, True

    except Exception as e:
        logger.error("Failed to execute edit: %s", e)
        return None, f"Edit execution error: {str(e)}", params, False
