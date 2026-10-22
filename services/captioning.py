"""
Image Analysis and Captioning Service for AI-Powered Image Editing Platform.
Provides local heuristic scene analysis, color profiling, searchable tag generation,
and an optional cloud vision adapter.
Runs 100% offline at ₹0 cost without external API keys.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image, ImageStat

from prompts import templates

logger = logging.getLogger(__name__)


def analyze_image_locally(image: Image.Image) -> Dict[str, Any]:
    """
    Perform deep heuristic visual analysis on a PIL image.
    Calculates color statistics, aspect ratio, lighting profile, and scene tags.
    """
    w, h = image.size
    aspect_ratio = w / h

    # Convert to RGB for uniform statistics
    rgb_img = image.convert("RGB")
    stat = ImageStat.Stat(rgb_img)

    # Average brightness (0 - 255)
    avg_r, avg_g, avg_b = stat.mean[:3]
    avg_brightness = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b)

    # Standard deviation (contrast indicator)
    r_std, g_std, b_std = stat.stddev[:3]
    avg_contrast = (r_std + g_std + b_std) / 3.0

    # Color temperature / warmth
    warmth_bias = avg_r - avg_b  # Positive = warm/amber, negative = cool/blue

    # Saturation heuristic via HSV
    hsv_img = rgb_img.convert("HSV")
    hsv_stat = ImageStat.Stat(hsv_img)
    avg_saturation = hsv_stat.mean[1]  # 0 to 255

    # Tags and classification logic
    tags = []
    composition_desc = ""

    # Aspect ratio classification
    if aspect_ratio >= 1.6:
        tags.extend(["panoramic", "widescreen", "landscape"])
        composition_desc = "wide landscape composition"
    elif aspect_ratio >= 1.2:
        tags.extend(["horizontal", "landscape"])
        composition_desc = "horizontal perspective"
    elif aspect_ratio <= 0.65:
        tags.extend(["vertical", "tall", "portrait"])
        composition_desc = "vertical story format"
    elif aspect_ratio <= 0.85:
        tags.extend(["portrait", "vertical"])
        composition_desc = "portrait orientation"
    else:
        tags.extend(["square", "balanced"])
        composition_desc = "balanced square framing"

    # Lighting classification
    if avg_brightness > 180:
        tags.extend(["high-key", "bright", "daylight"])
        lighting_desc = "bright, high-key illumination"
    elif avg_brightness < 70:
        tags.extend(["low-key", "moody", "shadows", "night"])
        lighting_desc = "dramatic low-key lighting with deep shadows"
    else:
        tags.extend(["natural-light", "balanced-exposure"])
        lighting_desc = "balanced natural lighting"

    # Contrast classification
    if avg_contrast > 65:
        tags.extend(["high-contrast", "vivid"])
    elif avg_contrast < 30:
        tags.extend(["soft", "low-contrast", "subtle"])

    # Color palette classification
    if avg_saturation < 25:
        tags.extend(["monochrome", "neutral", "black-and-white"])
        palette_desc = "monochromatic and muted palette"
    elif avg_saturation > 110:
        tags.extend(["vibrant", "colorful", "rich"])
        palette_desc = "rich, saturated colors"
    elif warmth_bias > 25:
        tags.extend(["warm", "golden-hour", "amber"])
        palette_desc = "warm golden tones"
    elif warmth_bias < -20:
        tags.extend(["cool", "cyan", "blue-hour"])
        palette_desc = "cool blue undertones"
    else:
        tags.extend(["natural-tones"])
        palette_desc = "natural color balance"

    # Generate synthesized caption
    caption = f"A {composition_desc} featuring {palette_desc} under {lighting_desc}."

    # Generate synthesized description
    description = (
        f"Image dimensions {w}x{h} px. Average luminosity is {avg_brightness:.1f}/255 "
        f"with contrast dispersion {avg_contrast:.1f}. Palette shows {palette_desc}."
    )

    # Deduplicate tags
    unique_tags = list(dict.fromkeys(tags))

    return {
        "caption": caption,
        "description": description,
        "tags": ", ".join(unique_tags),
        "metrics": {
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 2),
            "brightness": round(avg_brightness, 1),
            "contrast": round(avg_contrast, 1),
            "warmth_bias": round(warmth_bias, 1),
            "saturation": round(avg_saturation, 1)
        }
    }


def generate_caption_for_image(
    image: Image.Image,
    force_local: bool = True
) -> Dict[str, Any]:
    """
    Main captioning entry point.
    Runs local heuristic analysis by default ($0 cost, 100% offline).
    If an external vision API key is configured and force_local is False,
    can invoke the optional external adapter.
    """
    # Check for optional API key only if user opted into external vision
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if not force_local and (has_openai or has_gemini):
        # Optional cloud adapter
        try:
            return _call_optional_vision_api(image)
        except Exception as e:
            logger.warning(f"Cloud vision call failed ({e}). Falling back to local analysis.")

    # Default: 100% free local heuristic analysis
    analysis = analyze_image_locally(image)
    return {
        "caption": analysis["caption"],
        "description": analysis["description"],
        "tags": analysis["tags"],
        "source": "local_heuristics",
        "metrics": analysis["metrics"]
    }


def _call_optional_vision_api(image: Image.Image) -> Dict[str, Any]:
    """Optional external vision API caller for OpenAI or Gemini when configured."""
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": templates.VISION_CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": templates.VISION_CAPTION_USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            max_tokens=250
        )
        raw_text = response.choices[0].message.content or ""
        # Parse lines
        caption, desc, tags = "Photograph", raw_text, "photo, visual"
        for line in raw_text.splitlines():
            if line.lower().startswith("caption:"):
                caption = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:"):
                desc = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tags:"):
                tags = line.split(":", 1)[1].strip()

        return {
            "caption": caption,
            "description": desc,
            "tags": tags,
            "source": "openai_gpt4o_mini"
        }

    raise RuntimeError("No configured vision API key.")
