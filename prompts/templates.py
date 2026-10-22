"""
Documented Prompt Templates for AI-Powered Image Editing and Analysis.
Provides standard, reusable system and task prompts for vision analysis,
natural-language edit parsing, and optional generative adapters.
"""

# ==================== Vision & Captioning Prompts ====================

VISION_CAPTION_SYSTEM_PROMPT = """You are an expert AI vision assistant specialized in high-detail image cataloging.
Describe the given image accurately, concisely, and objectively.
Follow this format:
- Caption: A single descriptive sentence summarizing the primary subject and setting.
- Description: 2-3 sentences detailing colors, lighting, composition, and prominent secondary objects.
- Tags: 5-8 relevant comma-separated lowercase tags for search indexing.
Do not hallucinate elements not visible in the image."""

VISION_CAPTION_USER_PROMPT = """Analyze this image and return:
Caption: <one clear sentence>
Description: <2-3 descriptive sentences>
Tags: <tag1, tag2, tag3, tag4, tag5>"""


# ==================== Natural Language Edit Parsing Prompts ====================

EDIT_INTERPRETATION_SYSTEM_PROMPT = """You are an AI photo editing intent parser.
Given a user's natural language request to edit an image, identify the desired operation and extract parameters.
Supported operations:
1. "background_removal": Removing the background to make it transparent.
2. "background_replacement": Replacing background with solid color, gradient, or custom theme (e.g. snowy mountain, sunset).
3. "color_adjustment": Adjusting saturation, warmth/temperature, tint, sepia, or monochrome.
4. "lighting_enhancement": Adjusting brightness, contrast, shadows, highlights, or sharpness.
5. "image_restyling": Applying artistic styles (vintage, cinematic, neon/cyberpunk, sketch, watercolor).
6. "object_removal": Removing a specific foreground object or blemish.
7. "crop_transform": Changing aspect ratio (1:1, 16:9, 4:3, 9:16), rotating, or flipping.
8. "unsupported_generative": Adding complex new subjects or semantic modifications that require a text-to-image generative diffusion model.

Return valid JSON with:
{
    "action": "<operation_name>",
    "parameters": { ... },
    "is_generative_required": true/false,
    "user_explanation": "<concise explanation of how this instruction is interpreted>"
}"""


# ==================== Generative Edit Prompts (Optional Cloud Adapters) ====================

GENERATIVE_INPAINTING_PROMPT_TEMPLATE = """Inpaint the masked region of the image according to this instruction:
User instruction: "{instruction}"
Requirements: Maintain photographic realism, seamless edge blending, and matching illumination with surrounding pixels."""

GENERATIVE_BG_REPLACE_PROMPT_TEMPLATE = """Preserve the main foreground subject unchanged.
Replace the background with: "{backdrop_description}".
Lighting on foreground subject should harmonize naturally with the new environment."""

GENERATIVE_RESTYLING_PROMPT_TEMPLATE = """Restyle the photograph in the style of: "{style_description}".
Retain the overall composition, contours, and primary subjects while applying the new aesthetic palette and texture."""
