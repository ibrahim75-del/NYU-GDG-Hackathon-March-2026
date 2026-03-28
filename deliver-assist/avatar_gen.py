"""
avatar_gen.py — Generate AI avatar expressions using Nano Banana (Gemini image generation).
Produces 6 expression variants of a consistent delivery-worker character in parallel.
"""

import asyncio
import base64
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Base character description — kept IDENTICAL across all prompts for consistency
CHARACTER = (
    "A friendly young delivery worker character, cartoon illustration style, "
    "warm brown skin, short black hair, wearing a bright green jacket and bike helmet, "
    "simple clean background in soft blue gradient, "
    "portrait from chest up, facing the camera, high quality digital art, "
    "Pixar-style 3D render, expressive face, warm lighting"
)

EXPRESSIONS = {
    "neutral":    f"{CHARACTER}, calm neutral relaxed expression, slight gentle smile, looking directly at viewer",
    "happy":      f"{CHARACTER}, big warm genuine smile, eyes slightly squinted with joy, friendly and welcoming",
    "empathetic": f"{CHARACTER}, soft caring expression, gentle concerned eyes, slight sympathetic smile, head tilted slightly",
    "concerned":  f"{CHARACTER}, worried concerned expression, slight frown, eyebrows raised with worry, caring look",
    "thinking":   f"{CHARACTER}, thoughtful expression, looking slightly up and to the side, one eyebrow raised, pondering",
    "speaking":   f"{CHARACTER}, mid-speech expression, mouth slightly open as if talking, engaged and animated, making eye contact",
}


def _extract_b64(response) -> str:
    """Pull first inline image bytes out of a generate_content response."""
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    return base64.b64encode(data).decode("utf-8")
                if isinstance(data, str):
                    # Already base64
                    return data
    return ""


async def generate_one(name: str, prompt: str) -> tuple[str, str]:
    """Generate a single expression image using nano-banana-pro-preview."""
    try:
        response = await client.aio.models.generate_content(
            model="models/nano-banana-pro-preview",
            contents=prompt,
        )
        b64 = _extract_b64(response)
        if b64:
            print(f"[Avatar] ✓ {name} ({len(b64) // 1024}KB)")
            return (name, b64)
        print(f"[Avatar] No image data for {name}")
    except Exception as e:
        print(f"[Avatar] Failed for {name}: {e}")

    return (name, "")


async def generate_avatar_set() -> dict[str, str]:
    """
    Generate all 6 expression images in parallel.
    Returns dict of {expression_name: base64_jpeg_string}.
    Takes ~5-15s total (parallel requests).
    """
    tasks = [generate_one(name, prompt) for name, prompt in EXPRESSIONS.items()]
    results = await asyncio.gather(*tasks)

    avatar_set = {}
    for name, b64 in results:
        if b64:
            avatar_set[name] = b64
        else:
            print(f"[Avatar] Both models failed for {name} — will use fallback UI")

    print(f"[Avatar] Generation complete: {len(avatar_set)}/6 succeeded")
    return avatar_set
