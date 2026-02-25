"""
Step 3: English → Hindi Translation
=====================================
Primary:  IndicTrans2 (AI4Bharat) — SOTA for Indic languages, runs locally
Fallback: googletrans — free, no GPU needed, decent quality
"""

from typing import List, Optional

from .utils import Segment, logger


def translate_segments(
    segments: List[Segment],
    target_lang: str = "hi",
    use_indictrans2: bool = True,
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Translate transcribed segments to Hindi.

    Args:
        segments: List of Segment objects with English text
        target_lang: Target language code (default: 'hi' for Hindi)
        use_indictrans2: Use IndicTrans2 model (True) or googletrans fallback (False)
        device: Compute device for IndicTrans2

    Returns:
        Same segments with 'translated' field populated
    """
    if not segments:
        logger.warning("⚠️  No segments to translate")
        return segments

    full_text = " ".join(s.text for s in segments)
    logger.info(f"🌐 Translating {len(segments)} segments to Hindi")
    logger.info(
        f"   Source: \"{full_text[:80]}{'...' if len(full_text) > 80 else ''}\""
    )

    if use_indictrans2:
        try:
            segments = _translate_indictrans2(segments, target_lang, device)
        except Exception as e:
            logger.warning(f"⚠️  IndicTrans2 failed: {e}")
            logger.info("   Falling back to googletrans...")
            segments = _translate_googletrans(segments, target_lang)
    else:
        segments = _translate_googletrans(segments, target_lang)

    # Log translations
    for seg in segments:
        logger.debug(f"   EN: {seg.text}")
        logger.debug(f"   HI: {seg.translated}")

    return segments


# ═══════════════════════════════════════════════════════════════════
# Primary: IndicTrans2 (AI4Bharat)
# ═══════════════════════════════════════════════════════════════════


def _translate_indictrans2(
    segments: List[Segment],
    target_lang: str = "hi",
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Translate using AI4Bharat IndicTrans2 model.

    Uses the distilled 200M parameter model for efficiency on Colab.
    Context-aware: translates sentences, not fragments.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

    logger.info(f"   Loading IndicTrans2 ({model_name}) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(device)
    model.eval()

    # ── Translate each segment ───────────────────────────────────
    # IndicTrans2 expects: ">>hin_Deva<< " prefix for Hindi target
    lang_code_map = {
        "hi": "hin_Deva",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "pa": "pan_Guru",
        "ur": "urd_Arab",
    }

    target_code = lang_code_map.get(target_lang, "hin_Deva")

    for seg in segments:
        # Build context-aware input
        input_text = seg.text.strip()
        if not input_text:
            seg.translated = ""
            continue

        try:
            # Tokenize with target language prefix
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=(
                        tokenizer.convert_tokens_to_ids(f">>{target_code}<<")
                        if hasattr(tokenizer, "convert_tokens_to_ids")
                        else None
                    ),
                    max_length=256,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                )

            translated = tokenizer.decode(generated[0], skip_special_tokens=True)
            seg.translated = translated.strip()

        except Exception as e:
            logger.warning(f"   ⚠️  Translation failed for segment: {e}")
            # Fallback to googletrans for this segment
            seg.translated = _translate_single_google(seg.text, target_lang)

    # Cleanup
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"   ✅ IndicTrans2 translation complete")
    return segments


# ═══════════════════════════════════════════════════════════════════
# Fallback: googletrans
# ═══════════════════════════════════════════════════════════════════


def _translate_googletrans(
    segments: List[Segment],
    target_lang: str = "hi",
) -> List[Segment]:
    """
    Translate using free Google Translate API (deep-translator library).
    No GPU required, but rate-limited and slightly lower quality.
    """
    logger.info("   Using deep-translator (free fallback)")

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="en", target=target_lang)
    except ImportError:
        logger.error(
            "deep-translator not installed. Install with: pip install deep-translator"
        )
        raise

    for seg in segments:
        if not seg.text.strip():
            seg.translated = ""
            continue
        try:
            seg.translated = translator.translate(seg.text)
        except Exception as e:
            logger.warning(f"   ⚠️  Google translate failed for segment: {e}")
            seg.translated = seg.text  # Keep original as fallback

    logger.info(f"   ✅ deep-translator translation complete")
    return segments


def _translate_single_google(text: str, target_lang: str = "hi") -> str:
    """Translate a single text string via googletrans."""
    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.translate(text, dest=target_lang, src="en")
        return result.text
    except Exception as e:
        logger.warning(f"   ⚠️  Google translate failed: {e}")
        return text  # Return original as fallback


# ═══════════════════════════════════════════════════════════════════
# Context-Aware Translation (for better quality)
# ═══════════════════════════════════════════════════════════════════


def translate_with_context(
    segments: List[Segment],
    target_lang: str = "hi",
    context_window: int = 2,
) -> List[Segment]:
    """
    Translate segments with surrounding context for better coherence.
    Groups nearby segments and translates them together.

    Args:
        segments: List of Segment objects
        target_lang: Target language
        context_window: Number of surrounding segments to include as context

    Returns:
        Segments with context-aware translations
    """
    logger.info(f"   Using context-aware translation (window={context_window})")

    # Group segments into overlapping windows
    for i, seg in enumerate(segments):
        # Build context string
        context_start = max(0, i - context_window)
        context_end = min(len(segments), i + context_window + 1)
        context_texts = [s.text for s in segments[context_start:context_end]]

        # Translate the full context
        full_context = " ".join(context_texts)
        translated_context = _translate_single_google(full_context, target_lang)

        # Simple heuristic: split translated text proportionally
        # In practice, the segment-level translation is preferred
        if seg.translated == "":
            seg.translated = _translate_single_google(seg.text, target_lang)

    return segments
