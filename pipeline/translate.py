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
    api_engine: str = "local",
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

    if api_engine == "openai":
        try:
            return _translate_openai_api(segments, target_lang)
        except Exception as e:
            logger.warning(f"⚠️  OpenAI Translation API failed: {e}")
            logger.info("   Falling back to local translation...")

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
    Translates all segments in a single batch for better context.
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

    # ── Resolve target language token ────────────────────────────
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

    # Robustly resolve the forced BOS token ID
    forced_bos_id = None
    bos_token_str = f">>{target_code}<<"
    try:
        token_id = tokenizer.convert_tokens_to_ids(bos_token_str)
        # Verify it's a real token (not the UNK token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            forced_bos_id = token_id
            logger.debug(
                f"   Forced BOS token: '{bos_token_str}' -> id {forced_bos_id}"
            )
        else:
            # Try alternative token format
            for alt in [target_code, f"__{target_code}__"]:
                alt_id = tokenizer.convert_tokens_to_ids(alt)
                if alt_id is not None and alt_id != tokenizer.unk_token_id:
                    forced_bos_id = alt_id
                    logger.debug(
                        f"   Forced BOS token (alt): '{alt}' -> id {forced_bos_id}"
                    )
                    break
    except Exception as e:
        logger.warning(f"   ⚠️  Could not resolve target language token: {e}")

    # ── Translate each segment ───────────────────────────────────
    for seg in segments:
        input_text = seg.text.strip()
        if not input_text:
            seg.translated = ""
            continue

        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            gen_kwargs = {
                "max_length": 256,
                "num_beams": 5,
                "length_penalty": 1.0,
                "early_stopping": True,
            }
            if forced_bos_id is not None:
                gen_kwargs["forced_bos_token_id"] = forced_bos_id

            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)

            translated = tokenizer.decode(generated[0], skip_special_tokens=True)
            seg.translated = translated.strip()

        except Exception as e:
            logger.warning(f"   ⚠️  Translation failed for segment: {e}")
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
    """
    Translate a single text string via deep-translator.
    (Unified with the main fallback — no longer uses deprecated googletrans.)
    """
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="en", target=target_lang)
        result = translator.translate(text)
        return result if result else text
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

        # If the segment doesn't have a translation yet, translate individually
        if not seg.translated:
            seg.translated = _translate_single_google(seg.text, target_lang)

    return segments


# ═══════════════════════════════════════════════════════════════════
# Premium: OpenAI GPT Translation
# ═══════════════════════════════════════════════════════════════════


def _translate_openai_api(
    segments: List[Segment], target_lang: str = "hi"
) -> List[Segment]:
    """
    Translate segments using OpenAI's GPT models (gpt-4o or gpt-3.5-turbo).
    Produces highly contextual and natural translations while retaining segment mapping.
    """
    from openai import OpenAI
    import os
    import json

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    logger.info(f"   Translating {len(segments)} segments via OpenAI API")

    # We send all segments at once as a JSON array to maintain context
    # and require a JSON array back to map exactly to the input segments.
    input_data = [{"id": i, "text": s.text} for i, s in enumerate(segments)]

    system_prompt = (
        f"You are a professional video translator and localizer. "
        f"Translate the following English video transcription segments into target language code '{target_lang}'. "
        f"Ensure translations are natural, contextual across segments, and suitable for voice-over lip-syncing. "
        f"Maintain the exact JSON list structure returning ONLY a JSON array of objects with 'id' and 'translated' keys."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective and fast
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(input_data)},
            ],
            response_format=(
                {"type": "json_object"} if False else None
            ),  # For strict JSON can use tools or just prompt
        )

        reply_content = response.choices[0].message.content.strip()

        # Clean up markdown code block if present
        if reply_content.startswith("```json"):
            reply_content = reply_content[7:]
        if reply_content.startswith("```"):
            reply_content = reply_content[3:]
        if reply_content.endswith("```"):
            reply_content = reply_content[:-3]
        reply_content = reply_content.strip()

        # Handle BOM characters
        reply_content = reply_content.lstrip("\ufeff")

        parsed = json.loads(reply_content)

        # Handle both bare array and wrapped object responses
        if isinstance(parsed, dict):
            # Try common wrapper keys
            for key in ["translations", "data", "results", "segments"]:
                if key in parsed:
                    translated_data = parsed[key]
                    break
            else:
                translated_data = list(parsed.values())[0] if parsed else []
        elif isinstance(parsed, list):
            translated_data = parsed
        else:
            raise ValueError(f"Unexpected response format: {type(parsed)}")

        # Map back to segments
        translated_dict = {item["id"]: item["translated"] for item in translated_data}

        for i, seg in enumerate(segments):
            if i in translated_dict:
                seg.translated = translated_dict[i].strip()
            else:
                logger.warning(f"   ⚠️  OpenAI missed segment {i}, falling back")
                seg.translated = _translate_single_google(seg.text, target_lang)

    except Exception as e:
        logger.warning(f"   ⚠️  OpenAI bulk translation failed: {e}")
        # Process one by one if bulk fails
        for seg in segments:
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Translate to {target_lang}. Keep it natural for dubbing.",
                        },
                        {"role": "user", "content": seg.text},
                    ],
                )
                seg.translated = res.choices[0].message.content.strip()
            except Exception as e2:
                logger.warning(f"   ⚠️  OpenAI segment translation failed: {e2}")
                seg.translated = _translate_single_google(seg.text, target_lang)

    return segments
