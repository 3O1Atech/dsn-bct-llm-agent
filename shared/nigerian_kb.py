import random
import re
from typing import Dict

PIDGIN_PHRASES = {
    "positive": ["sweet die", "make sense", "correct guy", "na baba", "e get as e be", "chop life", "gbas gbos", "no be small thing", "e too choke"],
    "negative": ["wahala", "nonsense", "no dey work", "yeye", "shege", "buru", "na lie", "e no pure", "fail am"],
    "neutral": ["abeg", "how far", "omo", "sha", "sotey", "well well", "small small", "dey there", "no wahala"],
}

LOCAL_REFERENCES = {
    "food": ["jollof", "amala", "suya", "pepper soup", "bole", "tuwo", "miyan kuka", "afang", "egusi", "pounded yam"],
    "transport": ["danfo", "okada", "BRT", "keke", "molue", "uber", "bolt", "Lagos-Ibadan expressway", "Third Mainland Bridge"],
    "infra": ["NEPA", "PHCN", "MTN", "fuel scarcity", "generator", "power outage", "traffic", "go-slow"],
    "entertainment": ["Nollywood", "Burna Boy", "Wizkid", "BBNaija", "owambe", "Aso Ebi", "DJ Cuppy", "Davido"],
}

CITIES = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "Enugu", "Benin", "Calabar"]


def inject_nigerian_context(text: str, persona: Dict, item_category: str) -> str:
    register = persona.get("linguistic_register", "mixed")
    pidgin_ratio = persona.get("pidgin_ratio", 0.0)
    emotional_triggers = persona.get("emotional_triggers", [])

    # Determine how many injections (1-3)
    num_injections = random.randint(1, 3)

    if register == "formal":
        # Subtle local references only
        refs = _pick_refs(item_category, emotional_triggers, num_injections, pidgin=False)
        text = _blend_refs_formal(text, refs)
    elif register == "pidgin" or pidgin_ratio > 0.3:
        # Rewrite ~30-40% of sentences into pidgin
        text = _rewrite_pidgin(text, item_category, emotional_triggers)
        refs = _pick_refs(item_category, emotional_triggers, num_injections, pidgin=True)
        text = _blend_refs_pidgin(text, refs)
    elif register == "sarcastic":
        refs = _pick_refs(item_category, emotional_triggers, num_injections, pidgin=False)
        text = _blend_refs_sarcastic(text, refs)
    else:
        # Mixed: blend some pidgin phrases with local refs
        refs = _pick_refs(item_category, emotional_triggers, num_injections, pidgin=True)
        text = _blend_refs_mixed(text, refs)

    return text


def _pick_refs(item_category: str, emotional_triggers: list, n: int, pidgin: bool):
    candidates = []
    # Category-specific refs
    cat_key = item_category.lower()
    if cat_key in LOCAL_REFERENCES:
        candidates.extend(LOCAL_REFERENCES[cat_key])
    # Emotional triggers
    for trig in emotional_triggers:
        if trig == "NEPA" and "NEPA" not in candidates:
            candidates.append("NEPA")
        if trig == "traffic":
            candidates.extend(["traffic", "go-slow", "Third Mainland Bridge"])
        if trig == "fuel":
            candidates.extend(["fuel scarcity", "petrol queue"])
    # General refs
    candidates.extend(LOCAL_REFERENCES.get("food", []))
    candidates.extend(LOCAL_REFERENCES.get("transport", []))
    candidates.extend(LOCAL_REFERENCES.get("infra", []))
    candidates.extend(LOCAL_REFERENCES.get("entertainment", []))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    selected = random.sample(unique, min(n, len(unique)))

    if pidgin:
        # Add pidgin phrases
        sentiment = _detect_sentiment_from_text(" ".join(selected))
        phrases = PIDGIN_PHRASES.get(sentiment, PIDGIN_PHRASES["neutral"])
        extra = random.sample(phrases, min(2, len(phrases)))
        selected.extend(extra)

    return selected


def _detect_sentiment_from_text(text: str) -> str:
    negative_words = ["bad", "terrible", "worst", "hate", "disappointing", "poor", "awful", "nonsense", "wahala", "fail", "no dey"]
    positive_words = ["good", "great", "excellent", "love", "best", "amazing", "sweet", "correct", "make sense", "perfect"]
    lower = text.lower()
    neg = sum(1 for w in negative_words if w in lower)
    pos = sum(1 for w in positive_words if w in lower)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _blend_refs_formal(text: str, refs: list) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text
    for ref in refs:
        # Append subtly to a random sentence or prepend
        idx = random.randint(0, len(sentences) - 1)
        if ref in ["NEPA", "traffic", "Third Mainland Bridge", "Lagos-Ibadan expressway"]:
            # Blend as context
            sentences[idx] += f" — a familiar situation for anyone navigating {ref} in Nigeria."
        else:
            sentences[idx] += f", reminiscent of {ref}."
    return " ".join(sentences)


def _blend_refs_pidgin(text: str, refs: list) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text
    # Append pidgin phrases at end or weave into sentences
    for ref in refs:
        if ref in sum(PIDGIN_PHRASES.values(), []):
            text += f" {ref}."
        else:
            idx = random.randint(0, max(0, len(sentences) - 1))
            sentences[idx] += f" — {ref} matter no dey hide."
    return " ".join(sentences)


def _blend_refs_sarcastic(text: str, refs: list) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text
    for ref in refs:
        idx = random.randint(0, max(0, len(sentences) - 1))
        sentences[idx] += f" (because {ref} definitely makes everything better)."
    return " ".join(sentences)


def _blend_refs_mixed(text: str, refs: list) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text
    for ref in refs:
        idx = random.randint(0, max(0, len(sentences) - 1))
        if ref in sum(PIDGIN_PHRASES.values(), []):
            sentences[idx] += f", {ref}."
        else:
            sentences[idx] += f" — {ref} vibes strong here."
    return " ".join(sentences)


def _rewrite_pidgin(text: str, item_category: str, emotional_triggers: list) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text

    pidgin_templates = {
        "positive": [
            "This {item} sweet die!",
            "{item} make sense well well.",
            "Na correct {item} be this.",
            "I enjoy am, no be small thing.",
        ],
        "negative": [
            "This {item} get wahala.",
            "{item} no dey work at all.",
            "Na shege be this.",
            "The {item} fall hand.",
        ],
        "neutral": [
            "The {item} dey okay sha.",
            "{item} get as e be.",
            "I no sure yet but e dey there.",
            "Abeg, {item} no bad.",
        ],
    }

    num_to_rewrite = max(1, int(len(sentences) * 0.35))
    indices = random.sample(range(len(sentences)), min(num_to_rewrite, len(sentences)))

    for idx in indices:
        sent = sentences[idx].lower()
        sentiment = _detect_sentiment_from_text(sent)
        templates = pidgin_templates.get(sentiment, pidgin_templates["neutral"])
        template = random.choice(templates)
        # Try to extract noun/subject
        item_name = item_category if item_category else "thing"
        rewritten = template.format(item=item_name)
        sentences[idx] = rewritten

    return " ".join(sentences)
