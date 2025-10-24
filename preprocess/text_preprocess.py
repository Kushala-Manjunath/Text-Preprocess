"""Utility functions for simple text preprocessing.

These helpers avoid heavyweight model downloads and work with standard
Python/third‑party libs available in requirements.
"""

from __future__ import annotations

import re
from typing import List, Optional, Collection, Tuple, Dict
import json
import os
import unicodedata
import nltk

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    BeautifulSoup = None  # type: ignore


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_DIGIT_RE = re.compile(r"\d")
_EMAIL_RE = re.compile(
    r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
_RETWEET_RE = re.compile(r"\bRT\b\s+@[_A-Za-z0-9]+:?", re.IGNORECASE)

# Optional: try to pull NLTK stopwords if available locally without downloads
try:  # pragma: no cover - optional dependency at runtime
    from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore
    try:
        _NLTK_STOPWORDS_EN = set(_nltk_stopwords.words("english"))  # type: ignore
    except Exception:
        _NLTK_STOPWORDS_EN = None  # type: ignore
except Exception:  # pragma: no cover
    _NLTK_STOPWORDS_EN = None  # type: ignore

# Optional: TextBlob (spelling, noun phrases, sentiment, translation helpers)
try:  # pragma: no cover - optional dependency at runtime
    from textblob import TextBlob, Word  # type: ignore
    try:
        from textblob.sentiments import NaiveBayesAnalyzer  # type: ignore
    except Exception:
        NaiveBayesAnalyzer = None  # type: ignore
except Exception:  # pragma: no cover
    TextBlob = None  # type: ignore
    Word = None  # type: ignore
    NaiveBayesAnalyzer = None  # type: ignore

# Optional: spaCy for noun chunks (requires installed model like 'en_core_web_sm')
try:  # pragma: no cover - optional dependency at runtime
    import spacy as _spacy  # type: ignore
except Exception:  # pragma: no cover
    _spacy = None  # type: ignore


def download_nltk_packages(
    packages: Optional[List[str]] = None,
    *,
    quiet: bool = True,
) -> Dict[str, bool]:
    """Download common NLTK resources used by downstream users.

    By default, downloads these resources:
    - stopwords
    - averaged_perceptron_tagger (handles common misspelling 'averaged_perception_agger')
    - tagsets
    - wordnet
    - maxent_ne_chunker
    - punkt

    Parameters
    - packages: optional explicit list of resource IDs to download.
    - quiet: pass through to nltk.download for reduced console output.

    Returns a dict mapping resource id -> True if download succeeded or was
    already available, False otherwise.
    """
    default_packages = [
        "stopwords",
        "averaged_perceptron_tagger",
        "tagsets",
        "wordnet",
        "maxent_ne_chunker",
        "punkt",
    ]

    # Map common typos/aliases to correct NLTK resource IDs
    alias_map = {
        "averaged_perception_agger": "averaged_perceptron_tagger",
    }

    to_download = packages or default_packages
    resolved: List[str] = []
    for p in to_download:
        resolved.append(alias_map.get(p, p))

    results: Dict[str, bool] = {}
    for rid in resolved:
        try:
            # nltk.download returns True if successful or already present
            ok = bool(nltk.download(rid, quiet=quiet))
        except Exception:
            ok = False
        results[rid] = ok
    return results

# Lightweight fallback stopword list (subset of common English stopwords)
DEFAULT_STOPWORDS_EN = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
    "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})


# Lazy-load contractions/abbreviations map from bundled JSON
_CONTRACTIONS_MAP: Optional[Dict[str, str]] = None

def _load_contractions_map() -> Dict[str, str]:
    global _CONTRACTIONS_MAP
    if _CONTRACTIONS_MAP is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "contractions.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalize keys to lowercase for case-insensitive lookups
                _CONTRACTIONS_MAP = {str(k): str(v) for k, v in data.items()}
        except Exception:
            _CONTRACTIONS_MAP = {}
    return _CONTRACTIONS_MAP


def strip_html(x: str) -> str:
    """Remove HTML tags, preserving visible text.

    Falls back to a regex-based strip if BeautifulSoup is unavailable.
    """
    if not x:
        return ""
    if BeautifulSoup is None:
        # Minimal fallback: drop tags with a regex (imperfect but safe)
        return re.sub(r"<[^>]+>", " ", x)
    soup = BeautifulSoup(x, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def to_lowercase(x: str) -> str:
    """Lowercase the input text (safe for None/empty)."""
    return (x or "").lower()


def normalize_whitespace(x: str) -> str:
    """Collapse consecutive whitespace to single spaces and trim."""
    return _WHITESPACE_RE.sub(" ", x or "").strip()


def remove_urls(x: str) -> str:
    """Remove URLs from text."""
    return _URL_RE.sub(" ", x or "")


def count_urls(x: str) -> int:
    """Count URLs in the input text."""
    return len(_URL_RE.findall(x or ""))


def simple_tokenize(x: str) -> List[str]:
    """Simple whitespace tokenizer after basic cleanup.

    This intentionally avoids NLTK data downloads to keep usage lightweight.
    """
    cleaned = normalize_whitespace(x or "")
    return cleaned.split(" ") if cleaned else []


def clean_text(x: str) -> str:
    """Pipeline: strip HTML -> remove URLs -> lowercase -> normalize whitespace."""
    return normalize_whitespace(to_lowercase(remove_urls(strip_html(x))))


def word_count(x: str) -> int:
    """Count words in text using simple whitespace tokenization.

    Applies `normalize_whitespace` implicitly via `simple_tokenize` and counts
    the resulting tokens. URLs are not removed and punctuation is preserved
    within tokens; for stricter counting, call `clean_text` first.
    """
    return len(simple_tokenize(x))


def character_count(x: str) -> int:
    """Count total characters in the input (including spaces and punctuation)."""
    return len(x or "")


def average_word_length(x: str) -> float:
    """Average token length: total characters of tokens divided by word count.

    Uses `simple_tokenize` (whitespace-based). Punctuation inside tokens is
    included; URLs are not removed. Returns 0.0 for empty input.
    """
    tokens = simple_tokenize(x)
    if not tokens:
        return 0.0
    total_chars = sum(len(t) for t in tokens)
    return total_chars / len(tokens)


def to_ascii(x: str) -> str:
    """Convert accented characters to closest ASCII equivalents.

    Uses Unicode NFKD normalization and drops non-ASCII marks.
    """
    if not x:
        return ""
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")


def remove_special_characters(x: str) -> str:
    """Remove punctuation/special characters, keeping letters, digits, and spaces.

    Collapses multiple spaces afterward.
    """
    s = x or ""
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    return normalize_whitespace(s)


def squeeze_repeated_chars(x: str, max_repeats: int = 2) -> str:
    """Reduce character runs to at most `max_repeats` (default 2).

    Example: "loooove!!!" -> "loove!!"
    """
    if max_repeats < 1:
        max_repeats = 1
    # Replace any run of 3+ of the same char with exactly `max_repeats`.
    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        return ch * max_repeats

    return re.sub(r"(.)\1{2,}", repl, x or "")


def expand_contractions(
    x: str,
    *,
    mapping: Optional[Dict[str, str]] = None,
) -> str:
    """Expand contractions/abbreviations based on `contractions.json`.

    The bundled JSON contains many social/colloquial forms (e.g., "idk" ->
    "i do not know", "b&b" -> "bed and breakfast"). Matching is case-insensitive
    on token cores (leading/trailing punctuation is preserved).
    """
    s = x or ""
    if not s:
        return s
    mp = mapping or _load_contractions_map()
    if not mp:
        return s

    # Split on whitespace and expand token-by-token, preserving edge punctuation.
    tokens = simple_tokenize(s)
    if not tokens:
        return s

    PUNCT = "\"'“”‘’`.,!?;:()[]{}<>…"

    def expand_token(tok: str) -> str:
        if not tok:
            return tok
        start = 0
        end = len(tok)
        while start < end and tok[start] in PUNCT:
            start += 1
        while end > start and tok[end - 1] in PUNCT:
            end -= 1
        core = tok[start:end]
        if not core:
            return tok
        repl = mp.get(core) or mp.get(core.lower())
        if repl:
            return tok[:start] + repl + tok[end:]
        return tok

    return normalize_whitespace(" ".join(expand_token(t) for t in tokens))


def stopwords_in_text(
    x: str,
    *,
    stopwords: Optional[Collection[str]] = None,
) -> List[str]:
    """Return a list of stopword tokens found in `text` (duplicates kept).

    Matching is case-insensitive via lowercasing. Provide a custom `stopwords`
    collection to override the default. When available, NLTK's English
    stopwords are used; otherwise a small built-in list is applied.
    """
    sw = set(stopwords or _NLTK_STOPWORDS_EN or DEFAULT_STOPWORDS_EN)
    return [t for t in simple_tokenize(to_lowercase(x or "")) if t in sw]


def stopword_count(
    x: str,
    *,
    stopwords: Optional[Collection[str]] = None,
) -> int:
    """Count stopword tokens present in the text."""
    return len(stopwords_in_text(x, stopwords=stopwords))


def remove_stopwords(
    x: str,
    *,
    stopwords: Optional[Collection[str]] = None,
) -> str:
    """Remove stopword tokens (case-insensitive) and return cleaned text."""
    sw = set(stopwords or _NLTK_STOPWORDS_EN or DEFAULT_STOPWORDS_EN)
    tokens = simple_tokenize(x)
    kept = [t for t in tokens if t.lower() not in sw]
    return " ".join(kept)


def count_hashtags_and_mentions(x: str) -> Tuple[int, int]:
    """Return counts of hashtags and mentions as a tuple (hashtags, mentions)."""
    s = x or ""
    return (len(_HASHTAG_RE.findall(s)), len(_MENTION_RE.findall(s)))


def count_emails(x: str) -> int:
    """Count email addresses in the input text."""
    return len(_EMAIL_RE.findall(x or ""))


def remove_emails(x: str) -> str:
    """Remove email addresses from the text."""
    return _EMAIL_RE.sub(" ", x or "")


def count_and_remove_emails(x: str) -> Tuple[str, int]:
    """Return (cleaned_text, count) after removing emails."""
    cnt = count_emails(x)
    return (remove_emails(x), cnt)


def count_and_remove_urls(x: str) -> Tuple[str, int]:
    """Return (cleaned_text, count) after removing URLs."""
    cnt = count_urls(x)
    return (remove_urls(x), cnt)


def has_digits(x: str) -> bool:
    """Return True if any numeric digit is present in the text."""
    return bool(_DIGIT_RE.search(x or ""))


def uppercase_word_count(x: str) -> int:
    """Count tokens that are all-uppercase (with at least one letter).

    Uses `simple_tokenize`. Tokens like "NASA" count; tokens without letters
    or with mixed case do not. Trailing punctuation may prevent a match.
    """
    tokens = simple_tokenize(x)
    return sum(1 for t in tokens if re.search(r"[A-Za-z]", t) and t == t.upper())


def remove_retweets(x: str) -> str:
    """Remove Twitter-style retweet markers like 'RT @user:' from text."""
    return normalize_whitespace(_RETWEET_RE.sub(" ", x or ""))


def remove_html_tags(x: str) -> str:
    """Alias for `strip_html`: remove HTML tags and return visible text."""
    return strip_html(x)


def to_base_form(x: str) -> str:
    """Convert tokens to a base/root form.

    Tries NLTK WordNet lemmatizer if available; falls back to Porter stemming.
    If neither is available, returns the input normalized whitespace.
    """
    tokens = simple_tokenize(x)
    if not tokens:
        return ""

    # Try lemmatization first (may fail if wordnet data missing)
    try:  # pragma: no cover - optional runtime path
        from nltk.stem import WordNetLemmatizer  # type: ignore
        lemm = WordNetLemmatizer()

        def lemma_token(t: str) -> str:
            try:
                # Try verb then noun to handle common verb forms without POS tagger
                t1 = lemm.lemmatize(t, pos="v")
                t2 = lemm.lemmatize(t1, pos="n")
                return t2
            except Exception:
                return t

        base = [lemma_token(t) for t in tokens]
        return " ".join(base)
    except Exception:
        pass

    # Fallback: Porter stemming (no corpus download required)
    try:  # pragma: no cover - optional runtime path
        from nltk.stem import PorterStemmer  # type: ignore
        stemmer = PorterStemmer()
        return " ".join(stemmer.stem(t) for t in tokens)
    except Exception:
        return " ".join(tokens)


def lemmatize(
    x: str,
    *,
    pos: Optional[str] = None,
) -> str:
    """Lemmatize tokens to their dictionary form.

    Attempts NLTK's WordNetLemmatizer if available. If `pos` is provided,
    it should be one of {"n","v","a","r"}. Without `pos`, a light heuristic
    tries verb then noun. Falls back to TextBlob's Word.lemmatize if NLTK
    lemmatizer or corpora are unavailable; otherwise returns the input tokens.
    """
    tokens = simple_tokenize(x)
    if not tokens:
        return ""

    # Preferred path: NLTK WordNetLemmatizer
    try:  # pragma: no cover - optional runtime path
        from nltk.stem import WordNetLemmatizer  # type: ignore
        lemm = WordNetLemmatizer()

        if pos in {"n", "v", "a", "r"}:
            return " ".join(lemm.lemmatize(t, pos=pos) for t in tokens)

        def lemma_token(t: str) -> str:
            try:
                t1 = lemm.lemmatize(t, pos="v")
                t2 = lemm.lemmatize(t1, pos="n")
                return t2
            except Exception:
                return t

        return " ".join(lemma_token(t) for t in tokens)
    except Exception:
        pass

    # Fallback: TextBlob's lemmatize if available
    if Word is not None:
        out: List[str] = []
        for t in tokens:
            try:
                if pos in {"n", "v", "a", "r"}:
                    out.append(str(Word(t).lemmatize(pos=pos)))  # type: ignore[arg-type]
                else:
                    # try verb then noun
                    t1 = str(Word(t).lemmatize(pos="v"))
                    t2 = str(Word(t1).lemmatize(pos="n"))
                    out.append(t2)
            except Exception:
                out.append(t)
        return " ".join(out)

    return " ".join(tokens)


def remove_common_words(
    x: str,
    *,
    top_n: int = 10,
    freq: Optional[Dict[str, int]] = None,
) -> str:
    """Remove the `top_n` most frequent tokens.

    If `freq` mapping is provided, uses it; otherwise computes frequencies from `x`.
    Case-insensitive matching on token lowercase.
    """
    tokens = simple_tokenize(x)
    if not tokens or top_n <= 0:
        return normalize_whitespace(" ".join(tokens))
    if freq is None:
        counts: Dict[str, int] = {}
        for t in tokens:
            k = t.lower()
            counts[k] = counts.get(k, 0) + 1
    else:
        counts = {k.lower(): int(v) for k, v in freq.items()}
    # Determine top_n by frequency (stable tie-break by token)
    top = {k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]}
    kept = [t for t in tokens if t.lower() not in top]
    return " ".join(kept)


def remove_rare_words(
    x: str,
    *,
    min_freq: int = 1,
    freq: Optional[Dict[str, int]] = None,
) -> str:
    """Remove tokens whose frequency is <= `min_freq`.

    If `freq` mapping is provided, uses it; otherwise computes from `x`.
    Case-insensitive on token lowercase.
    """
    tokens = simple_tokenize(x)
    if not tokens:
        return ""
    if freq is None:
        counts: Dict[str, int] = {}
        for t in tokens:
            k = t.lower()
            counts[k] = counts.get(k, 0) + 1
    else:
        counts = {k.lower(): int(v) for k, v in freq.items()}
    kept = [t for t in tokens if counts.get(t.lower(), 0) > min_freq]
    return " ".join(kept)


def correct_spelling(x: str) -> str:
    """Spelling correction using TextBlob if available; returns input on failure.

    Warning: This can be slow and may require corpora. Use sparingly.
    """
    s = x or ""
    if not s or TextBlob is None:
        return s
    try:
        return str(TextBlob(s).correct())
    except Exception:
        return s


def noun_chunks_and_phrases(
    x: str,
    *,
    spacy_model: Optional[str] = None,
) -> List[str]:
    """Extract noun chunks/phrases using spaCy or TextBlob as fallback.

    - If `spacy_model` is provided and available (e.g., 'en_core_web_sm'), use it.
    - Else, try TextBlob.noun_phrases.
    - Else, return an empty list.
    """
    s = x or ""
    if not s:
        return []
    # Try spaCy with a provided or default model
    if _spacy is not None:
        model_name = spacy_model or os.environ.get("SPACY_MODEL", "en_core_web_sm")
        try:
            nlp = _spacy.load(model_name)  # requires installed model
            doc = nlp(s)
            return [np.text for np in getattr(doc, "noun_chunks", [])]
        except Exception:
            pass
    # Fallback to TextBlob noun phrases
    if TextBlob is not None:
        try:
            return list(TextBlob(s).noun_phrases)
        except Exception:
            return []
    return []


def ngrams(
    x: str,
    *,
    n: int = 2,
) -> List[Tuple[str, ...]]:
    """Return list of token n-grams (tuples) of size `n` (default 2)."""
    tokens = simple_tokenize(x)
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def bigrams(x: str) -> List[Tuple[str, str]]:
    """Return list of token bigrams (n=2)."""
    return [(a, b) for (a, b) in ngrams(x, n=2)]  # type: ignore[misc]


def singularize_words(x: str) -> str:
    """Convert tokens to singular form using TextBlob if available."""
    tokens = simple_tokenize(x)
    if not tokens:
        return ""
    if Word is None:
        # naive fallback: remove trailing 's' for simple plurals
        return " ".join(t[:-1] if t.lower().endswith("s") and len(t) > 1 else t for t in tokens)
    out: List[str] = []
    for t in tokens:
        try:
            out.append(str(Word(t).singularize()))
        except Exception:
            out.append(t)
    return " ".join(out)


def pluralize_words(x: str) -> str:
    """Convert tokens to plural form using TextBlob if available."""
    tokens = simple_tokenize(x)
    if not tokens:
        return ""
    if Word is None:
        # naive fallback: append 's' for simple nouns
        return " ".join(t + "s" if t and t[-1].isalpha() else t for t in tokens)
    out: List[str] = []
    for t in tokens:
        try:
            out.append(str(Word(t).pluralize()))
        except Exception:
            out.append(t)
    return " ".join(out)


def naive_bayes_sentiment(x: str) -> Optional[Dict[str, float]]:
    """Return Naive Bayes sentiment {'p_pos','p_neg','classification'} if available.

    Uses TextBlob's NaiveBayesAnalyzer which requires NLTK movie_reviews corpus.
    Returns None if analyzer is unavailable.
    """
    s = x or ""
    if not s or TextBlob is None or NaiveBayesAnalyzer is None:
        return None
    try:
        blob = TextBlob(s, analyzer=NaiveBayesAnalyzer())
        res = blob.sentiment  # type: ignore[attr-defined]
        return {
            "p_pos": float(getattr(res, "p_pos", 0.0)),
            "p_neg": float(getattr(res, "p_neg", 0.0)),
            "classification": str(getattr(res, "classification", "neutral")),
        }
    except Exception:
        return None


def detect_language(x: str) -> Optional[str]:
    """Detect language code using TextBlob if available/networked; else None."""
    s = x or ""
    if not s or TextBlob is None:
        return None
    try:
        # This may need network access; failures are handled.
        return TextBlob(s).detect_language()  # type: ignore[no-any-return]
    except Exception:
        return None


def translate_text(
    x: str,
    *,
    to: str = "en",
    source_lang: Optional[str] = None,
) -> str:
    """Translate text using TextBlob if available/networked; else return input.

    Set `source_lang` to bypass auto-detection when known.
    """
    s = x or ""
    if not s or TextBlob is None:
        return s
    try:
        blob = TextBlob(s)
        if source_lang:
            return str(blob.translate(from_lang=source_lang, to=to))
        return str(blob.translate(to=to))
    except Exception:
        return s
