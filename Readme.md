# preprocess

Lightweight text-cleaning and analysis helpers for Python projects. The
`preprocess.text_preprocess` module keeps hard dependencies to a minimum while
exposing convenience wrappers for common Natural Language Processing (NLP)
tasks such as HTML stripping, tokenization, stopword handling, keyword
extraction, and optional enrichment via third-party libraries.

## Installation from GitHub
```bash
pip install git+https://github.com/Kushala-Manjunath/Text-Preprocess.git --upgrade --force-reinstall
```

To work from a local checkout instead, run:

```bash
pip install -e .
```

## Quickstart

```python
from preprocess import text_preprocess as tp

raw = """
RT @kushala: Excited about the new product launch!
Visit https://example.com for details.<br><br>
Email: contact@example.com
"""

clean = tp.clean_text(raw)
tokens = tp.simple_tokenize(clean)

stats = {
    "word_count": tp.word_count(raw),
    "urls_removed": tp.count_urls(raw),
    "emails_removed": tp.count_emails(raw),
    "uppercase_words": tp.uppercase_word_count(raw),
}

print(clean)
# rt @kushala: excited about the new product launch! visit for details email: contact@example.com

print(tokens[:5])
# ['rt', '@kushala:', 'excited', 'about', 'the']
print(stats)
```

## Feature Highlights

- **Cleaning helpers**: `strip_html`, `remove_urls`, `remove_emails`,
  `remove_retweets`, `normalize_whitespace`, `to_lowercase`, `to_ascii`.
- **Tokenization and counts**: `simple_tokenize`, `word_count`,
  `character_count`, `count_hashtags_and_mentions`, `count_urls`.
- **Normalization**: `remove_stopwords`, `squeeze_repeated_chars`,
  `expand_contractions`, `to_base_form`, `lemmatize`.
- **Frequency filtering**: `remove_frequent_words`, `remove_rare_words`,
  `ngrams`, `bigrams`.
- **Optional NLP extras**: `correct_spelling`, `noun_chunks_and_phrases`,
  `singularize_words`, `pluralize_words`, all of which gracefully degrade when
  optional dependencies are missing.

Browse `preprocess/text_preprocess.py` for docstrings and inline guidance on
each helper.

## Optional Dependencies & Data

Most functions work with the standard library only. Some features require
additional packages or corpora:

- **NLTK** (required library): install with `pip install nltk`. To fetch the
  corpora used by lemmatizers and stopwords, call:

  ```python
  from preprocess import text_preprocess as tp
  tp.download_nltk_packages()
  ```

  Pass a custom list (e.g., `tp.download_nltk_packages(["wordnet", "punkt"])`)
  to control downloads.

- **BeautifulSoup** (`beautifulsoup4`): improves HTML stripping.
- **TextBlob** (`textblob`): enables spelling correction, noun phrase extraction,
  and verb inflection utilities.
- **spaCy** (`spacy` plus a model such as `en_core_web_sm`): powers richer noun
  chunk extraction. Install via:

  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```

All optional imports are wrapped in try/except blocks; functions fall back to
safe, lightweight behavior when these dependencies are absent.

## Project Structure

- `preprocess/`: Python package containing `text_preprocess.py` and bundled
  assets (e.g., `contractions.json`).
- `tests/`: Recommended location for unit tests (add as needed).
- `Readme.md`: Project overview and usage documentation.

Package metadata (version, author info, etc.) should be set in `setup.py` or
`pyproject.toml` before publishing.

## Contributing & Support

Bug reports and enhancement requests are welcome. Feel free to open an issue or
submit a pull request. If you encounter missing corpora or optional dependency
errors, install the requirements listed above and re-run
`download_nltk_packages()` before retrying your workflow.
