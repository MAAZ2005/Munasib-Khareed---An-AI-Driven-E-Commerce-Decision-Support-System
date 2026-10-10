"""Asset downloader script for spaCy language models and NLTK corpora."""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("munasib.assets")

SPACY_MODEL = "en_core_web_sm"

NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
]


def download_spacy_model(model_name: str = SPACY_MODEL) -> None:
    """Download the specified spaCy model if not already available.

    Args:
        model_name: Name of the spaCy pipeline package to download.
    """
    logger.info("Checking spaCy model: %s", model_name)
    try:
        import spacy
        spacy.load(model_name)
        logger.info("spaCy model '%s' is already installed and loadable.", model_name)
    except Exception as err:
        logger.info("Downloading spaCy model '%s'...", model_name)
        cmd = [sys.executable, "-m", "spacy", "download", model_name]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info("Successfully downloaded '%s'.", model_name)
        else:
            logger.error("Failed to download '%s': %s", model_name, result.stderr)
            raise RuntimeError(f"Could not install spaCy model {model_name}") from err


def download_nltk_corpora() -> None:
    """Download required NLTK tokenizers, stopwords, and linguistic corpora."""
    logger.info("Checking and downloading NLTK corpora...")
    try:
        import nltk

        for pkg in NLTK_PACKAGES:
            logger.info("Downloading NLTK package: %s", pkg)
            nltk.download(pkg, quiet=True)
        logger.info("All NLTK packages verified successfully.")
    except ImportError:
        logger.warning("NLTK is not yet installed in the current environment.")


def main() -> None:
    """Execute asset downloads for NLP pipelines."""
    logger.info("Starting asset preparation for Munasib Khareed NLP pipelines...")
    download_nltk_corpora()
    download_spacy_model()
    logger.info("All NLP assets are ready.")


if __name__ == "__main__":
    main()
