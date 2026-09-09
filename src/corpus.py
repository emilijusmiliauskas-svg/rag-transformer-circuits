"""
Corpus loading.

The papers are saved as single-file HTML with inline base64 images and
webpack bundles. Reading them as plain text — which is what
`SimpleDirectoryReader` does — puts that binary into the index: the original
build produced 110,370 nodes for six papers, of which a sampled 97.7% were
base64 blobs rather than prose.

This module parses the HTML instead and keeps only the readable text, which
is 1.16% of the bytes on disk (51.3 MB -> ~596k characters, ~330 chunks).
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core.schema import Document

from src.config import DATA_PATH

# Tags that carry no prose. `head` goes too: it holds the loader bundles that
# dominated the original index.
NON_CONTENT_TAGS = ("script", "style", "noscript", "svg", "iframe", "head")

# Readable titles for the six files, which otherwise report their filename.
PAPER_TITLES = {
    "framework": "A Mathematical Framework for Transformer Circuits",
    "induction_heads": "In-Context Learning and Induction Heads",
    "interpretability_dreams": "Interpretability Dreams",
    "mech_interp_essay": "Mechanistic Interpretability Essay",
    "monosemanticity": "Towards Monosemanticity",
    "scaling_monosemanticity": "Scaling Monosemanticity",
}


def extract_text(html: str) -> str:
    """Strip markup, scripts and data URIs; return normalised prose."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(list(NON_CONTENT_TAGS)):
        tag.decompose()

    text = soup.get_text(" ", strip=True)

    # Any data: URI that survived as visible text (alt attributes, stray
    # inline content) is binary, not prose.
    text = re.sub(r"data:[a-zA-Z0-9/.+-]+;base64,[A-Za-z0-9+/=\s]+", " ", text)
    # Long unbroken base64-ish runs with no spaces are never real sentences.
    text = re.sub(r"\b[A-Za-z0-9+/]{200,}={0,2}\b", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def load_documents(data_path: Path | None = None) -> list[Document]:
    """Load the corpus as one Document per paper, markup removed."""
    directory = Path(data_path or DATA_PATH)
    files = sorted(directory.glob("*.html"))

    if not files:
        raise ValueError(f"No .html files found in {directory}.")

    documents: list[Document] = []
    for path in files:
        text = extract_text(path.read_text(errors="ignore"))
        if not text:
            print(f"--- Skipping {path.name}: no text extracted ---")
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "title": PAPER_TITLES.get(path.stem, path.stem),
                },
            )
        )
        print(
            f"--- {path.name}: {path.stat().st_size / 1e6:.1f} MB on disk "
            f"-> {len(text) / 1000:.0f}k characters of text ---"
        )

    return documents
