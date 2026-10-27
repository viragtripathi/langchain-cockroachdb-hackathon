"""Load the sample documents in data/<company>/ as LangChain Documents."""

from __future__ import annotations

import uuid

from langchain_core.documents import Document

from kit.common import ROOT


def load_docs(company: str) -> list[Document]:
    """Read every data/<company>/*.md file.

    Each file starts with "title:" and "category:" lines, then a blank line,
    then the text. The title, category and file name become metadata, which
    you can filter on.
    """
    docs = []
    for path in sorted((ROOT / "data" / company).glob("*.md")):
        header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        metadata = {"source": path.name, "company": company}
        for line in header.splitlines():
            key, _, value = line.partition(":")
            if key.strip():
                metadata[key.strip()] = value.strip()
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{path.name}"))
        docs.append(Document(page_content=body.strip(), metadata=metadata, id=doc_id))
    return docs
