"""Unit tests for text extraction."""

import pytest
from pathlib import Path
from app.services.ingestion.extractor import TextExtractor, ExtractionResult


@pytest.fixture
def extractor():
    return TextExtractor()


def test_txt_extraction(extractor, sample_txt_file):
    result = extractor.extract(sample_txt_file, "sample.txt")
    assert isinstance(result, ExtractionResult)
    assert result.file_type == "txt"
    assert result.total_pages >= 1
    assert len(result.pages) >= 1
    assert "artificial intelligence" in result.full_text.lower()


def test_txt_extraction_preserves_filename(extractor, sample_txt_file):
    result = extractor.extract(sample_txt_file, "my_document.txt")
    assert result.filename == "my_document.txt"
    for page in result.pages:
        assert page.source == "my_document.txt"


def test_clean_text_removes_extra_whitespace(extractor):
    raw = "  Hello   world  \n\n\n\n  This is  a test  "
    cleaned = extractor._clean_text(raw)
    assert "  " not in cleaned
    assert cleaned == "Hello world\n\nThis is a test"


def test_unsupported_file_type(extractor, tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extractor.extract(bad, "file.xyz")


def test_docx_extraction(extractor, tmp_path):
    from docx import Document as DocxDoc
    docx_path = tmp_path / "test.docx"
    doc = DocxDoc()
    doc.add_paragraph("This is a DOCX test paragraph about machine learning.")
    doc.add_paragraph("Neural networks are a key component of deep learning systems.")
    doc.save(str(docx_path))

    result = extractor.extract(docx_path, "test.docx")
    assert result.file_type == "docx"
    assert result.total_pages >= 1
    assert "machine learning" in result.full_text.lower()
