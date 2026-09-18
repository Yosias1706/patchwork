from rag_document_search.chunking import TextSplitter


def test_splitter_preserves_content_and_offsets() -> None:
    text = "First paragraph has useful content.\n\nSecond paragraph has more useful content."
    chunks = TextSplitter(chunk_size=44, chunk_overlap=8).split("doc-1", text)

    assert len(chunks) >= 2
    assert all(text[chunk.start_offset : chunk.end_offset] == chunk.text for chunk in chunks)
    assert chunks[1].start_offset < chunks[0].end_offset
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_splitter_ignores_whitespace_only_documents() -> None:
    assert TextSplitter(chunk_size=20, chunk_overlap=5).split("doc-1", " \n\t ") == []
