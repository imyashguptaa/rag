from docqa import ingest, store
from docqa.config import settings
from docqa.retrieve import retrieve

from .conftest import make_pdf

SOLAR = "Solar panels convert sunlight into electricity using photovoltaic cells made of silicon."
WHALE = "The blue whale is the largest animal on Earth and feeds mostly on tiny krill."
TEA = "Darjeeling tea is grown in the foothills of the Himalayas and harvested in several flushes."
VOLCANO = "Mount Etna is an active stratovolcano on the east coast of Sicily in Italy."


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(250)]
    chunks = ingest.chunk_words(" ".join(words), size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]  # shared overlap
    assert chunks[-1].split()[-1] == "w249"  # nothing dropped at the end


def test_short_page_is_one_chunk():
    assert ingest.chunk_words("just a few words", size=100, overlap=20) == ["just a few words"]
    assert ingest.chunk_words("", size=100, overlap=20) == []


def test_chunks_store_source_page_and_date(tmp_index):
    make_pdf(tmp_index / "energy.pdf", [SOLAR, WHALE])
    ingest.sync()
    hits, _ = retrieve("Which animal eats krill?", mode="dense")
    assert (hits[0].source, hits[0].page) == ("energy.pdf", 2)
    assert len(hits[0].doc_date) == 10  # YYYY-MM-DD


def test_update_and_delete_touch_only_the_changed_document(tmp_index, monkeypatch):
    make_pdf(tmp_index / "a.pdf", [SOLAR])
    make_pdf(tmp_index / "b.pdf", [WHALE])
    make_pdf(tmp_index / "c.pdf", [TEA])
    assert sorted(ingest.sync()["added"]) == ["a.pdf", "b.pdf", "c.pdf"]

    embedded = []
    real_add = ingest.add_document
    monkeypatch.setattr(ingest, "add_document", lambda path, sha, pages: embedded.append(path.name) or real_add(path, sha, pages))

    # Update: b.pdf gets new content. Only b is re-embedded.
    make_pdf(tmp_index / "b.pdf", [VOLCANO])
    report = ingest.sync()
    assert report["updated"] == ["b.pdf"] and sorted(report["unchanged"]) == ["a.pdf", "c.pdf"]
    assert embedded == ["b.pdf"]
    texts = [h.text for h in retrieve("blue whale krill", mode="hybrid", top_k=10)[0]]
    assert not any("whale" in t for t in texts)  # the old version is gone
    assert retrieve("Where is Mount Etna?", mode="hybrid")[0][0].source == "b.pdf"

    # Delete: c.pdf is removed from the folder. Nothing is re-embedded.
    (tmp_index / "c.pdf").unlink()
    report = ingest.sync()
    assert report["deleted"] == ["c.pdf"] and embedded == ["b.pdf"]
    assert all(h.source != "c.pdf" for h in retrieve("Darjeeling tea", mode="hybrid", top_k=10)[0])
    assert sorted(store.get_documents(settings.collection)) == ["a.pdf", "b.pdf"]


def test_index_version_changes_on_every_index_change(tmp_index):
    make_pdf(tmp_index / "a.pdf", [SOLAR])
    ingest.sync()
    v1 = store.index_version()
    ingest.sync()
    assert store.index_version() == v1  # nothing changed
    ingest.delete_document("a.pdf")
    assert store.index_version() > v1
