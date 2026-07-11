"""The /api/docs endpoints that back the in-UI documentation browser."""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from traqmania.config import load_config  # noqa: E402
from traqmania.server.app import create_app, discover_docs  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return fastapi_testclient.TestClient(create_app(load_config()))


def test_docs_index_lists_repo_docs(client):
    docs = client.get("/api/docs").json()["docs"]
    ids = [d["id"] for d in docs]
    # in a source checkout the QML story (SCIENCE) leads; every entry has a
    # human title parsed from the first heading
    if not discover_docs():
        assert docs == []  # bare install: feature reports empty, UI hides it
        return
    assert ids[0] == "SCIENCE"
    assert "README" in ids
    assert all(d["title"] for d in docs)


def test_doc_content_and_404(client):
    if not discover_docs():
        pytest.skip("no repo docs next to the package")
    doc = client.get("/api/docs/SCIENCE").json()
    assert doc["id"] == "SCIENCE"
    assert doc["markdown"].lstrip().startswith("#")
    assert client.get("/api/docs/nope").status_code == 404
    # path traversal never reaches the filesystem: ids are whitelist-matched
    assert client.get("/api/docs/..%2FREADME").status_code == 404
