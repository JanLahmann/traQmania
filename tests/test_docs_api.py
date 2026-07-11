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
    # in a source checkout the README (traQmania) leads; every entry carries
    # its curated menu title, and the parked TM2020 concept stays off the menu
    if not discover_docs():
        assert docs == []  # bare install: feature reports empty, UI hides it
        return
    assert ids == ["README", "EXHIBITION", "SCIENCE", "ARCHITECTURE"]
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
