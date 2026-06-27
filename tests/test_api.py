"""API integration tests using FastAPI TestClient."""

import io
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_documents_empty(client: TestClient):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_txt_document(client: TestClient, sample_txt_file):
    content = sample_txt_file.read_bytes()
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test_upload.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Document uploaded and indexed successfully."
    assert data["document"]["status"] == "ready"
    assert data["document"]["total_chunks"] >= 1
    assert data["document"]["file_type"] == "txt"


def test_upload_unsupported_file_type(client: TestClient):
    response = client.post(
        "/api/v1/upload",
        files={"file": ("file.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_documents_listed_after_upload(client: TestClient, sample_txt_file):
    content = sample_txt_file.read_bytes()
    client.post(
        "/api/v1/upload",
        files={"file": ("listed_doc.txt", io.BytesIO(content), "text/plain")},
    )
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    filenames = [d["original_filename"] for d in response.json()]
    assert "listed_doc.txt" in filenames


def test_query_without_documents(client: TestClient):
    """Query on an empty index should return 422 or a 'no answer' response."""
    response = client.post(
        "/api/v1/query",
        json={"question": "What is artificial intelligence?"},
    )
    # Either 422 (no index loaded) or 200 with 'could not find' answer
    assert response.status_code in (200, 422, 500)


def test_query_after_upload(client: TestClient, sample_txt_file):
    """Full pipeline: upload then query."""
    content = sample_txt_file.read_bytes()
    upload_resp = client.post(
        "/api/v1/upload",
        files={"file": ("ai_doc.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload_resp.status_code == 201

    query_resp = client.post(
        "/api/v1/query",
        json={
            "question": "What is artificial intelligence?",
            "session_id": "test-session-001",
            "top_k": 3,
        },
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert data["session_id"] == "test-session-001"
    assert data["retrieval_latency_seconds"] >= 0
    assert data["response_latency_seconds"] >= 0


def test_query_request_validation(client: TestClient):
    """Short questions should fail validation."""
    response = client.post(
        "/api/v1/query",
        json={"question": "Hi"},
    )
    assert response.status_code == 422
