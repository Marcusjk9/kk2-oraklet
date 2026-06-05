"""
Tester för FastAPI-endpoints med TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from app import data
from app.main import app

client = TestClient(app)

GILTIG_CSV = (
    b"name,age,overall_rating,potential,nationality,positions\n"
    b"L. Messi,31,94,94,Argentina,\"CF,RW,ST\"\n"
    b"C. Ronaldo,33,94,93,Portugal,\"ST,LW\"\n"
    b"K. De Bruyne,27,91,91,Belgium,\"CAM,CM\"\n"
)


@pytest.fixture(autouse=True)
def rensa_dataset():
    """Återställer det globala datasetet innan varje test."""
    data._dataset = None
    yield
    data._dataset = None


# --- /health ---


def test_health_returnerar_ok():
    svar = client.get("/health")
    assert svar.status_code == 200
    assert svar.json() == {"status": "ok"}


# --- /data/upload ---


def test_upload_giltig_csv_returnerar_metadata():
    svar = client.post(
        "/data/upload",
        files={"fil": ("fifa_players.csv", GILTIG_CSV, "text/csv")},
    )
    assert svar.status_code == 200
    kropp = svar.json()
    assert kropp["rows"] == 3
    assert "name" in kropp["columns"]
    assert "overall_rating" in kropp["columns"]
    assert "dtypes" in kropp


def test_upload_fel_filtyp_returnerar_400():
    svar = client.post(
        "/data/upload",
        files={"fil": ("data.txt", b"foo,bar\n1,2\n", "text/plain")},
    )
    assert svar.status_code == 400


def test_upload_tom_fil_returnerar_400():
    svar = client.post(
        "/data/upload",
        files={"fil": ("tom.csv", b"", "text/csv")},
    )
    assert svar.status_code == 400


def test_upload_ogiltig_csv_returnerar_400():
    svar = client.post(
        "/data/upload",
        files={"fil": ("fel.csv", b"\x00\xff\xfe ogiltig", "text/csv")},
    )
    assert svar.status_code == 400


# --- /data/stats ---


def test_stats_utan_dataset_returnerar_404():
    svar = client.get("/data/stats")
    assert svar.status_code == 404


def test_stats_med_dataset_returnerar_200():
    client.post("/data/upload", files={"fil": ("fifa_players.csv", GILTIG_CSV, "text/csv")})
    svar = client.get("/data/stats")
    assert svar.status_code == 200
    kropp = svar.json()
    assert "overall_rating" in kropp or "age" in kropp


# --- /ai/ask ---


def test_ask_utan_dataset_returnerar_400():
    svar = client.post("/ai/ask", json={"question": "Vilken spelare har högst overall?"})
    assert svar.status_code == 400


def test_ask_med_mockad_modell(monkeypatch):
    """Testar hela /ai/ask-flödet med en mockad oraklet-kedja."""
    from app.chain.steps import ResponseParserOutput
    from unittest.mock import MagicMock
    import app.main as main_modul

    mock_oraklet = MagicMock()
    mock_oraklet.invoke.return_value = ResponseParserOutput(
        answer="L. Messi och C. Ronaldo har högst overall_rating med 94.",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
    )

    monkeypatch.setattr(main_modul, "oraklet", mock_oraklet)

    client.post("/data/upload", files={"fil": ("fifa_players.csv", GILTIG_CSV, "text/csv")})
    svar = client.post("/ai/ask", json={"question": "Vilken spelare har högst overall?"})
    assert svar.status_code == 200
    kropp = svar.json()
    assert "question" in kropp
    assert "answer" in kropp
    assert "model" in kropp
    assert kropp["answer"] == "L. Messi och C. Ronaldo har högst overall_rating med 94."
