import logging

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import HealthResponse, AskRequest, AskResponse, UploadResponse 
from app import data

logger = logging.getLogger(__name__)

app = FastAPI(title="KK2 – Oraklet", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/data/upload", response_model=UploadResponse)
async def upload_data(fil: UploadFile = File(...)) -> UploadResponse:
    if not (fil.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Filen måste vara en CSV-fil (.csv).")

    innehall = await fil.read()
    if not innehall:
        raise HTTPException(status_code=400, detail="Filen är tom.")

    try:
        metadata = data.spara_dataset(innehall)
    except Exception as e:
        logger.error("Kunde inte läsa CSV-fil '%s': %s", fil.filename, e)
        raise HTTPException(status_code=400, detail=f"Kunde inte tolka CSV-filen: {e}") from e

    return UploadResponse(**metadata)

@app.get("/data/stats")
def get_stats() -> dict:
    statistik = data.hamta_statistik()
    if statistik is None:
        raise HTTPException(status_code=404, detail="Inget dataset har laddats upp ännu.")
    return statistik