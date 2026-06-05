import logging

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import HealthResponse, AskRequest, AskResponse, UploadResponse 
from app import data
from app.chain.pipeline import skapa_oraklet
from app.chain.steps import PromptBuilderInput


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="KK2 – Oraklet", version="0.1.0")
oraklet = skapa_oraklet()


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

@app.post("/ai/ask", response_model=AskResponse)
def ask(begaran: AskRequest) -> AskResponse:
    if not data.dataset_laddat():
        raise HTTPException(
            status_code=400,
            detail="Ladda upp ett dataset via POST /data/upload innan du ställer en fråga.",
        )
    statistik = data.hamta_statistik()
    logger.info("Kör oraklet för fråga: %s", begaran.question)

    try:
        resultat = oraklet.invoke(PromptBuilderInput(question=begaran.question, stats=statistik))
    except Exception as e:
        logger.error("Modellfel: %s", e)
        raise HTTPException(status_code=502, detail=f"Modellen svarade inte: {e}") from e

    return AskResponse(
        question=begaran.question,
        answer=resultat.answer,
        model=resultat.model,
    )