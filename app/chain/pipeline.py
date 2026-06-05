from transformers import pipeline as hf_pipeline

from app.chain.steps import LLMRunner, PromptBuilder, ResponseParser
from app.config import settings


def skapa_oraklet():
    pipe = hf_pipeline("text-generation", model=settings.model_name)
    return (
        PromptBuilder()
        | LLMRunner(pipeline=pipe, model_name=settings.model_name)
        | ResponseParser(model_name=settings.model_name)
    )
