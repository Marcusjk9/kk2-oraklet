import logging
from typing import Any

from pydantic import BaseModel

from app.chain.runnable import Runnable

logger = logging.getLogger(__name__)




class PromptBuilderInput(BaseModel):
    question: str
    stats: dict[str, Any]


class PromptBuilderOutput(BaseModel):
    system: str
    user: str


class LLMRunnerOutput(BaseModel):
    raw_text: str


class ResponseParserOutput(BaseModel):
    answer: str
    model: str




class PromptBuilder(Runnable[PromptBuilderInput, PromptBuilderOutput]):
    name: str = "prompt_builder"

    def invoke(self, data: PromptBuilderInput) -> PromptBuilderOutput:
        stats_rader = []
        for kolumn, varden in data.stats.items():
            if isinstance(varden, dict):
                detaljer = ", ".join(f"{k}={v}" for k, v in varden.items() if v != "")
                stats_rader.append(f"  {kolumn}: {detaljer}")
            else:
                stats_rader.append(f"  {kolumn}: {varden}")
        stats_text = "\n".join(stats_rader)

        system = (
            "Du är ett dataanalysverktyg. Du får beskrivande statistik från ett CSV-dataset "
            "och svarar på svenska med korta, faktabaserade svar. "
            "Svara enbart på frågan – inga inledningar, hälsningar eller avslutningar."
        )
        user = (
            f"Här är beskrivande statistik från datasetet:\n{stats_text}\n\n"
            f"Fråga: {data.question}"
        )
        logger.info("Prompt byggd för fråga: %s", data.question)
        return PromptBuilderOutput(system=system, user=user)


class LLMRunner(Runnable[PromptBuilderOutput, LLMRunnerOutput]):
    name: str = "llm_runner"
    pipeline: Any  # transformers.pipeline-objekt
    model_name: str

    def invoke(self, data: PromptBuilderOutput) -> LLMRunnerOutput:
        messages = [
            {"role": "system", "content": data.system},
            {"role": "user", "content": data.user},
        ]
        logger.info("Anropar modell: %s", self.model_name)
        output = self.pipeline(messages, max_new_tokens=300)
        generated = output[0]["generated_text"]
        # Chat-format: generated_text är en lista av meddelanden
        if isinstance(generated, list):
            raw_text = generated[-1]["content"]
        else:
            raw_text = str(generated)
        logger.info("Modell svarade med %d tecken", len(raw_text))
        return LLMRunnerOutput(raw_text=raw_text)


class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
    name: str = "response_parser"
    model_name: str

    def invoke(self, data: LLMRunnerOutput) -> ResponseParserOutput:
        svar = data.raw_text.strip()
        logger.info("Svar parsat, längd: %d tecken", len(svar))
        return ResponseParserOutput(answer=svar, model=self.model_name)