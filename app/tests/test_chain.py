
# Tester för Runnable-kedjans steg i isolation.


from unittest.mock import MagicMock

from app.chain.runnable import Runnable, RunnableSequence
from app.chain.steps import (
    LLMRunner,
    LLMRunnerOutput,
    PromptBuilder,
    PromptBuilderInput,
    PromptBuilderOutput,
    ResponseParser,
    ResponseParserOutput,
)


def _mock_pipe(svar: str) -> MagicMock:
    """Skapar en mock som imiterar transformers.pipeline chat-output."""
    pipe = MagicMock()
    pipe.return_value = [{"generated_text": [{"role": "assistant", "content": svar}]}]
    return pipe


# PromptBuilder


def test_prompt_builder_inkluderar_fragan():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="Vad är max?", stats={"temp_c": {"max": 8.3}})
    utdata = steg.invoke(indata)
    assert "Vad är max?" in utdata.user

def test_prompt_builder_inkluderar_statistik():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="Test?", stats={"temp_c": {"mean": 7.7, "max": 8.3}})
    utdata = steg.invoke(indata)
    assert "temp_c" in utdata.user
    assert len(utdata.system) > 0

def test_prompt_builder_returnerar_pydantic_modell():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="?", stats={})
    utdata = steg.invoke(indata)
    assert isinstance(utdata, PromptBuilderOutput)


# LLMRunner

def test_llm_runner_anropar_pipeline_med_meddelanden():
    mock_pipeline = _mock_pipe("Göteborg är varmast.")
    steg = LLMRunner(pipeline=mock_pipeline, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    indata = PromptBuilderOutput(system="Du är ett verktyg.", user="Vilken stad är varmast?")
    utdata = steg.invoke(indata)

    assert utdata.raw_text == "Göteborg är varmast."
    mock_pipeline.assert_called_once()

def test_llm_runner_returnerar_llmrunneroutput():
    steg = LLMRunner(pipeline=_mock_pipe("Svar."), model_name="test-modell")
    utdata = steg.invoke(PromptBuilderOutput(system="s", user="u"))
    assert isinstance(utdata, LLMRunnerOutput)


# ResponseParser

def test_response_parser_trimmar_blanksteg():
    steg = ResponseParser(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    indata = LLMRunnerOutput(raw_text="  Malmö har 8.3 °C.  \n")
    utdata = steg.invoke(indata)
    assert utdata.answer == "Malmö har 8.3 °C."

def test_response_parser_satter_modellnamn():
    steg = ResponseParser(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    utdata = steg.invoke(LLMRunnerOutput(raw_text="svar"))
    assert utdata.model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert isinstance(utdata, ResponseParserOutput)


# Runnable-kedjan

def test_pipe_operatorn_skapar_runnable_sequence():
    steg1 = PromptBuilder()
    steg2 = LLMRunner(pipeline=_mock_pipe("Svar."), model_name="modell")
    steg3 = ResponseParser(model_name="modell")

    kedja = steg1 | steg2 | steg3
    assert isinstance(kedja, RunnableSequence)

def test_hel_kedja_med_mockad_llm():
    kedja = (
        PromptBuilder()
        | LLMRunner(pipeline=_mock_pipe("Malmö är varmast."), model_name="test-modell")
        | ResponseParser(model_name="test-modell")
    )
    indata = PromptBuilderInput(question="Vilken stad är varmast?", stats={"temp_c": {"max": 8.3}})
    resultat = kedja.invoke(indata)

    assert isinstance(resultat, ResponseParserOutput)
    assert resultat.answer == "Malmö är varmast."
    assert resultat.model == "test-modell"
