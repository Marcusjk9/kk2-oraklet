
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

FIFA_STATS = {
    "overall_rating": {"mean": 66.24, "min": 47.0, "max": 94.0},
    "age": {"mean": 25.57, "min": 17.0, "max": 46.0},
    "potential": {"mean": 71.43, "min": 48.0, "max": 95.0},
    "wage_euro": {"mean": 9902.0, "min": 1000.0, "max": 565000.0},
}


def _mock_pipe(svar: str) -> MagicMock:
    """Skapar en mock som imiterar transformers.pipeline chat-output."""
    pipe = MagicMock()
    pipe.return_value = [{"generated_text": [{"role": "assistant", "content": svar}]}]
    return pipe


# PromptBuilder


def test_prompt_builder_inkluderar_fragan():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="Vilken spelare har högst overall?", stats=FIFA_STATS)
    utdata = steg.invoke(indata)
    assert "Vilken spelare har högst overall?" in utdata.user


def test_prompt_builder_inkluderar_statistik():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="Vad är medelåldern?", stats=FIFA_STATS)
    utdata = steg.invoke(indata)
    assert "overall_rating" in utdata.user
    assert "age" in utdata.user
    assert len(utdata.system) > 0


def test_prompt_builder_returnerar_pydantic_modell():
    steg = PromptBuilder()
    indata = PromptBuilderInput(question="?", stats={})
    utdata = steg.invoke(indata)
    assert isinstance(utdata, PromptBuilderOutput)


def test_prompt_builder_tom_statistik_ger_giltig_prompt():
    steg = PromptBuilder()
    utdata = steg.invoke(PromptBuilderInput(question="Fråga?", stats={}))
    assert "Fråga?" in utdata.user
    assert len(utdata.system) > 0


def test_prompt_builder_inkluderar_wage_och_potential():
    steg = PromptBuilder()
    indata = PromptBuilderInput(
        question="Vem tjänar mest?",
        stats={"wage_euro": {"mean": 9902.0, "max": 565000.0}, "potential": {"mean": 71.43, "max": 95.0}},
    )
    utdata = steg.invoke(indata)
    assert "wage_euro" in utdata.user
    assert "potential" in utdata.user


# LLMRunner


def test_llm_runner_anropar_pipeline_med_meddelanden():
    mock_pipeline = _mock_pipe("L. Messi har högst overall_rating med 94.")
    steg = LLMRunner(pipeline=mock_pipeline, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    indata = PromptBuilderOutput(system="Du är ett verktyg.", user="Vilken spelare har högst overall?")
    utdata = steg.invoke(indata)

    assert utdata.raw_text == "L. Messi har högst overall_rating med 94."
    mock_pipeline.assert_called_once()


def test_llm_runner_returnerar_llmrunneroutput():
    steg = LLMRunner(pipeline=_mock_pipe("Svar."), model_name="test-modell")
    utdata = steg.invoke(PromptBuilderOutput(system="s", user="u"))
    assert isinstance(utdata, LLMRunnerOutput)


# ResponseParser


def test_response_parser_trimmar_blanksteg():
    steg = ResponseParser(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    indata = LLMRunnerOutput(raw_text="  L. Messi har overall 94.  \n")
    utdata = steg.invoke(indata)
    assert utdata.answer == "L. Messi har overall 94."


def test_response_parser_satter_modellnamn():
    steg = ResponseParser(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    utdata = steg.invoke(LLMRunnerOutput(raw_text="svar"))
    assert utdata.model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert isinstance(utdata, ResponseParserOutput)


def test_response_parser_hanterar_tomt_svar():
    steg = ResponseParser(model_name="test")
    utdata = steg.invoke(LLMRunnerOutput(raw_text="   "))
    assert utdata.answer == ""


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
        | LLMRunner(pipeline=_mock_pipe("L. Messi har högst overall med 94."), model_name="test-modell")
        | ResponseParser(model_name="test-modell")
    )
    indata = PromptBuilderInput(
        question="Vilken spelare har högst overall?",
        stats=FIFA_STATS,
    )
    resultat = kedja.invoke(indata)

    assert isinstance(resultat, ResponseParserOutput)
    assert resultat.answer == "L. Messi har högst overall med 94."
    assert resultat.model == "test-modell"
