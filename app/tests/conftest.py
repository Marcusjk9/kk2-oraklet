from unittest.mock import MagicMock, patch


def pytest_configure(config):

    import app.chain.pipeline as pipeline_module 

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = [
        {"generated_text": [{"role": "assistant", "content": "mock svar"}]}
    ]

    mock_hf_pipeline = MagicMock(return_value=mock_pipeline_instance)


    patch.object(pipeline_module, "hf_pipeline", mock_hf_pipeline).start()