from remote_llm.client import ClientLLM

import pytest

@pytest.mark.asyncio
async def test_async_client():
    client = ClientLLM(host='127.0.0.1', port=50051)
    result = await client.generate_text(['hello'])
    assert result.generations[0][0].text == 'world'
    mock_type = await client.llm_name()
    assert mock_type == 'mock'

def test_client():
    client = ClientLLM(host='127.0.0.1', port=50051)
    result = client.sync_generate_text(['hello'])
    assert result.generations[0][0].text == 'world'
    mock_type = client.sync_llm_name()
    assert mock_type == 'mock'