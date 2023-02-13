from remote_llm import ClientLLM

import pytest

@pytest.mark.asyncio
async def test_async_client():
    client = ClientLLM(host='127.0.0.1', port=50051)
    result = await client.agenerate(['hello'])
    assert result.generations[0][0].text == 'world'
    mock_type = client._llm_type
    assert mock_type == 'remote:mock'

def test_client():
    client = ClientLLM(host='127.0.0.1', port=50051)
    result = client.generate(['hello'])
    assert result.generations[0][0].text == 'world'
    mock_type = client._llm_type
    assert mock_type == 'remote:mock'