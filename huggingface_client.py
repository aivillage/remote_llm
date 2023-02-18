'''
An example client with a example API key to a remote LLM.
'''

import logging
import asyncio
from remote_llm.client import ClientLLM
from grpclib.client import Channel

logger = logging.getLogger(__name__)
api_key = "482fdd5f-b59c-43de-98b9-4e19a21b4d85"
async def main():
    # Create a client to the remote LLM.
    async with Channel(host="localhost", port=50055) as channel:
        client = ClientLLM(channel=channel, api_key=api_key)
        # Get model info.
        result = await client.model_info()
        print(f"Running Model: {result}")
        # Generate text.
        result = await client.generate_text(["This is a prompt."])
        print(f"Generated Text: {result}")

if __name__ == "__main__":
    asyncio.run(main())