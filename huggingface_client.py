'''
An example client with a SSL connection to a remote LLM.
'''

import logging
from remote_llm.client import ClientLLM
import asyncio
from grpclib.client import Channel

logger = logging.getLogger(__name__)
api_key = "482fdd5f-b59c-43de-98b9-4e19a21b4d85"
api_key = "2909b59b-a5df-4b95-b1d6-e96094f65267"
async def main():
    # Create a client to the remote LLM.
    async with Channel(host="localhost", port=50055) as channel:
        client = ClientLLM(channel=channel, api_key=api_key)

        # Generate text.
        result = await client.async_generate_text(["This is a prompt."])
        print(result)

if __name__ == "__main__":
    asyncio.run(main())