'''
An example client with a SSL connection to a remote LLM.
'''

import logging
from remote_llm import ClientLLM
import asyncio
from grpclib.client import Channel

logger = logging.getLogger(__name__)

async def main():
    # Create a client to the remote LLM.
    async with Channel(host="localhost", port=50055) as channel:
        client = ClientLLM(channel=channel)

        # Generate text.
        result = await client.async_generate_text(["This is a prompt."])
        print(result)

if __name__ == "__main__":
    asyncio.run(main())