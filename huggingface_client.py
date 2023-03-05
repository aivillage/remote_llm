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
    client = ClientLLM(host="localhost", port=50055, api_key=api_key)
    # Get model info.
    result = await client.llm_name()
    print(f"Running Model: {result}")
    # Generate text.
    result = await client.generate_text(["This is a prompt."])
    print(f"Generated Text: {result}")
    # Get generational guts.
    guts = await client.generational_guts("This is a prompt.")
    print(f"Generational Guts: {guts}")

if __name__ == "__main__":
    asyncio.run(main())