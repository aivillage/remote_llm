'''
An example client with a SSL connection to a remote LLM.
'''

import logging
from remote_llm import ClientLLM
import ssl
import asyncio

server_cert="ssl_config/server_keys/ca.crt"
client_cert="ssl_config/client_keys/client.crt"
client_key="ssl_config/client_keys/client.key"

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=server_cert)
context.check_hostname = False
context.load_cert_chain(keyfile=client_key, certfile=client_cert)

logger = logging.getLogger(__name__)

async def main():
    # Create a client to the remote LLM.
    client = ClientLLM(
        host="localhost",
        port=50055,
    )

    # Generate text.
    result = await client._agenerate(["This is a prompt."])
    print(result)

if __name__ == "__main__":
    asyncio.run(main())