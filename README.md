# Remote LLM

This is a simple wrapper for a large language model that allows us to call it remotely using gRPC through betterproto. There's a simple server that provides GPTNeo-1.3b too. If you just need a basic client install just the client requirements, `client_requirements.txt`. 

There's an example server and client in the scripts `huggingface_service.py` and `huggingface_client.py`. Modify these to suit your needs, then deploy.

## Development

To regenerate the protobuf interface from the `llm_rpc.proto` run this:
```bash
python -m grpc_tools.protoc -I . --python_betterproto_out=remote_llm llm_rpc.proto
```

## Testing

First start the service LLM with `python mock_llm.py` in one terminal, then in another terminal run the tests with `pytest`. Once it passes all those test, run the example server and 


## Building

`python -m build`
