# Remote LLM

This is a simple wrapper for langchain that allows us to call it remotely using gRPC through betterproto. 

To regenerate the protobuf interface from the `llm_rpc.proto` run this:
```bash
python -m grpc_tools.protoc -I . --python_betterproto_out=remote_llm llm_rpc.proto
```