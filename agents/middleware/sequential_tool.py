from langchain.agents.middleware import AgentMiddleware


class SequentialToolCallMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model = request.model.bind_tools(
            request.tools,
            parallel_tool_calls=False
        )
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model = request.model.bind_tools(
            request.tools,
            parallel_tool_calls=False
        )
        return await handler(request)