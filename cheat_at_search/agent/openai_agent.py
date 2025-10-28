from cheat_at_search.agent.search_client import Agent, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.pydantize import make_tool_adapter
from openai import OpenAI
from typing import Optional


logger = log_to_stdout("openai_search_client")


class OpenAIAgent(Agent):
    def __init__(self,
                 tools,
                 model: str,
                 system_prompt: str = "",
                 max_tokens: Optional[int] = None,
                 response_model=SearchResults):
        self.search_tools = {tool.__name__: make_tool_adapter(tool) for tool in tools}

        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        self.openai_key = key_for_provider("openai")
        self.system_prompt = system_prompt
        if self.provider != 'openai':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports OpenAI.")
        self.response_model = response_model
        self.openai = OpenAI(api_key=self.openai_key)
        self.last_usage = None
        self.max_tokens = max_tokens

    def chat(self, user_prompt: str = None, inputs=None, return_usage=False) -> SearchResults:
        """Chat, handle any response."""
        tool_call_logs = []
        if not inputs:
            inputs = [
                {"role": "system", "content": self.system_prompt},
            ]
        if user_prompt:
            next_msg = {"role": "user", "content": user_prompt}
            inputs.append(next_msg)
        tools = []
        for tool in self.search_tools.values():
            tool_spec = tool[1]
            tools.append(tool_spec)
        total_tokens = 0
        try:
            tool_calls_found = True
            while tool_calls_found:
                if self.response_model:
                    resp = self.openai.responses.parse(
                        model=self.model,
                        input=inputs,
                        tools=tools,
                        text_format=self.response_model
                    )
                else:
                    resp = self.openai.responses.create(
                        model=self.model,
                        input=inputs,
                        tools=tools,
                    )
                # Iterate over tool calls
                inputs += resp.output

                total_tokens += resp.usage.total_tokens

                logger.debug("Usage: ", resp.usage)
                logger.info(f"Total tokens so far: {total_tokens}")
                if self.max_tokens and total_tokens >= self.max_tokens:
                    logger.info(f"Reached max tokens limit of {self.max_tokens}. Stopping further tool calls.")
                    break

                tool_calls_found = False

                for item in resp.output:
                    if item.type == "function_call":
                        tool_calls_found = True
                        tool_name = item.name
                        if tool_name not in self.search_tools:
                            raise ValueError(f"Tool {tool_name} not found in registered tools.")
                        tool_calls_found = True
                        tool = self.search_tools[tool_name]
                        ToolArgsModel = tool[0]
                        tool_fn = tool[2]

                        tool_call_logs.append({
                            "tool_name": tool_name,
                            "arguments": item.arguments,
                        })

                        fn_args: ToolArgsModel = ToolArgsModel.model_validate_json(item.arguments)
                        py_resp, json_resp = tool_fn(fn_args)
                        # 4. Provide function call results to the model
                        inputs.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json_resp,
                        })
            if return_usage:
                resp.usage = resp.usage
            if len(tool_call_logs) > 0:
                logger.info("**** Search completed ****")
                logger.info("Tool call summary:")
                for log in tool_call_logs:
                    logger.info(f"Tool called: {log['tool_name']}")
            return resp, inputs, total_tokens
        except Exception as e:
            logger.error("Error calling MCP search tool:", e)
            raise e

    def loop(self, user_prompt: str = None, return_usage=False) -> SearchResults:
        """Issue a 'search' and expect structured output response."""
        assert self.response_model is not None, "response_model must be set for structured search results."
        resp, _, total_tokens = self.chat(user_prompt)
        self.last_usage = resp.usage
        if return_usage:
            return resp.output_parsed, total_tokens
        return resp.output_parsed


class OpenAIChatAdapter:
    def __init__(self, client: OpenAIAgent):
        self.client = client
        self.reset()

    def chat(self, message):
        try:
            resp, inputs = self.client.chat(message, inputs=self.inputs)
            self.inputs = inputs
            return resp.output[-1].content[-1].text
        except Exception as e:
            logger.error("Error calling OpenAI chat:", e)
            raise e

    def reset(self, system=None):
        if system is None:
            system = self.client.system_prompt
        self.inputs = []
