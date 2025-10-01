from cheat_at_search.agent.search_client import SearchClient, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.pydantize import make_tool_adapter
from openai import OpenAI


logger = log_to_stdout("openai_search_client")


class OpenAISearchClient(SearchClient):
    def __init__(self,
                 tools,
                 model: str,
                 system_prompt: str = "",
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

    def chat(self, prompt: str, inputs=None) -> SearchResults:
        """Chat, handle any response."""
        if not inputs:
            inputs = [
                {"role": "system", "content": self.system_prompt},
            ]
        next_msg = {"role": "user", "content": prompt}
        inputs.append(next_msg)
        tools = []
        for tool in self.search_tools.values():
            tool_spec = tool[1]
            tools.append(tool_spec)
        try:
            tool_calls_found = True
            while tool_calls_found:
                print("Calling model...")
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

                tool_calls_found = False

                print("Calling tools...")
                for item in resp.output:
                    if item.type == "function_call":
                        tool_calls_found = True
                        tool_name = item.name
                        print("Tool call found:", tool_name, item)
                        if tool_name not in self.search_tools:
                            raise ValueError(f"Tool {tool_name} not found in registered tools.")
                        tool_calls_found = True
                        tool = self.search_tools[tool_name]
                        ToolArgsModel = tool[0]
                        tool_fn = tool[2]

                        fn_args: ToolArgsModel = ToolArgsModel.model_validate_json(item.arguments)
                        print("Calling tool:", tool_name, fn_args)
                        py_resp, json_resp = tool_fn(fn_args)
                        # 4. Provide function call results to the model
                        inputs.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json_resp,
                        })
            return resp, inputs
        except Exception as e:
            print("Error calling MCP search tool:", e)
            raise e

    def search(self, prompt: str):
        """Issue a 'search' and expect structured output response."""
        resp, _ = self.chat(prompt)
        return resp.output_parsed


class OpenAIChatAdapter:
    def __init__(self, client: OpenAISearchClient):
        self.client = client
        self.reset()

    def chat(self, message):
        try:
            resp, inputs = self.client.chat(message, inputs=self.inputs)
            self.inputs = inputs
            return resp.output[-1].content[-1].text
        except Exception as e:
            print("Error calling OpenAI chat:", e)
            raise e

    def reset(self, system=None):
        if system is None:
            system = self.client.system_prompt
        self.inputs = []
