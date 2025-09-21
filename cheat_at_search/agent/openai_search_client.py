from cheat_at_search.agent.search_client import SearchClient, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from openai import OpenAI


logger = log_to_stdout("openai_search_client")


class OpenAISearchClient(SearchClient):
    def __init__(self, mcp_url: str, model=str,
                 system_prompt: str = "",
                 response_model=SearchResults):
        self.mcp_url = mcp_url
        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        self.openai_key = key_for_provider("openai")
        self.system_prompt = system_prompt
        if self.provider != 'openai':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports OpenAI.")
        if not self.mcp_url.endswith("/mcp"):
            self.mcp_url = self.mcp_url + "/mcp"
        self.response_model = response_model
        self.openai = OpenAI(api_key=self.openai_key)

    def search(self, prompt: str) -> SearchResults:
        logger.info(f"Calling MCP search tools at {self.mcp_url}")
        input = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            resp = self.openai.responses.parse(
                model=self.model,
                input=input,
                tools=[
                    {
                        "type": "mcp",
                        "server_label": "search-server",
                        "server_url": self.mcp_url,
                        "require_approval": "never",
                    },
                ],
                text_format=self.response_model
            )
            return resp.output_parsed
        except Exception as e:
            print("Error calling MCP search tool:", e)
            raise e


class OpenAIChatAdapter:
    def __init__(self, client: OpenAISearchClient):
        self.client = client
        self.reset()

    def chat(self, message):
        try:
            self.inputs.append({"role": "user", "content": message})
            resp = self.client.openai.responses.create(
                model=self.client.model,
                input=self.inputs,
                tools=[
                    {
                        "type": "mcp",
                        "server_label": "search-server",
                        "server_url": self.client.mcp_url,
                        "require_approval": "never",
                    },
                ],
            )
            agent_resp = resp.output_text
            self.inputs.append({"role": "assistant", "content": agent_resp})
            return agent_resp
        except Exception as e:
            print("Error calling OpenAI chat:", e)
            raise e

    def reset(self, system=None):
        if system is None:
            system = self.client.system_prompt
        self.inputs = [{"role": "system", "content": system}]
