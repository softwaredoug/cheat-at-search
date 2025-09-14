from cheat_at_search.agent.search_client import SearchClient, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from openai import OpenAI


logger = log_to_stdout("openai_search_client")


class OpenAISearchClient(SearchClient):
    def __init__(self, mcp_url: str, model=str):
        self.mcp_url = mcp_url
        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        self.openai_key = key_for_provider("openai")
        if self.provider != 'openai':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports OpenAI.")
        if not self.mcp_url.endswith("/mcp"):
            self.mcp_url = self.mcp_url + "/mcp"
        self.response_model = SearchResults

    def search(self, prompt: str) -> SearchResults:
        self.openai = OpenAI(api_key=self.openai_key)
        logger.info(f"Calling MCP search tool at {self.mcp_url}")
        try:
            resp = self.openai.responses.parse(
                model=self.model,
                input=prompt,
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
