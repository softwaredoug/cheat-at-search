# server.py
from fastmcp.tools import Tool
from fastmcp import FastMCP
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
import threading
from pyngrok import ngrok


logger = log_to_stdout("mcp")


# Create a backend index
mcp = FastMCP("search-server",
              instructions="This MCP can search some content (see tool) and give results",
              stateless_http=True)


def serve_tools(fns,
                port=8000):
    for fn in fns:
        name = fn.__name__
        description = fn.__doc__ or "No description provided"
        tool = Tool.from_function(fn, name=name, description=description)
        mcp.add_tool(tool)
    ngrok_key = key_for_provider('ngrok')
    ngrok.set_auth_token(ngrok_key)
    public_url = ngrok.connect(port, bind_tls=True).public_url
    print(" * Ngrok public URL:", public_url)

    # Run is background thread
    # mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    thread = threading.Thread(target=mcp.run, kwargs={"transport": "streamable-http",
                                                      "host": "0.0.0.0",
                                                      "port": port}, daemon=True)
    thread.start()
    return thread, public_url


def stop_serving():
    ngrok_process = ngrok.get_ngrok_process()
    if ngrok_process:
        ngrok.kill()
