# SID Legacy Agent

SID.ai is an agentic search model platform. It implements the completions API and
claims to do a better job with search at lower latency.

Documentation can be found here
https://platform.sid.ai/docs/build

## Match the OpenAIAgent

The SIDLegacyAgent matches the OpenAIAgent interface so that they're interchangable

 Essentially SIDLegacyAgent is an agentic tool calling loop that takes context and iterates

## SIDLegacyAgent special requirements

- Use `key_for_provider("sid")` to get the API key
- Fail if a tool called `read`
- SID wants a [specific XML format][1]. Don't require `search` tools to implement this format, do the conversion
  yourself in the SIDLegacyAgent



[1] https://platform.sid.ai/docs/reference/tools#format-tool-results
