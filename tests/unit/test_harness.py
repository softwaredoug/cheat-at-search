from cheat_at_search.agent.harness import Harness


def test_harness_passes_agent_state_to_chat():
    class CapturingAgent:
        def __init__(self):
            self.agent_state = None

        def loop(self, inputs=None, agent_state=None, return_usage=False):
            self.agent_state = agent_state
            return "result", inputs, {
                "input_tokens": 1,
                "output_tokens": 1,
                "num_tool_calls": 0,
            }

        def config_hash(self):
            return "test-config"

    agent = CapturingAgent()
    agent_state = {"tenant": "example"}
    harness = Harness(agent=agent, agent_state=agent_state)

    harness.run(inputs=[])

    assert agent.agent_state is agent_state
