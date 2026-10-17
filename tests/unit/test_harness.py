from cheat_at_search.agent.harness import Harness


def test_harness_passes_agent_state_to_chat():
    class CapturingAgent:
        def __init__(self):
            self.agent_state = None

        def loop(self, inputs=None, agent_state=None):
            self.agent_state = agent_state
            agent_state["resource.output_tokens"] = 1
            return "result", inputs

        def config_hash(self):
            return "test-config"

    agent = CapturingAgent()
    agent_state = {"tenant": "example"}
    harness = Harness(agent=agent, agent_state=agent_state)

    response, inputs = harness.run(inputs=[])

    assert agent.agent_state is agent_state
    assert harness.agent_state["resource.output_tokens"] == 1
    assert (response, inputs) == ("result", [])


def test_harness_creates_agent_state_when_not_provided():
    class CapturingAgent:
        def loop(self, inputs=None, agent_state=None):
            self.agent_state = agent_state
            agent_state["resource.num_tool_calls"] = 2
            return "result", inputs

        def config_hash(self):
            return "test-config"

    agent = CapturingAgent()
    harness = Harness(agent=agent)

    harness.run(inputs=[])

    assert agent.agent_state is harness.agent_state
    assert agent.agent_state == {"resource.num_tool_calls": 2}
