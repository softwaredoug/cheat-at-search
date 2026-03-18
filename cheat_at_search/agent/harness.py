def stop_after_one_call(resp, inputs, call_count):
    """Stop after one call."""
    return True


class Harness:
    """Execute inputs until stop conditions."""

    def __init__(
        self,
        agent,
        stoppers=None,
        validators=None,
        response_model=str,
        agent_state=None,
    ):
        self.agent = agent
        self.stoppers = stoppers
        self.validators = validators
        self.response_model = response_model
        self.agent_state = agent_state

    def run(self, inputs):
        stop = False
        stoppers = self.stoppers
        if stoppers is None:
            stoppers = [stop_after_one_call]
        validators = self.validators
        if validators is None:
            validators = []
        call_count = 0
        while not stop:
            call_count += 1
            resp, inputs, total_tokens = self.agent.chat(
                inputs=inputs, return_usage=True
            )
            # Call validators, append any
            invalid = False
            for validator in validators:
                input_len_before = len(inputs)
                valid = validator(resp, inputs)
                if not valid:
                    if len(inputs) <= input_len_before:
                        raise ValueError(
                            f"Validator {validator} should append an error message"
                        )
                    invalid = True
            if invalid:
                continue
            # Call stoppers, append any
            for stopper in stoppers:
                stop = stopper(resp, inputs, call_count)
                if stop:
                    break
        return resp, inputs, total_tokens

    def config_hash(self) -> str:
        return self.agent.config_hash()
