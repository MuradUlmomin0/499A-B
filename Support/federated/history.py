from support.utils.parameters import clone_state


class UpdateHistory:
    """Only observed participation rounds are recorded; no anomaly scoring."""
    def __init__(self):
        self.records = {}

    def record(self, result):
        rounds = self.records.setdefault(result.client_id, {})
        if result.round_id in rounds:
            raise ValueError('Duplicate client history round')
        rounds[result.round_id] = clone_state(result.update)

    def state_dict(self):
        return {client: {rnd: clone_state(update) for rnd,update in rounds.items()}
                for client,rounds in self.records.items()}
