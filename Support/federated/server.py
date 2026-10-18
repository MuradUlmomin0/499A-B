"""Synchronous local FL simulation with a replaceable aggregator."""
import math
import random
from time import perf_counter
import torch
from support.utils.parameters import clone_state
from support.evaluation.metrics import evaluate_model
from .history import UpdateHistory


class FederatedServer:
    def __init__(self, model, clients, aggregator, evaluation_loader, num_classes, participation_rate, seed, device='cpu', logger=None):
        if not clients or len({c.client_id for c in clients}) != len(clients):
            raise ValueError('Clients must be nonempty with unique IDs')
        if not math.isfinite(participation_rate) or not 0 < participation_rate <= 1:
            raise ValueError('participation_rate must be in (0,1]')
        self.model = model.to(device)
        self.clients = sorted(clients,key=lambda c:c.client_id)
        self.aggregator, self.evaluation_loader = aggregator, evaluation_loader
        self.num_classes, self.participation_rate = num_classes, participation_rate
        self.seed, self.device, self.logger = seed, device, logger
        self.history, self.round_records = UpdateHistory(), []
        self.current_round = 0

    def select_clients(self, round_id):
        count = max(1,math.ceil(self.participation_rate*len(self.clients)))
        return random.Random(self.seed+round_id).sample(self.clients,count)

    def run(self, rounds):
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be positive')
        final_metrics = None
        for round_id in range(self.current_round+1,self.current_round+rounds+1):
            start = perf_counter()
            selected = self.select_clients(round_id)
            global_state = clone_state(self.model.state_dict())
            results = []
            for client in selected:
                client.receive_global(global_state)
                result = client.train(round_id)
                results.append(result)
            aggregation_start = perf_counter()
            state, weights = self.aggregator.aggregate(results)
            aggregation_duration = perf_counter()-aggregation_start
            self.model.load_state_dict(state,strict=True)
            # Evaluation happens only after aggregation; predictions never set weights.
            final_metrics = evaluate_model(self.model,self.evaluation_loader,self.device,self.num_classes)
            for result in results:
                self.history.record(result)
            record = {'round_id':round_id,'selected_clients':[c.client_id for c in selected],
                      'round_duration':perf_counter()-start,'aggregation_duration':aggregation_duration,
                      **final_metrics}
            self.round_records.append(record)
            self.current_round = round_id
            if self.logger is not None:
                self.logger.log_round(record,results,weights)
        if self.logger is not None:
            self.logger.finish(final_metrics,self.checkpoint())
        return final_metrics

    def checkpoint(self):
        return {'model_state':clone_state(self.model.state_dict()),'round_id':self.current_round,
                'history':self.history.state_dict(),'seed':self.seed,
                'resume_supported':False}
