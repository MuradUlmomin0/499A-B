"""Approved convex fusion of current temporal and class-utility weights."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import torch
from Support.Utils.parameters import validate_state
from Support.federated.aggregation import FedAvgAggregator


def blend_weights(temporal,class_utility,lam=.5):
    if type(lam) not in (int,float) or not math.isfinite(lam) or not 0<=lam<=1:raise ValueError('Invalid fusion lambda')
    if not temporal or set(temporal)!=set(class_utility):raise ValueError('Component client identities differ')
    for weights in [temporal,class_utility]:
        if any(not math.isfinite(v) or v<0 for v in weights.values()) or not math.isclose(sum(weights.values()),1.,rel_tol=0,abs_tol=1e-12):raise ValueError('Invalid component mass; no fallback')
    if lam==1:return dict(temporal)
    if lam==0:return dict(class_utility)
    return {c:lam*temporal[c]+(1-lam)*class_utility[c] for c in temporal}


class ProposedAggregator:
    """Advance both components once, then weight original local states once."""
    def __init__(self,temporal,class_utility,lam=.5,diagnostic_dir=None):
        if type(lam) not in (int,float) or not math.isfinite(lam) or not 0<=lam<=1:raise ValueError('Invalid fusion lambda')
        self.temporal=temporal;self.class_utility=class_utility;self.lam=lam
        self.directory=Path(diagnostic_dir) if diagnostic_dir is not None else None
        if self.directory:self.directory.mkdir(parents=True,exist_ok=False)
        self.last=None

    def aggregate(self,results):
        stage='initialization';temporal=None;utility=None
        try:
            stage='input_validation'
            FedAvgAggregator().aggregate(results)
            stage='temporal'
            _,temporal=self.temporal.aggregate(results)
            stage='class_utility'
            _,utility=self.class_utility.aggregate(results)
            stage='fusion_weights'
            weights=blend_weights(temporal,utility,self.lam)
            if set(weights)!={r.client_id for r in results}:raise ValueError('Missing fusion contributions')
            stage='weighted_state'
            reference=results[0].state;state=OrderedDict()
            for key,value in reference.items():
                acc=torch.zeros_like(value,dtype=torch.float64,device='cpu')
                for result in results:acc.add_(result.state[key].detach().cpu().to(torch.float64),alpha=weights[result.client_id])
                state[key]=acc.to(value.dtype)
            validate_state(state,reference)
            self.last=dict(round_id=results[0].round_id,lambda_value=self.lam,temporal_weights=temporal,class_utility_weights=utility,weights=weights)
            if self.directory:
                with (self.directory/f'round_{results[0].round_id:03d}.json').open('x') as f:json.dump(self.last,f,indent=2,allow_nan=False)
            return state,weights
        except Exception as error:
            if self.directory:
                rnd=results[0].round_id if results else 0
                failure=dict(round_id=rnd,stage=stage,error_type=type(error).__name__,error=str(error))
                with (self.directory/f'failure_round_{rnd:03d}.json').open('x') as f:json.dump(failure,f,indent=2,allow_nan=False)
                evidence=dict(client_order=[r.client_id for r in results],client_states={r.client_id:{k:v.detach().cpu().clone() for k,v in r.state.items()} for r in results},
                    temporal_weights=temporal,class_utility_weights=utility)
                with (self.directory/f'failure_round_{rnd:03d}.pt').open('xb') as f:torch.save(evidence,f)
            raise
