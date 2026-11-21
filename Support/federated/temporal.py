"""Approved unsupervised SecureDyn-inspired adaptation, not a reproduction.

Temporal coordinates are fixed; thresholds use previous observations only.
Statistical rejection is not evidence that a client is an attacker.
"""
from collections import OrderedDict, deque
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import torch

from Support.Utils.parameters import clone_state, flatten_update, validate_state
from .aggregation import FedAvgAggregator

DEFAULTS=dict(projection_dim=5,projection_seed=1729,window=5,components=2,
    reg_covar=1e-6,max_iter=200,tol=1e-3,forgetting=.8,warmup_rounds=3,
    calibration_min=20,threshold_std=3.,threshold_floor=1e-8,severe_ratio=2.,
    downweight=.5)


def thresholds(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Invalid historical calibration')
    return np.maximum(x.mean(axis=0)+DEFAULTS['threshold_std']*x.std(axis=0),DEFAULTS['threshold_floor'])


def decisions(evidence, limits):
    x=np.asarray(evidence,dtype=np.float64);limits=np.asarray(limits,dtype=np.float64)
    if (x.ndim!=2 or limits.shape!=(x.shape[1],) or not np.isfinite(x).all()
            or not np.isfinite(limits).all() or (x<0).any() or (limits<=0).any()):
        raise ValueError('Invalid evidence/thresholds')
    ratios=(x/limits).max(axis=1)
    trust=np.where(ratios<=1,1.,np.where(ratios<=2,.5,0.))
    status=np.where(ratios<=1,'accept',np.where(ratios<=2,'downweight','reject'))
    return trust,status,ratios


def blend_mixture(old,new,forgetting):
    if old is None:return {k:np.array(v,copy=True) for k,v in new.items()}
    rows,cols=linear_sum_assignment(((old['means'][:,None]-new['means'][None,:])**2).sum(axis=2))
    order=cols[np.argsort(rows)]
    return {k:forgetting*old[k]+(1-forgetting)*new[k][order] for k in old}


def mahalanobis(points,mixture):
    x=np.asarray(points,dtype=np.float64);v=mixture['variances'];w=mixture['weights'];means=mixture['means']
    if not np.isfinite(v).all() or (v<=0).any() or (w<=0).any():raise ValueError('Invalid mixture covariance/mass')
    distances=((x[:,None,:]-means[None,:,:])**2/v[None,:,:]).sum(axis=2)
    logp=np.log(w)[None,:]-.5*(distances+np.log(v).sum(axis=1)[None,:]+x.shape[1]*np.log(2*np.pi))
    posterior=logp-logsumexp(logp,axis=1,keepdims=True)
    labels=posterior.argmax(axis=1);md=np.sqrt(distances[np.arange(len(x)),labels])
    if not np.isfinite(md).all():raise ValueError('Nonfinite Mahalanobis distance')
    return md,labels


class TemporalTrustAggregator:
    """Sample-count times temporal trust, with bounded observation history."""
    def __init__(self, *, diagnostic_dir=None):
        self.settings=dict(DEFAULTS);self.observations=deque(maxlen=5);self.calibration=deque(maxlen=5)
        self.mixture=None;self.projection=None;self.reference=None;self.last_round=0
        self.previous_md={};self.last=None
        self.directory=Path(diagnostic_dir) if diagnostic_dir is not None else None
        if self.directory:self.directory.mkdir(parents=True,exist_ok=False)

    def aggregate(self,results):
        # Shared validator checks counts, duplicate IDs, state schema and finite values.
        FedAvgAggregator().aggregate(results)
        rnd=results[0].round_id
        if type(rnd) is not int or rnd<=self.last_round:raise ValueError('Rounds must strictly increase')
        ordered=sorted(results,key=lambda r:r.client_id);ids=[r.client_id for r in ordered]
        reference=ordered[0].state
        if self.reference is not None:validate_state(reference,self.reference)
        for r in ordered:validate_state(r.update,reference)
        vectors=np.stack([flatten_update(r.update).numpy() for r in ordered])
        if self.projection is None:
            self.reference=clone_state(reference)
            self.projection=np.random.default_rng(1729).normal(0,1/np.sqrt(5),size=(vectors.shape[1],5))
            if self.directory:np.save(self.directory/'projection.npy',self.projection,allow_pickle=False)
        projected=vectors@self.projection;norms=np.linalg.norm(vectors,axis=1)
        if not np.isfinite(projected).all() or not np.isfinite(norms).all():raise ValueError('Nonfinite projected update/norm')
        # Window counts communication rounds, not the last five participation events.
        while self.observations and self.observations[0][0]<rnd-4:self.observations.popleft()
        while self.calibration and self.calibration[0][0]<rnd-5:self.calibration.popleft()
        pool=np.concatenate([v for _,v in self.observations]+[projected])
        md=[None]*len(ids);trajectory=[None]*len(ids);labels=None;reason=None
        try:
            if len(np.unique(pool,axis=0))<2:reason='fewer_than_two_distinct_observations'
            else:
                fitted=GaussianMixture(n_components=2,covariance_type='diag',reg_covar=1e-6,
                    n_init=1,max_iter=200,tol=1e-3,random_state=1729).fit(pool)
                if not fitted.converged_:raise ValueError('GMM EM did not converge')
                new=dict(means=fitted.means_,variances=fitted.covariances_,weights=fitted.weights_)
                self.mixture=blend_mixture(self.mixture,new,.8)
                if any(not np.isfinite(v).all() for v in self.mixture.values()):raise ValueError('Nonfinite fitted mixture')
                md,labels=mahalanobis(projected,self.mixture);md=md.tolist()
                for i,c in enumerate(ids):
                    prior=self.previous_md.get(c)
                    if prior is not None and prior[0]==rnd-1:trajectory[i]=abs(md[i]-prior[1])
            previous=[r for _,rows in self.calibration for r in rows]
            norm_history=[r['norm'] for r in previous]
            md_history=[r['md'] for r in previous if r['md'] is not None]
            trajectory_history=[r['trajectory'] for r in previous if r['trajectory'] is not None]
            limits={key:float(thresholds(np.asarray(values)[:,None])[0]) if len(values)>=20 else None
                    for key,values in [('norm',norm_history),('md',md_history),('trajectory',trajectory_history)]}
            if reason is None:
                if rnd<=3:reason='initial_three_rounds'
                elif limits['norm'] is None or limits['md'] is None:reason='insufficient_prior_calibration'
            rows=[]
            for i,c in enumerate(ids):
                row=dict(client_id=c,sample_count=ordered[i].sample_count,norm=float(norms[i]),md=md[i],trajectory=trajectory[i],
                    component=int(labels[i]) if labels is not None else None,trust=1.,decision='warmup',ratio=None)
                if reason is None:
                    names=[k for k in ['norm','md','trajectory'] if row[k] is not None and limits[k] is not None]
                    trust,status,ratio=decisions([[row[k] for k in names]],[limits[k] for k in names])
                    row.update(trust=float(trust[0]),decision=str(status[0]),ratio=float(ratio[0]))
                rows.append(row)
            total=sum(r['sample_count']*r['trust'] for r in rows)
            self.last=dict(round_id=rnd,phase='warmup' if reason else 'scored',reason=reason,thresholds=limits,clients=rows)
            if total<=0:raise ValueError('Every client rejected; aggregation stopped')
            weights={r['client_id']:r['sample_count']*r['trust']/total for r in rows}
            ordinary_total=sum(r['sample_count'] for r in rows)
            for row in rows:
                row['weight']=weights[row['client_id']];row['fedavg_weight']=row['sample_count']/ordinary_total
            state=OrderedDict()
            for key,value in reference.items():
                acc=torch.zeros_like(value,dtype=torch.float64,device='cpu')
                for r in results:acc.add_(r.state[key].detach().cpu().to(torch.float64),alpha=weights[r.client_id])
                state[key]=acc.to(value.dtype)
            validate_state(state,reference)
            self._save(rnd,results,projected,state)
        except Exception as error:
            if self.directory:
                (self.directory/f'failure_{rnd:03d}.json').write_text(json.dumps(dict(round_id=rnd,error=str(error),diagnostics=self.last),indent=2,allow_nan=False))
                torch.save(dict(client_states={r.client_id:clone_state(r.state) for r in results},
                    client_updates={r.client_id:clone_state(r.update) for r in results}),self.directory/f'failure_{rnd:03d}.pt')
            raise
        self.observations.append((rnd,projected.copy()));self.calibration.append((rnd,rows))
        self.previous_md={c:(rnd,d) for c,d in zip(ids,md) if d is not None}
        self.last_round=rnd
        return state,weights

    def _save(self,rnd,results,projected,state):
        if self.directory is None:return
        prefix=self.directory/f'round_{rnd:03d}'
        prefix.with_suffix('.json').write_text(json.dumps(self.last,indent=2,allow_nan=False)+'\n')
        mixture=self.mixture or {}
        np.savez(prefix.with_suffix('.npz'),projected=projected,**mixture)
        torch.save(dict(client_states={r.client_id:clone_state(r.state) for r in results},
            client_order=[r.client_id for r in results],aggregated_state=clone_state(state)),prefix.with_suffix('.pt'))
