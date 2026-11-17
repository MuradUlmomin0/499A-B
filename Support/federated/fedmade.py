"""User-approved FedMADE adaptation: DBSCAN + nonnegative least squares.

Paper Eqs. 1–5 are mapped in support/docs/cp3_fedmade_design.md. NNLS is an
explicit replacement for the paper's incompletely specified Adam weight solver.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.cluster import DBSCAN
import torch
from torch.utils.data import DataLoader

from support.federated.cpm import class_probability_matrix
from support.utils.parameters import clone_state, validate_state


def validate_cpms(cpms):
    values=np.asarray(cpms,dtype=np.float64)
    if (values.ndim!=3 or not len(values) or values.shape[1]<2 or values.shape[1]!=values.shape[2]
            or not np.isfinite(values).all() or (values<0).any() or (values>1).any()
            or not np.allclose(values.sum(axis=2),1,atol=1e-10,rtol=0)):
        raise ValueError('CPMs must be finite row-stochastic [models, C, C] matrices')
    return values


def cluster_cpms(cpms, eps):
    """Unscaled Euclidean distance on row-major flattened CPMs; no noise clients."""
    cpms=validate_cpms(cpms)
    if type(eps) not in (int,float) or not np.isfinite(eps) or eps<=0:
        raise ValueError('DBSCAN eps must be positive and finite')
    labels=DBSCAN(eps=eps,min_samples=1,metric='euclidean',algorithm='brute',n_jobs=1).fit_predict(cpms.reshape(len(cpms),-1))
    if (labels<0).any():raise ValueError('Unassigned DBSCAN client; no invented noise policy')
    matrices=np.stack([cpms[labels==k].mean(axis=0) for k in range(int(labels.max())+1)])
    return labels,matrices


def solve_cluster_weights(cluster_cpms, maxiter=10000):
    """Nonnegative Frobenius objective, before normalizing client coefficients."""
    matrices=validate_cpms(cluster_cpms)
    if type(maxiter) is not int or maxiter<1:raise ValueError('Invalid NNLS iteration cap')
    a=matrices.reshape(len(matrices),-1).T;b=np.eye(matrices.shape[1]).reshape(-1)
    alpha,residual=nnls(a,b,maxiter=maxiter)
    if not np.isfinite(alpha).all() or (alpha<0).any() or not np.isfinite(residual):
        raise ValueError('Invalid NNLS solution')
    gradient=a.T@(a@alpha-b)
    projected=np.where(alpha>1e-10,gradient,np.minimum(gradient,0))
    kkt=float(np.max(np.abs(projected)))
    tolerance=1e-7*(1+float(np.max(np.abs(a.T@b))))
    if kkt>tolerance:raise ValueError('NNLS optimality check failed')
    return alpha,float(residual),kkt


def client_weights(labels, alpha):
    """Paper Eq. 4, then Algorithm 1 line 14. Do not multiply by sample counts."""
    labels=np.asarray(labels);alpha=np.asarray(alpha,dtype=np.float64)
    if (labels.ndim!=1 or not len(labels) or labels.dtype.kind not in 'iu' or alpha.ndim!=1
            or not len(alpha) or not np.isfinite(alpha).all() or (alpha<0).any()
            or (labels<0).any() or (labels>=len(alpha)).any()):raise ValueError('Invalid cluster coefficients')
    sizes=np.bincount(labels,minlength=len(alpha))
    if (sizes==0).any():raise ValueError('Cluster coefficients include an empty cluster')
    beta=alpha[labels]/sizes[labels];total=float(beta.sum())
    if not np.isfinite(total) or total<=0:raise ValueError('All cluster coefficients are zero; no fallback')
    return beta/total


class FedMADEAggregator:
    """Same aggregate(results) contract as FedAvg; auxiliary models evaluated serially."""
    def __init__(self, model_factory, auxiliary_dataset, num_classes, *, eps=.5,
                 batch_size=128, maxiter=10000, device='cpu', diagnostic_dir=None):
        if type(num_classes) is not int or num_classes<2:raise ValueError('Invalid class count')
        if type(batch_size) is not int or batch_size<1:raise ValueError('Invalid auxiliary batch size')
        labels=auxiliary_dataset.tensors[1]
        if labels.ndim!=1 or labels.dtype!=torch.int64 or ((labels<0)|(labels>=num_classes)).any():
            raise ValueError('Invalid auxiliary labels')
        self.support=torch.bincount(labels,minlength=num_classes).cpu().numpy()
        if (self.support==0).any():raise ValueError('Missing true auxiliary class')
        self.loader=DataLoader(auxiliary_dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                               generator=torch.Generator().manual_seed(0))
        with torch.random.fork_rng(devices=[]):self.model=model_factory().to(device)
        self.num_classes,self.eps,self.maxiter,self.device=num_classes,eps,maxiter,device
        self.diagnostic_dir=None if diagnostic_dir is None else Path(diagnostic_dir)
        self.last_diagnostics=None

    def aggregate(self, results):
        if not results:raise ValueError('Cannot aggregate no clients')
        if len({r.client_id for r in results})!=len(results):raise ValueError('Duplicate client contributions')
        if len({r.round_id for r in results})!=1:raise ValueError('Mixed communication rounds')
        if any(type(r.round_id) is not int or r.round_id<1 or type(r.sample_count) is not int or r.sample_count<1 for r in results):
            raise ValueError('Invalid round/sample count')
        reference=results[0].state
        for r in results:validate_state(r.state,reference)
        ordered=sorted(results,key=lambda r:r.client_id);cpms=[]
        for r in ordered:
            self.model.load_state_dict(r.state,strict=True)
            cpms.append(class_probability_matrix(self.model,self.loader,self.num_classes,self.device))
        cpms=np.stack(cpms);labels,ccpms=cluster_cpms(cpms,self.eps)
        alpha,residual,kkt=solve_cluster_weights(ccpms,self.maxiter);beta=client_weights(labels,alpha)
        weights={r.client_id:float(w) for r,w in zip(ordered,beta)}
        state=OrderedDict()
        for key,value in reference.items():
            accumulator=torch.zeros_like(value,dtype=torch.float64,device='cpu')
            for r in results:accumulator.add_(r.state[key].detach().cpu().to(torch.float64),alpha=weights[r.client_id])
            state[key]=accumulator.to(value.dtype)
        validate_state(state,reference)
        diagnostics={'round_id':results[0].round_id,'client_ids':[r.client_id for r in ordered],
            'cluster_labels':labels.tolist(),'cluster_count':len(alpha),'cluster_sizes':np.bincount(labels).tolist(),
            'raw_alpha':alpha.tolist(),'normalized_cluster_mass':(alpha/alpha.sum()).tolist(),
            'client_weights':weights,'auxiliary_class_counts':self.support.tolist(),
            'objective_frobenius':residual,'projected_kkt_residual':kkt,
            'solver':'scipy_nnls','objective':'nonnegative_frobenius_then_client_normalization','eps':self.eps,
            'min_samples':1,'norm':'raw_flattened_euclidean','adapted_not_exact_reproduction':True}
        self.last_diagnostics=diagnostics
        if self.diagnostic_dir is not None:
            self.diagnostic_dir.mkdir(parents=True,exist_ok=True);prefix=self.diagnostic_dir/f'round_{results[0].round_id:03d}'
            with prefix.with_suffix('.json').open('x') as stream:json.dump(diagnostics,stream,indent=2,allow_nan=False)
            with prefix.with_suffix('.npz').open('xb') as stream:
                np.savez(stream,client_ids=np.array(diagnostics['client_ids']),cpms=cpms,ccpms=ccpms,
                         labels=labels,alpha=alpha,beta=beta,auxiliary_counts=self.support)
            with prefix.with_suffix('.pt').open('xb') as stream:
                torch.save({'round_id':results[0].round_id,'client_order':[r.client_id for r in results],
                    'client_states':{r.client_id:clone_state(r.state) for r in results},'aggregated_state':clone_state(state)},stream)
        return state,weights
