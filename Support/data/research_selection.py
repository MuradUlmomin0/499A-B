"""Approved proportional quotas and whole-input-group simulated clients."""
from collections import defaultdict
import hashlib
import numpy as np


def proportional_quotas(counts, total):
    """Exact largest-remainder row quotas; ascending class ID breaks ties."""
    counts=np.asarray(counts,dtype=np.int64)
    if counts.ndim!=1 or (counts<0).any() or type(total) is not int or total<1 or total>counts.sum():
        raise ValueError('Invalid proportional sampling population or budget')
    population=int(counts.sum())
    numerators=[int(n)*total for n in counts]
    quota=np.array([n//population for n in numerators],dtype=np.int64)
    order=sorted(range(len(counts)),key=lambda i:(-(numerators[i]%population),i))
    quota[order[:total-int(quota.sum())]]+=1
    return quota


def allocate_groups(groups, labels, clients, seed, condition, alpha):
    """PCG64(seed), ascending class IDs and SHA-256 seeded group order.

    IID-style minimizes per-class row counts, then total rows, then client ID.
    Dirichlet draws one symmetric class vector, then one categorical owner per
    whole group. No resampling for empty clients and no per-client class floors.
    """
    if condition not in ('iid','dirichlet') or clients<1 or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('Invalid group allocation settings')
    if len(groups)!=len(labels) or not len(groups): raise ValueError('Invalid aligned groups/labels')
    members=defaultdict(list); group_label={}
    for i,(g,y) in enumerate(zip(groups,labels)):
        if g in group_label and group_label[g]!=int(y): raise ValueError('Conflicting group labels')
        group_label[g]=int(y); members[g].append(i)
    classes=defaultdict(list)
    for g,y in group_label.items(): classes[y].append(g)
    rng=np.random.Generator(np.random.PCG64(seed))
    totals=np.zeros(clients,dtype=np.int64); assignment=np.empty(len(groups),dtype=np.int64)
    for y in sorted(classes):
        ordered=sorted(classes[y],key=lambda g:(hashlib.sha256(f'partition:{seed}:{g}'.encode()).digest(),g))
        class_rows=np.zeros(clients,dtype=np.int64)
        probabilities=rng.dirichlet(np.full(clients,alpha)) if condition=='dirichlet' else None
        for g in ordered:
            owner=(int(rng.choice(clients,p=probabilities)) if probabilities is not None else
                   min(range(clients),key=lambda j:(class_rows[j],totals[j],j)))
            indices=members[g]; assignment[indices]=owner
            class_rows[owner]+=len(indices); totals[owner]+=len(indices)
    if (totals==0).any(): raise ValueError('Allocation produced an empty client; no retry')
    return assignment
