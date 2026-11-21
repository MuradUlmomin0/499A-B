"""Controlled local label poisoning; evaluator truth never enters ClientResult."""
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def select_malicious(client_ids,seed,count):
    ids=sorted(client_ids)
    if len(set(ids))!=len(ids) or type(count) is not int or not 0<=count<=len(ids):raise ValueError('Invalid malicious-client selection')
    return random.Random(seed+5005).sample(ids,count)


def original_labels(dataset):
    if hasattr(dataset,'tensors'):labels=dataset.tensors[1]
    else:labels=torch.stack([dataset[i][1] for i in range(len(dataset))])
    if labels.dtype!=torch.int64 or labels.ndim!=1:raise ValueError('Labels must be an int64 vector')
    return labels.detach().cpu().numpy()


def label_hash(dataset):return hashlib.sha256(np.asarray(original_labels(dataset),dtype='<i8').tobytes()).hexdigest()


class LabelPoisonView(Dataset):
    """Return original feature storage and fresh changed labels, with access proof."""
    def __init__(self,dataset,target,destination,num_classes):
        if (type(target) is not int or type(destination) is not int or not 0<=target<num_classes
                or not 0<=destination<num_classes or target==destination):raise ValueError('Invalid target/destination labels')
        self.dataset,self.target,self.destination=dataset,target,destination
        self.observed_labels=np.full(len(dataset),-1,dtype=np.int64)
        self.access_counts=np.zeros(len(dataset),dtype=np.int32)

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        features,label=self.dataset[index]
        if not isinstance(label,torch.Tensor) or label.dtype!=torch.int64 or label.ndim!=0:raise ValueError('Expected scalar int64 label')
        value=int(label)
        output=self.destination if value==self.target else value
        self.observed_labels[index]=output;self.access_counts[index]+=1
        return features,torch.tensor(output,dtype=torch.int64) if value==self.target else label


class PoisonedClient:
    """Compose the existing client; train normally except for an active label view."""
    def __init__(self,client,attack,selected,num_classes,audit_dir):
        self.client=client;self.client_id=client.client_id;self.attack=dict(attack)
        self.selected=selected;self.num_classes=num_classes;self.events=[]
        self.directory=Path(audit_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.source_hash=label_hash(client.train_dataset)
        self.target_rows=int(np.count_nonzero(original_labels(client.train_dataset)==attack['target_class']))

    def receive_global(self,state):self.client.receive_global(state)

    def train(self,round_id):
        a=self.attack;source=self.client.train_dataset
        active=bool(a['enabled'] and self.selected and a['start_round']<=round_id<=a['end_round'])
        view=LabelPoisonView(source,a['target_class'],a['destination_class'],self.num_classes) if active else None
        try:
            if view is not None:self.client.train_dataset=view
            result=self.client.train(round_id)
        finally:self.client.train_dataset=source
        if label_hash(source)!=self.source_hash:raise ValueError('Original local labels changed during training')
        audit_file=None;flipped=0;observations=0
        if view is not None:
            if not np.all(view.access_counts==self.client.training['local_epochs']):raise ValueError('Incomplete or repeated local record visits')
            original=original_labels(source);expected=np.where(original==a['target_class'],a['destination_class'],original)
            if not np.array_equal(view.observed_labels,expected):raise ValueError('Delivered training labels differ from attack definition')
            flipped=int(np.count_nonzero(expected!=original));observations=int(view.access_counts[expected!=original].sum())
            audit_file=f'round_{round_id:03d}_{self.client_id}.npz'
            with (self.directory/audit_file).open('xb') as stream:
                np.savez_compressed(stream,observed_labels=view.observed_labels,access_counts=view.access_counts)
        self.events.append(dict(round_id=round_id,client_id=self.client_id,selected_malicious=bool(self.selected),active_attack=active,
            sample_count=len(source),local_epochs=self.client.training['local_epochs'],target_rows=self.target_rows,
            flipped_unique_rows=flipped,flipped_observations=observations,audit_file=audit_file))
        return result
