"""Append-only CSVs and collision-safe experiment directories."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import torch
import yaml


class ExperimentLogger:
    def __init__(self, output_dir, name, config, metadata):
        root = Path(output_dir)
        root.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.path = root / f'{name}_{stamp}_{uuid4().hex[:8]}'
        self.path.mkdir(exist_ok=False)
        self.finished = False
        self.testing_only = bool(metadata.get('testing_only',False))
        with (self.path/'config.yaml').open('x') as stream:
            yaml.safe_dump(config,stream,sort_keys=False)
        self._json('metadata.json',metadata)
        self.columns = {
            'round_metrics.csv': ['round_id','selected_clients','round_duration','aggregation_duration','loss','accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','per_class','confusion_matrix'],
            'client_metrics.csv': ['round_id','client_id','sample_count','train_loss'],
            'aggregation_weights.csv': ['round_id','client_id','sample_count','weight'],
        }
        for filename,columns in self.columns.items():
            with (self.path/filename).open('x',newline='') as stream:
                csv.DictWriter(stream,fieldnames=columns).writeheader()

    def _json(self, filename, value):
        with (self.path/filename).open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)

    def _append(self, filename, row):
        encoded={key:json.dumps(value,allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()}
        with (self.path/filename).open('a',newline='') as stream:
            csv.DictWriter(stream,fieldnames=self.columns[filename]).writerow(encoded)

    def log_round(self, record, results, weights):
        if self.finished:
            raise ValueError('Run already finalized')
        self._append('round_metrics.csv',record)
        for result in results:
            self._append('client_metrics.csv',{'round_id':result.round_id,'client_id':result.client_id,
                         'sample_count':result.sample_count,'train_loss':result.metrics['train_loss']})
            self._append('aggregation_weights.csv',{'round_id':result.round_id,'client_id':result.client_id,
                         'sample_count':result.sample_count,'weight':weights[result.client_id]})

    def finish(self, metrics, checkpoint):
        if self.finished:
            raise ValueError('Run already finalized')
        self._json('final_metrics.json',{'testing_only':self.testing_only,'metrics':metrics})
        checkpoint = {**checkpoint,'testing_only':self.testing_only}
        with (self.path/'checkpoint.pt').open('xb') as stream:
            torch.save(checkpoint,stream)
        self.finished = True
