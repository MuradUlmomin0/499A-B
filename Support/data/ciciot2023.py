"""Stream the audited CSV export; folder labels remain a declared assumption."""
from collections import Counter
import json
import warnings
from zipfile import ZipFile
import numpy as np
import pandas as pd

CLASS_NAMES = ('Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-based', 'Brute Force')
FOLDER_TO_CLASS = {'Benign_Final': 'Benign', 'DictionaryBruteForce': 'Brute Force'}
for _prefix, _names in {
    'DDoS': ['ACK_Fragmentation', 'HTTP_Flood', 'ICMP_Flood', 'ICMP_Fragmentation', 'PSHACK_FLOOD',
             'RSTFINFLOOD', 'SlowLoris', 'SynonymousIP_Flood', 'SYN_Flood', 'TCP_Flood', 'UDP_Flood', 'UDP_Fragmentation'],
    'DoS': ['HTTP_Flood', 'SYN_Flood', 'TCP_Flood', 'UDP_Flood'],
    'Mirai': ['greeth_flood', 'greip_flood', 'udpplain'],
}.items():
    FOLDER_TO_CLASS.update({f'{_prefix}-{name}': _prefix for name in _names})
FOLDER_TO_CLASS.update(dict.fromkeys(['Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PingSweep',
                                    'Recon-PortScan', 'VulnerabilityScan'], 'Recon'))
FOLDER_TO_CLASS.update(dict.fromkeys(['DNS_Spoofing', 'MITM-ArpSpoofing'], 'Spoofing'))
FOLDER_TO_CLASS.update(dict.fromkeys(['Backdoor_Malware', 'BrowserHijacking', 'CommandInjection',
                                    'SqlInjection', 'Uploading_Attack', 'XSS'], 'Web-based'))
SUBTYPES = tuple(sorted(FOLDER_TO_CLASS))


def stream_chunks(config, quarantine_stream, statistics):
    """Yield finite float32 chunks plus original source-member/row lineage.

    No class balancing, scaling or splitting. Incomplete rows padded by pandas,
    NaN/inf and values overflowing float32 are logged and excluded. Unknown
    source schema/folders are errors. Input member paths are never extracted.
    """
    statistics.update(rows=0, invalid_rows=0, valid_counts=dict.fromkeys(CLASS_NAMES, 0), members=[])
    with ZipFile(config['source']['archive_path']) as archive:
        members = sorted((i for i in archive.infolist() if i.filename.lower().endswith('.csv')),
                         key=lambda i: i.filename)
        if not members or len({i.filename for i in members}) != len(members):
            raise ValueError('Require uniquely named CSV members')
        for info in members:
            pieces = info.filename.split('/')
            if len(pieces) != 3 or pieces[0] != 'CSV' or pieces[1] not in FOLDER_TO_CLASS:
                raise ValueError(f'Unknown source folder/layout: {info.filename}')
        statistics['members'] = [{'path': i.filename, 'bytes': i.file_size, 'crc32': f'{i.CRC:08x}'} for i in members]
        for file_index, info in enumerate(members):
            folder = info.filename.split('/')[1]
            label_name = FOLDER_TO_CLASS[folder]
            label, subtype = CLASS_NAMES.index(label_name), SUBTYPES.index(folder)
            offset = 0
            with archive.open(info) as stream, warnings.catch_warnings():
                warnings.simplefilter('error', pd.errors.ParserWarning)
                try:
                    with pd.read_csv(stream, dtype=np.float64, chunksize=config['pilot']['chunk_rows'],
                                     encoding='utf-8-sig', skip_blank_lines=False, on_bad_lines='error') as reader:
                        for frame in reader:
                            if list(frame.columns) != config['source']['feature_names'] or not isinstance(frame.index, pd.RangeIndex):
                                raise ValueError(f'Unexpected header/index: {info.filename}')
                            raw = frame.to_numpy(copy=False)
                            with np.errstate(over='ignore', invalid='ignore'):
                                values = raw.astype('<f4')
                            valid = np.isfinite(raw).all(axis=1) & np.isfinite(values).all(axis=1)
                            ordinals = np.arange(offset, offset + len(frame), dtype=np.uint64)
                            statistics['rows'] += len(frame)
                            statistics['invalid_rows'] += int((~valid).sum())
                            for ordinal in ordinals[~valid]:
                                quarantine_stream.write(json.dumps({'member': info.filename, 'file_index': file_index,
                                    'row_ordinal': int(ordinal), 'reason': 'nonfinite_or_incomplete_or_float32_overflow'}) + '\n')
                            statistics['valid_counts'][label_name] += int(valid.sum())
                            offset += len(frame)
                            values = values[valid]
                            values[values == 0] = 0  # Canonical +0 and -0 group together.
                            if len(values):
                                yield {'features': values, 'row_ordinals': ordinals[valid],
                                       'file_index': file_index, 'label': label, 'subtype': subtype}
                except (ValueError, pd.errors.ParserWarning) as error:
                    raise ValueError(f'CSV source {info.filename}, at/after data-row offset {offset}: {error}') from error
