import torch
from pathlib import Path
DATA_DIR = Path('experiments') / 'chair'
OUTPUT_DIR = Path('experiments') / 'logs'


def mkdir(p):
    if not p.exists():
        p.mkdir()


mkdir(OUTPUT_DIR)
num_examples = 10
for i in range(0, 100, 25):
    p = OUTPUT_DIR / str(i)
    mkdir(p)
    for j in range(num_examples):
        item = torch.load(DATA_DIR / str(i) / f'{j}.pt')
        chairs = item['pcd']
        betas = item['beta']
        for k in range(num_examples):
            index = j * num_examples + k
            torch.save({
                'pcd': chairs[k],
                'betas': betas
            }, p / f'{index}.pt', _use_new_zipfile_serialization=False)
