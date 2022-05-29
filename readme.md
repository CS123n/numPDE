# MultiGrid Method

Hi, everyone! This repo provides a parallel version of 2-D multigrid method based on Pytorch.

Install necessary python packages:
```bash
pip install -r requirements.txt
```

Run single node version:
```bash
python main.py -n 64 -p 1
```

Run multi-node version:
```bash
torchrun --nproc_per_node 4 main.py -n 64 -p 2
```