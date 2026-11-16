# 1. Download + tokenize + split (creates processed_linux_cmd/)
python data_prep.py

# 2. Sanity checks (init-loss, padding invariance, causality, overfit-one-batch)
python train.py --check

# 3. Train
python train.py --epochs 40 --batch_size 64

# 4. Resume
python train.py --resume runs/linux_cmd_v1/best.pt
