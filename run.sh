python sl_train_dl/eval.py --ckpt_path ckpt/AI_12_192.pth
python sl_train_rl/run_rl.py --ckpt_path ckpt/AI_12_192.pth --num_parallel 32 --minibatch 32
python sl_train_rl/run_rl.py --ckpt_path ckpt/AI_12_192.pth --num_parallel 4 --minibatch 4 --max_step 250