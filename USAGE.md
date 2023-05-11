```

salloc --gres=gpu:rtx_3090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=50G -A visualai
# a6000
salloc --gres=gpu:a6000:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=50G -A allcs
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install -r recurrent-vln-bert.yml
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install protobuf==3.20.*
/n/fs/kl-project/conda_env/envs/bert_vln/bin/pip install jsonlines
/n/fs/kl-project/conda_env/envs/bert_vln/bin/pip install h5py
/n/fs/kl-project/conda_env/envs/bert_vln/bin/pip install -r requirements.txt
/n/fs/kl-project/conda_env/envs/bert_vln/bin/pip install transformers==4.12.3
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install opencv-python
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install git+https://github.com/huggingface/transformers
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install progressbar2
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install --upgrade --quiet jupyter_client ipywidgets
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install accelerate
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install --upgrade accelerate bitsandbytes

/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

export PYTHONPATH=.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/fs/kl-project/libcrypt/usr/lib64
export LDFLAGS="-L//n/fs/kl-project/libcrypt/usr/lib64"
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
      --resume_file ../datasets/R2R/trained_models/vitbase-finetune/ckpts/best_val_unseen \
      --test --submit
      
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
--resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/best_val_unseen \
--test --submit --no_cand_backtrack

CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
--resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/best_val_unseen \
--test --submit --ob_type cand

CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
--resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen \
--test --submit
      
cmake -DPYTHON_EXECUTABLE:FILEPATH=/n/fs/kl-project/conda_env/envs/vlnhamt/bin/python3.8 ..


python -m ipykernel install --user --name vlnhamt --display-name "Python (vlnhamt)"

python download_mp.py -o base_dir --type undistorted_camera_parameters
```

```
Prompt:
You are an AI agent tasked with navigating a virtual room based on a set of instructions provided by a user. 
At each step, you will be presented with a series of views that the user observed and the actions they took. 
Your objective is to determine whether the user correctly followed the instructions and stopped at the correct location. 
If not, you should identify which step needs to be revisited and explain why.


```

Mask visit + Modify history
Env name: val_unseen, steps: 5.82, lengths: 11.37, nav_error: 3.57, oracle_error: 2.25, sr: 67.22, oracle_sr: 75.14, spl: 61.84, nDTW: 69.41, SDTW: 57.95, CLS: 68.21


