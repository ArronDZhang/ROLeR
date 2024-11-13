# ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems

This repository provides the official PyTorch implementation, reproduction and experiment logs for the paper titled "ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems." Within this paper, experiments are conducted on four RL environments: ```KuaiEnv-v0```, ```KuaiRand-v0```, ```CoatEnv-v0``` and ```Yahoo-v0```, whose introduction can be found at [KuaiRec](https://kuairec.com/), [KuaiRand](https://kuairand.com/), [Coat](https://www.cs.cornell.edu/~schnabts/mnar/), and [Yahoo](https://dl.acm.org/doi/10.1145/1639714.1639717).

More details can be found in our [paper](https://arxiv.org/abs/2407.13163). The authors are Yi Zhang, Ruihong Qiu, Jiajun Liu, and Sen Wang.



## Installation

1. Clone this repo and create a new virtual environment with:

   ```shell
   git clone https://github.com/ArronDZhang/ROLeR.git && cd ROLeR
   conda create --name roler python=3.10 -y
   ```

2. Activate the created environment and install the requirements with:

   ```bash
   conda activate roler
   sh install.sh
   ```

3. Install the tianshou package from [DORL](https://github.com/chongminggao/DORL-codes)'s forked version: 

   ```bash
   cd src
   git clone https://github.com/chongminggao/tianshou.git
   git reset --hard 0f59e38
   cd ..
   ```



## Dataset

1. Download the datasets used in our work:

   ```bash
   wget https://chongming.myds.me:61364/DORL/environments.tar.gz
   ```

   Or you can download them manually from [here](https://rec.ustc.edu.cn/share/9fe264f0-ae09-11ed-b9ef-ed1045d76757).

2. Uncompress the downloaded `environments.tar.gz` and put the files in ```ROLeR/```:

   ```bash
   tar -zxvf environments.tar.gz
   ```



## Reproduce

To reproduce the results, we have two main steps. In the first step, we train world models (DeepFM) to provide user and item embeddings for offline model-free RL methods, as well as uncertainty penalties (e.g., MOPO), entropy penalties (e.g., DORL), and reward models for offline model-based RL methods. In the second step, the recommendation policies are trained.

### Step 1: World Model Learning 

```bash
python run_worldModel_IPS.py --env KuaiEnv-v0 --seed 0 --cuda 0 --loss "pointneg" --message "DeepFM-IPS"
python run_linUCB.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 --cuda 0 --loss "pointneg" --message "UCB"
python run_epsilongreedy.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 --cuda 1 --loss "pointneg" --message "epsilon-greedy"
python run_worldModel_ensemble.py --env KuaiEnv-v0 --cuda 0 --epoch 5 --loss "pointneg" --message "pointneg"
```

Note: the above commands are exammplified with ```KuaiEnv-v0```. To train the world models for KuaiRand, Coat and Yahoo, you just need to change the argument of ```--env``` to ```KuaiRand-v0```, ```CoatEnv-v0``` and ```Yahoo-v0```, respectively.