## World model learning
python run_worldModel_IPS.py --env KuaiEnv-v0 --seed 0 --cuda 0 --loss "pointneg" --message "DeepFM-IPS"
python run_linUCB.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 --cuda 0 --loss "pointneg" --message "UCB"
python run_epsilongreedy.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 --cuda 0 --loss "pointneg" --message "epsilon-greedy"
python run_worldModel_ensemble.py --env KuaiEnv-v0 --cuda 0 --epoch 5 --loss "pointneg" --message "pointneg"

## Policy training
## Model-free
python run_Policy_SQN.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --message "SQN"
python run_Policy_CRR.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --message "CRR"
python run_Policy_CQL.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "pointneg" --message "CQL"
python run_Policy_BCQ.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg" --message "BCQ"
## Model-based
python run_Policy_IPS.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "DeepFM-IPS" --message "IPS"
python run_Policy_Main.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0 --lambda_entropy 0 --window_size 3 --read_message "pointneg" --message "MBPO"
python run_Policy_Main.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0 --window_size 3 --read_message "pointneg" --message "MOPO"
python run_Policy_Main.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5 --window_size 3 --read_message "pointneg" --message "DORL"

## Our ROLeR
python run_Policy_Main.py --env KuaiEnv-v0 --seed 0 --cuda 1 --num_leave_compute 1 --leave_threshold 0 --which_tracker att --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5 --window_size 3 --read_message "pointneg" --message "ROLeR" --remark std --scratch True --change_pred_reward True --change_var True --kr 30 --ku 30