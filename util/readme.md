Created at 2024 Feb. 22

- The original DORL's util folder only contains utils.py;
- Now we add one folder (vis_explore_before_kdd2024) and several python files:
    - The folder contains 
        - the visualization about the distribution of **item embedding**, **state distribution**, **user embedding**, **intro_plot**, **rew_kuairec** (the up-to-date comprehensive visualization about the GT reward and DeepFM estimated reward on KuaiRec)
        - the exploration about **knn_on_kuairand**, **knn_on_kuairec**, the relationship between the uncertainty and knn's k (**uncertainty_k**)
        - Note: Basically, the final version in each sub-folder has the largest version number; Strictly, you need to look into the comments with *Don YZ*
        - The contents in this folder are only for reference and used for kdd2024 Feb.
    - The python files
        - knn_cluster4coat.py
        - knn_cluster4kuairand.py (original knn_cluster4kuairand_v2.py)
        - knn_cluster4kuairec.py (original knn_clustering_v1.py)
        - vis_in_intro (original vis_in_intro_v2.py)
        - vis_reward_kuairec.py (original vis_reward.py)