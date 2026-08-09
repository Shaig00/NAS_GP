import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from Neural_Architecture_Search.Framework import NAS_Framework

if __name__ == '__main__':
    nas_framework = NAS_Framework(embedding_dimension=100,
                                max_number_episodes=1000, timesteps_per_episode=10,
                                ac_units=500, ac_lr=0.0007)
    nas_framework.create_search_space(print_components=True)
    nas_framework.run_macro_nas(entropy_baseline=False,
                               save_sampling_history=True)