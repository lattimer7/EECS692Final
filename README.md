# BabyDungeon: Multi-Agent Collaboration to Solve Escape-Room Style Tasks

This repo contains the PyTorch implementation for BabyDungeon, with framework taken from Learning to Ground Multi-Agent Communication with Autoencoders.

### Code layout

Please see each sub-directory for more details.


| Directory          | Detail |
| :-------------: |:-------------:|
| cifar-game | environment and models for training "CIFAR Game" |
| :-------------: |:-------------:|
| marl-grid/env | environments for training "FindGoal" and "RedBlueDoors" | 
| marl-grid/find-goal | models for training "FindGoal" |
| marl-grid/red-blue-doors | models for training "RedBlueDoors" | 

### Paper citation

If you used this code or found our work helpful, please consider citing:

<pre>
@misc{lin2021learning,
      title={Learning to Ground Multi-Agent Communication with Autoencoders}, 
      author={Toru Lin and Minyoung Huh and Chris Stauffer and Ser-Nam Lim and Phillip Isola},
      year={2021},
      eprint={2110.15349},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
