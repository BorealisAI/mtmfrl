# Multi Type Mean Field  Reinforcement Learning 

Implementation of MTMFQ in the paper [Multi Type Mean Field Reinforcement Learning](https://arxiv.org/pdf/2002.02513.pdf).


The environments contain 4 teams training and fighting against each other. Multi Battle Game environment has four teams with 72 agents each. 
 
## Code structure

- See folder Multibattle for training and testing environments of the Multi battle environment. 

- See folder Multigather for training and testing environments of the Multi gather environment. 

- See folder Predatorprey for training and testing environments of the Predaotor Prey environment. 

### In each of these three game directories, the files most relevant to our research are:

- /mfrl/examples/battle_model/python/magent/builtin/config/battle.py: Script that defines the rewards for the different games and is game-dependent.

- /mfrl/examples/battle_model/senario_battle.py: Script to run the training and testing, which other scripts call into, and is game-dependent.

- /mfrl/train_battle.py: Script to begin training the game for 2000 iterations and almost identical across games. The algorithm can be specified as a parameter (MFAC, MFQ, IL, or MTMFQ).

- /mfrl/battle.py: Script to run comparative testing (creating 4 types of agents, where each type is one of the 4 algorithms) and is almost identical across games.

- /mfrl/examples/battle_model/algo: This directory contains the learning algorithms, and is identical across all three games.





## Instructions for Ubuntu

### Requirements

Atleast 

- `python==3.6.1`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

- `gym==0.9.2`


```shell
pip install gym
```

- `scikit-learn==0.22.0`


```shell
sudo pip install scikit-learn
```


- `tensorflow 2`


[Check Documentation](https://www.tensorflow.org/install).


- `libboost libraries`


```shell
sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
```
 

### Clone the repository

```shell
git clone https://github.com/BorealisAI/mtmfrl
```

### Build the MAgent framework 

```shell
cd mtmfq/multibattle/mfrl/examples/battle_model
./build.sh
```

Similarly change directory and build for multigather and predatorprey folders for those testbeds. 

### Training and Testing

```shell
cd mtmfq/multibattle/mfrl
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
python3 train_battle.py --algo mtmfq
```

Run file battle.py for running the test battles. 

For more help, look at the instrctions in [MAgent](https://github.com/geek-ai/MAgent) and [MFRL](https://github.com/mlii/mfrl)




## Instructions for OSX


### Clone the repository

```shell
git clone https://github.com/BorealisAI/mtmfrl
```

### Install dependencies

```shell
cd mtmfq
brew install cmake llvm boost@1.55
brew install jsoncpp argp-standalone
brew tap david-icracked/homebrew-websocketpp
brew install --HEAD david-icracked/websocketpp/websocketpp
brew link --force boost@1.55
```


### Build MAgent Framework 

```shell
cd mtmfq/multibattle/mfrl/examples/battle_model
./build.sh
```

Similarly change directory and build for multigather and predatorprey folders for those testbeds. 

### Training and Testing

```shell
cd mtmfq/multibattle/mfrl
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
python3 train_battle.py --algo mtmfq
```

Run file battle.py for running the test battles. 

For more help, look at the instrctions in [MAgent](https://github.com/geek-ai/MAgent) and [MFRL](https://github.com/mlii/mfrl)


## Note

This is research code and will not be actively maintained. Please send an email to ***s2ganapa@uwaterloo.ca*** for questions or comments. 


## Paper citation

If you found it helpful, please cite the following paper:

<pre>

@InProceedings{Srirammtmfrl2020,
  title = 	 {Multi Type Mean Field Reinforcement Learning},
  author = 	 {Ganapathi Subramanian, Sriram and Poupart, Pascal, and Taylor, Matthew E. and Hegde, Nidhi}, 
  booktitle = 	 {Proceedings of the Autonomous Agents and Multi Agent Systems (AAMAS 2020)},
  year = 	 {2020},
  address = 	 {Auckland, New Zealand},
  month = 	 {9--13 May},
  publisher = 	 {IFAAMAS}
}

</pre>
