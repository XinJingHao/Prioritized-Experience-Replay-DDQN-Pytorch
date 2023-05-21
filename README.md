# Prioritized-DQN-DDQN-Pytorch

A clean and robust implementation of Prioritized Experience Replay (PER) with DQN/DDQN. 

Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).


<br/>
<br/>

Here are three versions of PER :

+ **Version 1: PriorDQN_gym0.1x**

  Implemented with gym version *0.1x*, where ***s_next, a, r, done, info = env.step(a)***

  Prioritized sampling is realized by ***sum-tree***

  ```python
  # Dependencies of PriorDQN_gym0.1x
  gym==0.19.0
  numpy==1.21.6
  pytorch==1.11.0
  tensorboard==2.13.0
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.1x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.1x/IMGs/LLDV2.svg" width="320" height="200"> |

<br/>
<br/>



+ **Version 2: PriorDQN_gym0.2x**

  Implemented with gym version *0.2x*, where ***s_next, a, r, terminated, truncated, info = env.step(a)***

  Prioritized sampling is realized by ***sum-tree***

  ```python
  # Dependencies of PriorDQN_gym0.2x
  gymnasim==0.28.1 
  numpy==1.24.3  
  pytorch==2.0.1 
  tensorboard==2.13.0
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.2x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.2x/IMGs/LLDV2.svg" width="320" height="200"> |

<br/>
<br/>



+ **Version 3: LightPriorDQN_gym0.2x**

  An optimized version of PriorDQN_gym0.2x,

  where prioritized sampling is realized by ***torch.multinomial()***, which is 3X faster than sum-tree.

  ```python
  # Dependencies of LightPriorDQN_gym0.2x
  gymnasim==0.28.1 
  numpy==1.24.3  
  pytorch==2.0.1 
  tensorboard==2.13.0
  ```

  |                           CartPole                           |                         LunarLander                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/CPV1.svg" width="320" height="200"> | <img src="https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/LightPriorDQN_gym0.2x/IMGs/LLDV2.svg" width="320" height="200"> |


<br/>
<br/>

## How to use my code

### Train from scratch

```bash
cd PriorDQN_gym0.1x # or PriorDQN_gym0.2x, LightPriorDQN_gym0.2x

python main.py
```

where the default enviroment is CartPole-v1.  

<br/>

### Play with trained model

```bash
cd PriorDQN_gym0.1x # or PriorDQN_gym0.2x, LightPriorDQN_gym0.2x

python main.py --write False --render True --Loadmodel True --ModelIdex 50000
```

<br/>

### Change Enviroment

If you want to train on different enviroments

```bash
cd PriorDQN_gym0.1x # or PriorDQN_gym0.2x, LightPriorDQN_gym0.2x

python main.py --EnvIdex 1
```

The --EnvIdex can be set to be 0 and 1, where   

```bash
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
```

<br/>

### Visualize the training curve

You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'

<br/>

### Hyperparameter Setting

For more details of Hyperparameter Setting, please check 'main.py'

<br/>

### References

PER: Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.

DQN: Mnih V , Kavukcuoglu K , Silver D , et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013. 

Double DQN: Hasselt H V , Guez A , Silver D . Deep Reinforcement Learning with Double Q-learning[J]. Computer ence, 2015.

  
