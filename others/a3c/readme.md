# 华为项目



## Benchmarking

由于CES的源代码采用的是**NoFrameskip-v4版本的atari游戏，所有A3C算法的复现和baseline可能需要重新运行。

环境初始化参数

1. frameskip， 默认值 (2, 5)，表示取值在2到5之间

2. repeat_action_probability，默认值0. 

   >With probability p (default: p = 0:25), the previously executed action is executed again during the next frame, ignoring the agent's actual choice.

3. max_episode_steps (单个episode的最大步数，如何计算这个最大步数？)

atari环境之间的区别：

1. {}-v0, 
   1. frameskip 为默认值即2-5
   2. repeat_action_probability=0.25
   3. max_episode_steps=10000,
2. {}-v4
   1. frameskip 为默认值即2-5
   2. repeat_action_probability=0.
   3. max_episode_steps=100000,
3. {}NoFrameskip-v0
   1. frameskip=1
   2. repeat_action_probability=0.25
   3. max_episode_steps=100000,
4. {}NoFrameskip-v4
   1. frameskip=1
   2. repeat_action_probability=0.
   3. max_episode_steps=100000,