# Deep Q-Learning

Article: Mnih, et al. Playing atari with deep reinforcement learning. In NIPS Deep Learning Workshop. 2013.

## Intuition

结合深度神经网络在模式识别上的能力，做一个端到端的强化学习算法


## Model Architecture


- CNN1, 16 8x8 filters, stride=4, input=4x84x84, output=16x20x20
- ReLU
- CNN2, 32 4x4 filters, stride=2, input=16x20x20, output=32x9x9
- ReLU
- FC1, input: 32x9x9, output:256
- ReLU
- Output layer: input 256, output: the number of actions (no softmax)

## Preprocessing

[Mnih et al. 2013]

input: 210x160
1. RGB to gray-scale
2. down-sampling to a 110x84
3. crop an 84x84 region of the image

[Mnih et al. 2015]
input: 210x160
1. encode a single frame: take the maximum value for each pixel colour value over the frame being encoded and the previous frame. like np.maximum(now_frame, previou_frame)
benefit: remove flicking
why flicking? some objects only in even frames or odd frames
2. RGB to gray-scale
3. scale to 84x84

Notice: the preprocessing will be applied to the m most recent frames and stacks them to produce the input to the Q-function, most time m=4

## Ref
1. Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.
2. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529.
