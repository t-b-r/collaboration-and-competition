[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Tennis"


# Project 3: Collaboration and Competition

### Overview

In this project, we implement a Multiagent  Deep Deterministic Policy Gradient reinforcement learning technique to train agents to play tennis against eachother. The environment comes from OpenAI's Tennis environment. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. The action space consists of two continuous variables - one representing the agent's movement towards or away from the net and the other is for jumping, and each of the 2 agents receives its own local observation.

The reweards are structured episodically where hitting the ball over the net triggers a reward of +0.1 and if the ball hits the ground or the agent send the ball out of bounds on its oppenet's side a reward of -0.01 is received. The environment is considered solved when the average winning score of a match between the two agents equals +0.5 over 100 consecutive episodes.

![Tennis][image1]


### Instructions - Dependencies
1. Create (and activate) a new environment with Python 3.6.

    Linux or Mac:
    ```
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    Windows:
    ```
    conda create --name drlnd python=3.6 
    activate drlnd
    ```
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

    - Next, install the classic control environment group by following the instructions here.
    - Then, install the box2d environment group by following the instructions here.

3. Clone the repository, and navigate to the python/ folder. Then, install several dependencies.
    ```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

4. Create an IPython kernel for the drlnd environment.
    ```
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

5. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.


### Instructions - Project Code
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    
2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions - Executing the code

- Run all cells in `Tennis.ipynb`

### The files

- **Tennis-Final.ipynb**: Contains the code to run the agent in the environment
- **ddpg_agent.py**: Code for the agent
- **model.py**: Code defining the deep Q Network 
- **checkpoint_critic.pth**: Pytorch model weights for critic network
- **checkpoint_actor.pth**: Pytorch model weights for actor network