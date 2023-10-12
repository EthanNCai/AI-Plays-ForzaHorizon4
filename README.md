# AI Plays Horizon 4

## introduction

### overall

Forza Horizon 4 is a very popular and well known single
and multiplayer cross-platform racing game.
The main gameplay of the game is to drive a vehicle 
and then follow a fixed route (players must pass all the check points along the way), then try your best to be the first to cross 
the finish line.

The ultimate goal of this project is to create an ML model that can play this game
fully automated at an acceptable skill, allowing us to win or 
complete a race even if we are away from the keyboard. 

It is important to note that this project focuses on allowing
the model to clone human behavior and does not access to any of 
the game's built-in APIs (actually there's no such kind of APIs at all), and that the information 
(mostly visual) that the real player gets while playing 
will be used as the __only__ input to the model.
The model's outputs are also identical to the player's, 
i.e. keyboard action.







here shows how I develop everything

1. Generating datasets by recording human's gameplay (a sample is consists of screen and the keyboard actions)
2. Data preprocessing (cropping, resizing, gray-scaling etc.)
3. Training with different models
4. evaluate performance in actual game environments(Forza Horizon 4)

![](Pictures/final_ill.png)
### details 

## Getting Started

### Step1 : capture dataset

You need to collect real human in-game actions and the corresponding game 
screens as a training set. Here you can choose to record your own training 
set using the python script I provided or download the training set I previously collected.

[Download  dataset-01-.npy](www.baidu.com)

### Step3 : preview dataset

### Step2 : preprocess dataset


### Step3 : choose, train, preview and finally play model

