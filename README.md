# AI Plays Horizon 4

## introduction

Forza Horizon 4 is a very popular and well known single
and multiplayer cross-platform racing game.
The main gameplay of the game is to drive a vehicle 
and then follow a fixed route to be the first to cross 
the finish line.

The aim of this project is to create an AI that will enable 
fully automated play of this game, allowing us to win or 
complete a race even if we are away from the keyboard. 

It is important to note that this project focuses on allowing
the model to clone human behavior and does not use any of 
the game's built-in APIs, and that the information 
(mostly visual) that the real player gets while playing 
will be used as input to the model.
The model's outputs are also identical to the player's, 
i.e. keystrokes.


## route

* Generating datasets by recording human game playing
* Data preprocessing, e.g. edge processing, semantic segmentation
* Select the appropriate model for training and evaluate the performance
* Showcasing performance in actual gaming environments