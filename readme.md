The code is still rather messy and disorganised, with a lot of unused code and repeated code, but still works if any testing is needed. 

To run the code, the following are required:

- FCEUX (http://www.fceux.com/web/home.html), using luajit rather than the regular lua.
- torch (http://torch.ch/)
- CUDA (https://developer.nvidia.com/cuda-zone)
- The following lua packages:
    - RNN (https://github.com/Element-Research/rnn/)
    - CUTORCH (https://github.com/torch/cutorch)
    - CUNN (https://github.com/torch/cunn)
    - Optim (https://github.com/torch/optim/)
    - luasocket (https://github.com/diegonehab/luasocket)
    - json.lua (https://github.com/rxi/json.lua)
- The American release of Mega man 2
- MyCBR rest-api (https://github.com/ntnu-ai-lab/mycbr-rest)

If there is any need to change the name of the concept, amalgamation function or case base, this can be done in the mycbr.lua file. 

The save states inside FCEUX has to be set up in the following pattern:
- 1: Bubble man
- 2: Air man
- 3: Quick man
- 4: Wood man
- 5: Crash man
- 6: Flash man
- 7: Metal man
- 8: Heat man
At the beginning of each level.

The testing and training parameters should be updated in both files if changes are made, for the main\_script files. To change the network topology, edit the parameters in the main\_file or for more advanced changes, edit the rnn-rl.lua file.

To run the system, first set up the MyCBR rest api server with the desired mycbr project. Then run the emulator and open the game, and finally load one of the four main\_script files as a script in the emulator for either training or testing. 

If you're reading this on github, the pretrained model is not included here because of the size limitation github has on files, but they are included in the zip handed in with the thesis.
