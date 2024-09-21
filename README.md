1 - Install a conda environment to run the project using : `conda env create -f environement.yml`, and activate it  
2 - Check `conf/config.yaml` to set the experiments variables  
3 - Run `tiramisu_api_tutorial.py` to make sure the tiramisu environment works properly   
4 - You can use the previous code to test and explore the transformations  
5 - To run the RL experiment execute `python rl_train.py --num-workers=??`  
6 - Results are stores in `./experiment_dir/ray_results` you can use tensorboard to visualize them  
