import numpy as np
import math
import ray
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext
import torch
import os
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor
from env_api.tiramisu_api import TiramisuEnvAPI
from conf.config import Config
from env_api.core.services.converting_service import ConvertService

class TiramisuRlEnv(gym.Env):
    def __init__(self, config: EnvContext):
        Config.config = config["config"]
        # local_dataset=False => means that we are reading data from external source than the dataservice implemented in
        # TiramisuEnvAPI, this data is the annotations of a function + the leglaity of schedules
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        self.dataset_actor: DatasetActor = config["dataset_actor"]
        # Define action and observation spaces
        self.action_space = spaces.Discrete(62)
        self.observation_space = spaces.Dict(
            {
                "embedding": spaces.Box(-np.inf, np.inf, shape=(387,)),
                "actions_mask": spaces.Box(0, 1, shape=(62,)),
            }
        )
        # The variable `self.worker_index` indexes which worker/actor is working on the chosen function, it will help us avoid problems during compiling,
        # by adding the index of the worker to the name of the worker in order to not interfer with the compilation of another node
        if(isinstance(config,ray.rllib.env.env_context.EnvContext)):
            # This the case of training
            self.worker_index = str(config.worker_index)
        else :
            # This is the case of evaluating
            self.worker_index = ""
        self.current_program = ""
        self.reset()

    def reset(self, seed=None, options={}):
        embedded_tensor = None
        # Select a program randomly
        while embedded_tensor == None:
            # There is some programs that has unsupported loop levels , acces matrices , ...
            # These programs are not supported yet so the embedded_tensor will be None
            # program = random.choice(self.tiramisu_api.programs)
            #prog_infos = ray.get(self.dataset_actor.get_next_function.remote())
            # The shape of embedded_tensor : (180,)
            # Shape of actions mask : (35,)
            prog = None
            while prog == None:
                prog_infos = ray.get(self.dataset_actor.get_next_function.remote())
                prog = self.tiramisu_api.set_program(*prog_infos)
            
            #embedded_tensor, actions_mask = self.tiramisu_api.set_program(*prog_infos)
            embedded_tensor, actions_mask = prog
            self.current_program = prog_infos[0]

        self.state = {
            # Converting Tensor to numpy array
            "embedding": self.preprocess_embeddings(embeddings=embedded_tensor),
            "actions_mask": actions_mask,
        }

        self.previous_speedup = self.reward = 1
        self.done = self.truncated = False
        self.info = {}
        self.action_index = 0
        return self.state, self.info

    def step(self, action):
        self.action_index += 1
        
        speedup, embedded_tensor, legality, actions_mask = self.apply_flattened_action(
            action=action
        )
        if legality and not self.done:
            self.state = {
                "embedding": self.preprocess_embeddings(
                    embeddings=embedded_tensor, action=action
                ),
                "actions_mask": actions_mask,
            }


        self.reward = self.reward_process(action , legality, speedup)

        # Update dataset on episode end
        if self.done:
            tiramisu_program_dict = (
                self.tiramisu_api.get_current_tiramisu_program_dict()
            )
            self.dataset_actor.update_dataset.remote(
                self.current_program, tiramisu_program_dict
            )

        return self.state, self.reward, self.done, self.truncated, self.info


    def reward_process(self, action, legality, total_speedup):
        switching_branch_penality = 1
        illegal_action_penality = 1
        max_speedup = np.inf
        log_base = 4

        if legality:
            if action != 61:
                # If the action is not Next
                instant_speedup = total_speedup / self.previous_speedup
                self.previous_speedup = total_speedup
            else:
                instant_speedup = switching_branch_penality
        else:
            instant_speedup = illegal_action_penality

        instant_speedup = np.clip(instant_speedup, 0, max_speedup)
        
        print(f"{action}, instant_speedup: {instant_speedup}")

        try:
            reward = math.log(instant_speedup, log_base)
            
        except ValueError as e:
            print("Error:", e)
            reward = 1

        return reward

    def apply_flattened_action(self, action):
        if action < 4:
            loop_level = action
            # Interchange of loops (0,1) (1,2) (2,3) (3,4)
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.interchange(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 9:
            loop_level = action - 4
             # Reversal from 0 to 4
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.reverse(
                loop_level=loop_level, env_id=action, worker_id=self.worker_index
            )
        elif action < 12:
            loop_level = action - 9
            # Skewing 0,1 to 2,3
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.skew(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 14:
            loop_level = action - 12
            # For parallelization 0 and 1
            (
                speedup,
                embedded_tensor,
                legality,
                actions_mask,
            ) = self.tiramisu_api.parallelize(
                loop_level=loop_level, env_id=action, worker_id=self.worker_index
            )
        elif  action < 18:
            loop_level= action - 14

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=128,
                size_y=64,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif  action < 22:
            loop_level= action -18

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=64,
                size_y=128,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif  action < 26:
            loop_level= action - 22

            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=64,
                size_y=64,
                env_id=action,
                worker_id=self.worker_index,
            )
        elif action < 31:
            factor = action - 24
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.unroll(
                unrolling_factor=2**factor, env_id=action, worker_id=self.worker_index
            )
        elif action == 31:
            # The action add01
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add01(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 32:
            # The action add02
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add02(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 33:
            # The action add12
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add12(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 34:
            # The action add03
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add03(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 35:
            # The action add13
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add13(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 36:
            # The action add23
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add23(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 37:
            # The action add04
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add04(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 38:
            # The action add14
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add14(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 39:
            # The action add24
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add24(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 40:
            # The action add34
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.add34(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 41:
            # The action gauss01
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss01(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 42:
            # The action gauss10
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss10(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 43:
            # The action gauss02
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss02(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 44:
            # The action gauss12
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss12(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 45:
            # The action gauss20
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss20(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 46:
            # The action gauss21
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss21(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 47:
            # The action gauss03
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss03(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 48:
            # The action gauss13
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss13(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 49:
            # The action gauss23
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss23(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 50:
            # The action gauss30
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss30(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 51:
            # The action gauss31
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss31(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 52:
            # The action gauss32
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss32(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 53:
            # The action gauss04
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss04(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 54:
            # The action gauss14
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss14(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 55:
            # The action gauss24
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss24(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 56:
            # The action gauss34
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss34(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 57:
            # The action gauss40
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss40(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 58:
            # The action gauss41
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss41(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 59:
            # The action gauss42
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss42(
                env_id=action, worker_id=self.worker_index
            )
        elif action == 60:
            # The action gauss43
            speedup, embedded_tensor, legality, actions_mask = self.tiramisu_api.gauss43(
                env_id=action, worker_id=self.worker_index
            )

        else:
            # Next case
            next_branch = self.tiramisu_api.scheduler_service.next_branch()
            if next_branch == None:
                speedup, embedded_tensor, legality, actions_mask = (
                    1,
                    None,
                    True,
                    np.zeros(62),
                )
                self.done = True
            else:
                speedup, embedded_tensor, legality, actions_mask = (
                    1,
                    next_branch[0],
                    True,
                    next_branch[1],
                )

        return speedup, embedded_tensor, legality, actions_mask

    # def get_padded_transform_matrix(
    #     cls, global_matrix
    # ):

    #     MaxDepth = 5
        
    #     global_matrix = global_matrix.tolist()
        
    #     # Add padding to the matrix in case the number of iterators is less than MAX_DEPTH
    #     padded_matrix = [item for sublist in global_matrix for item in sublist]
    #     padded_matrix += [0]*(MaxDepth*MaxDepth - len(padded_matrix)) 
        
    #     return padded_matrix
    
    
    def preprocess_embeddings(self, embeddings: torch.Tensor, action=-1):
        
        comp = self.tiramisu_api.scheduler_service.branches[self.tiramisu_api.scheduler_service.current_branch].comps[0]
        matrix = self.tiramisu_api.scheduler_service.schedule_object.schedule_mat[comp]["matrix"]

        MaxDepth = 5
        matrix = matrix.tolist()
        padded_matrix = [item for sublist in matrix for item in sublist]
        padded_matrix += [0]*(MaxDepth*MaxDepth - len(padded_matrix)) 
        
        #padded_matrix = get_padded_transform_matrix(matrix)
        
        embeddings = torch.cat(
            (
                *embeddings,
                torch.tensor(padded_matrix),
                torch.tensor(
                    [
                        (
                            (self.tiramisu_api.scheduler_service.current_branch + 1)
                            / len(self.tiramisu_api.scheduler_service.branches)
                        ),
                        action,
                    ],
                    dtype=torch.float32,
                ),
            ),
            dim=0,
        )
        return embeddings.numpy()
