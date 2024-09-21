from typing import List
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.services.legality_service import LegalityService
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.utils.exceptions import ExecutingFunctionException
from ..models.schedule import Schedule
from ..models.action import *
from conf.config import Config
import numpy as np
import subprocess



class SchedulerService:
    def __init__(self):
        # The Schedule object contains all the informations of a program : annotatons , tree representation ...
        self.schedule_object: Schedule = None
        # The branches generated from the program tree 
        self.branches : List[Branch] = []
        self.current_branch = 0
        # The prediction service is an object that has a value estimator `get_predicted_speedup(schedule)` of the speedup that a schedule will have
        # This estimator is a recursive model that needs the schedule representation to give speedups
        self.prediction_service = PredictionService()
        # A schedules-legality service
        self.legality_service = LegalityService()
        

    def set_schedule(self, schedule_object: Schedule):
        """
        The `set_schedule` function is called first in `tiramisu_api` to initialize the fields when a new program is fetched from the dataset.
        input :
            - schedule_object : contains all the inforamtions on a program and the schedule
        output :
            - a tuple of vectors that represents the main program and the current branch , in addition to their respective actions mask
        """
        self.schedule_object = schedule_object
        # We create the branches of the program
        self.create_branches()
        # Init the index to the 1st branch
        self.current_branch = 0
        main_repr = ConvertService.get_schedule_representation(schedule_object)
        branch_repr = ConvertService.get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the main program and the branch in a 180 sized vector for each

        _, main_embed = self.prediction_service.get_predicted_speedup(*main_repr,schedule_object)
        _, branch_embed = self.prediction_service.get_predicted_speedup(*branch_repr,self.branches[self.current_branch])
        return ([main_embed, branch_embed], 
                self.branches[self.current_branch].actions_mask
                )     
  

    def create_branches(self):
        # Make sure to clear the branches of the previous function if there are ones
        self.branches.clear()
        for branch in self.schedule_object.branches : 
            # Create a mock-up of a program from the data of a branch
            program_data = {
                "program_annotation" : branch["annotations"],
                "schedules_legality" : {},
                "schedules_solver" : {}
            }
            # The Branch is an inherited class from Schedule, it has all its characteristics
            new_branch = Branch(TiramisuProgram.from_dict(self.schedule_object.prog.name,
                                                          data=program_data,
                                                          original_str=""))
            # The branch needs the original cpp code of the main function to calculate legality of schedules
            new_branch.prog.load_code_lines(self.schedule_object.prog.original_str)
            self.branches.append(new_branch)
            
    def next_branch(self):
        # Switch to the next branch to optimize it 
        self.current_branch += 1
        if (self.current_branch == len(self.branches)):
            # This matks the finish of exploring the branches
            return None
        main_repr = ConvertService.get_schedule_representation(self.schedule_object)
        branch_repr = ConvertService.get_schedule_representation(self.branches[self.current_branch])
        # Using the model to embed the program and the branch in a 180 sized vector each
  
        _, main_embed = self.prediction_service.get_predicted_speedup(*main_repr,self.schedule_object)
        _, branch_embed = self.prediction_service.get_predicted_speedup(*branch_repr,self.branches[self.current_branch])
        
        return ([main_embed, branch_embed], 
                self.branches[self.current_branch].actions_mask
                )
                

    def apply_action(self, action: Action):
        """
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        """
        legality_check = self.legality_service.is_action_legal(schedule_object=self.schedule_object,
                                                               branches=self.branches,
                                                               current_branch=self.current_branch,
                                                               action=action)
        #print(action.name, ": ", action.params)
        
        speedup = 1
        embedding_tensor = None
        if legality_check:
            try :
                self.update_schedule_dict(action)
            except TypeError as e:
                # If the execution went wrong remove it from the schedule list
                # self.schedule_object.schedule_list.pop()
                # # Rebuild the schedule string after removing the action 
                # schdule_str = self.schedule_object.build_sched_string()
                # # Storing the schedule string to use it later 
                # self.schedule_object.schedule_str = schdule_str
                #print("TypeError caught")
                legality_check = False
                speedup=1
                
            try : 
                # After successfuly applying an action we get the new representation of the main schedule and the branch
                main_repr_tensors = ConvertService.get_schedule_representation(
                    self.schedule_object)
                branch_repr_tensors = ConvertService.get_schedule_representation(
                    self.branches[self.current_branch])
                
                # We mesure the speedup from the main schedule and we get the embeddings for both (main and branch)
                speedup, main_embedding_tensor = self.prediction_service.get_predicted_speedup(
                    *main_repr_tensors, self.schedule_object)
                _, branch_embedding_tensor = self.prediction_service.get_predicted_speedup(
                    *branch_repr_tensors, self.branches[self.current_branch])
                
                # We pach the 2 tensors to represent the program and the current branch
                embedding_tensor = [main_embedding_tensor, branch_embedding_tensor]
                
                if Config.config.tiramisu.env_type == 'cpu' : 
                    if (Config.config.dataset.is_benchmark):
                        speedup = 1
                    else :
                        speedup = self.prediction_service.get_real_speedup(schedule_object=self.schedule_object)

                if isinstance(action, Tiling):
                    action.apply_on_branches(self.branches, self.schedule_object.schedule_list)
                    
                elif isinstance(action, Unrolling):
                    action.apply_on_branches(self.branches, self.current_branch)
                
                else : 
                    action.apply_on_branches(self.branches)

            except ExecutingFunctionException as e :
                # If the execution went wrong remove it from the schedule list
                self.schedule_object.schedule_list.pop()
                # Rebuild the scedule string after removing the action 
                schdule_str = self.schedule_object.build_sched_string()
                # Storing the schedule string to use it later 
                self.schedule_object.schedule_str = schdule_str
                legality_check = False

            except subprocess.CalledProcessError as e:
                # If the execution went wrong remove it from the schedule list
                self.schedule_object.schedule_list.pop()
                # Rebuild the scedule string after removing the action 
                schdule_str = self.schedule_object.build_sched_string()
                # Storing the schedule string to use it later 
                self.schedule_object.schedule_str = schdule_str
                legality_check = False
                speedup=1

            except TypeError as e:
                # If the execution went wrong remove it from the schedule list
                #print("TypeError caught")
                self.schedule_object.schedule_list.pop()
                # Rebuild the schedule string after removing the action 
                schdule_str = self.schedule_object.build_sched_string()
                # Storing the schedule string to use it later 
                self.schedule_object.schedule_str = schdule_str
                legality_check = False
                speedup=1

        # if legality_check:
        #     ##############################
        #     action_log_string = f"{action.name}: {action.params}"
        #     with open(f"/scratch/ss18596/dev9/act30_cpu/actions_files/act_file{action.worker_id}.txt", 'a') as file:
        #         file.write(action_log_string + '\n')
        #     ##############################
        
        return speedup, embedding_tensor, legality_check, self.branches[self.current_branch].actions_mask


    def update_schedule_dict(self, action):
        if isinstance(action, Parallelization):
            self.apply_parallelization(action=action)

        elif isinstance(action, Reversal):
            self.apply_reversal(action=action)

        elif isinstance(action, Interchange):
            self.apply_interchange(action=action)

        elif isinstance(action, Tiling):
            self.apply_tiling(action=action)
            
        elif isinstance(action, Unrolling):
            self.apply_unrolling(action=action)

        elif isinstance(action, Skewing):
            self.apply_skewing(action=action)
        
        elif isinstance(action, Add01):
            self.apply_add01(action=action)

        elif isinstance(action, Add02):
            self.apply_add02(action=action)

        elif isinstance(action, Add03):
            self.apply_add03(action=action)

        elif isinstance(action, Add04):
            self.apply_add04(action=action)

        elif isinstance(action, Add12):
            self.apply_add12(action=action)

        elif isinstance(action, Add13):
            self.apply_add13(action=action)

        elif isinstance(action, Add14):
            self.apply_add14(action=action)

        elif isinstance(action, Add23):
            self.apply_add23(action=action)

        elif isinstance(action, Add24):
            self.apply_add24(action=action)

        elif isinstance(action, Add34):
            self.apply_add34(action=action)

        elif isinstance(action, Gauss01):
            self.apply_gauss01(action=action)

        elif isinstance(action, Gauss02):
            self.apply_gauss02(action=action)

        elif isinstance(action, Gauss03):
            self.apply_gauss03(action=action)

        elif isinstance(action, Gauss04):
            self.apply_gauss04(action=action)

        elif isinstance(action, Gauss10):
            self.apply_gauss10(action=action)

        elif isinstance(action, Gauss12):
            self.apply_gauss12(action=action)

        elif isinstance(action, Gauss13):
            self.apply_gauss13(action=action)

        elif isinstance(action, Gauss14):
            self.apply_gauss14(action=action)

        elif isinstance(action, Gauss20):
            self.apply_gauss20(action=action)

        elif isinstance(action, Gauss21):
            self.apply_gauss21(action=action)

        elif isinstance(action, Gauss23):
            self.apply_gauss23(action=action)

        elif isinstance(action, Gauss24):
            self.apply_gauss24(action=action)

        elif isinstance(action, Gauss30):
            self.apply_gauss30(action=action)

        elif isinstance(action, Gauss31):
            self.apply_gauss31(action=action)

        elif isinstance(action, Gauss32):
            self.apply_gauss32(action=action)

        elif isinstance(action, Gauss34):
            self.apply_gauss34(action=action)

        elif isinstance(action, Gauss40):
            self.apply_gauss40(action=action)

        elif isinstance(action, Gauss41):
            self.apply_gauss41(action=action)

        elif isinstance(action, Gauss42):
            self.apply_gauss42(action=action)

        elif isinstance(action, Gauss43):
            self.apply_gauss43(action=action)


    def apply_parallelization(self, action: Action):
        # Getting the first comp of the selected branch
        computation = list(self.branches[self.current_branch].it_dict.keys())[0]
        # Getting the name of the iterator that points to the loop_level
        # action.params[0]] Represents the loop level 
        iterator = self.branches[self.current_branch].it_dict[computation][action.params[0]]["iterator"]
        # Add the tag of parallelized loop level to the computations of the action
        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["parallelized_dim"] = iterator
            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                    # Update the schedule
                    branch.schedule_dict[comp]["parallelized_dim"] = iterator


    def apply_reversal(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [2, 0, 0, action.params[0] , 0, 0, 0, 0]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(transformation)
            # Update the matrice of transformations
            self.schedule_object.schedule_mat[comp]["matrix"] = np.dot(self.schedule_object.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,self.schedule_object.schedule_mat[comp]["nb_it"]))
            # mark the computation as transformed
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(transformation)
                    # Update the matrice of transformations
                    branch.schedule_mat[comp]["matrix"] = np.dot(branch.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,branch.schedule_mat[comp]["nb_it"]))

        for branch in self.branches : 
            for comp in action.comps : 
                if (comp in branch.comps):
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed+=1
                    break

    def apply_interchange(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        
        transformation = [1, action.params[0], action.params[1], 0, 0, 0, 0, 0]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(transformation)
            # Update the matrice of transformations
            self.schedule_object.schedule_mat[comp]["matrix"] = np.dot(self.schedule_object.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,self.schedule_object.schedule_mat[comp]["nb_it"]))
            # mark the computation as transformed
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(transformation)
                    # Update the matrice of transformations
                    branch.schedule_mat[comp]["matrix"] = np.dot(branch.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,branch.schedule_mat[comp]["nb_it"]))
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
        for branch in self.branches : 
            for comp in action.comps : 
                if (comp in branch.comps):
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed+=1
                    break

    def apply_skewing(self, action):
        # The tag representation is as follows:
        #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
        #     Where the type_of_transformation tag is:
        #       - 0 for no transformation being applied
        #       - 1 for loop interchange
        #       - 2 for loop reversal
        #       - 3 for loop skewing
        transformation = [
            3, 0, 0, 0, action.params[0], action.params[1], action.params[2], action.params[3]
        ]

        for comp in action.comps:
            # Update main schedule
            self.schedule_object.schedule_dict[comp]["transformations_list"].append(transformation)
            # Update the matrice of transformations
            self.schedule_object.schedule_mat[comp]["matrix"] = np.dot(self.schedule_object.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,self.schedule_object.schedule_mat[comp]["nb_it"]))
            # mark the computation as transformed
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    branch.schedule_dict[comp]["transformations_list"].append(transformation)
                    # Update the matrice of transformations
                    branch.schedule_mat[comp]["matrix"] = np.dot(branch.schedule_mat[comp]["matrix"], ConvertService.get_trasnformation_matrix_from_vector(transformation,branch.schedule_mat[comp]["nb_it"]))
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
        for branch in self.branches : 
            for comp in action.comps : 
                if (comp in branch.comps):
                    # For the affine transformations we must keep track of how many of them are applied
                    # inside the variable branch.transformed , the limit is 4
                    branch.transformed+=1
                    break

    def apply_tiling(self, action):
        loop_levels = action.params[: len(action.params) // 2]
        tile_sizes = action.params[len(action.params) // 2 :]
        tiling_depth = len(loop_levels)
        tiling_factors = [str(p) for p in tile_sizes]
        for comp in action.comps:
            tiling_dims = [
                self.schedule_object.it_dict[comp][l]["iterator"]
                for l in loop_levels
            ]
            tiling_dict = {
                "tiling_depth": tiling_depth,
                "tiling_dims": tiling_dims,
                "tiling_factors": tiling_factors,
            }

            self.schedule_object.schedule_dict[comp]["tiling"] = tiling_dict
            # print("The comp : ", comp)
            # pp.pprint(self.schedule_object.schedule_dict[comp])
            for branch in self.branches:
                # Check for the branches that needs to be updated
                if comp in branch.comps:
                    # Update the branch schedule
                    branch.schedule_dict[comp]["tiling"] = tiling_dict


    def apply_unrolling(self, action):
        # Unrolling is always applied at the innermost level , so it includes only the computations from 
        # one branch , no need to check if the action will update other branches besides the current one
        for comp in action.comps:
            # Update the main schedule
            self.schedule_object.schedule_dict[comp]["unrolling_factor"] = str(action.params[1])
            # Update the branch schedule 
            self.branches[self.current_branch].schedule_dict[comp]["unrolling_factor"] = str(action.params[1])
            
    def apply_add01(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[0][1] = matrix[0][1] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[0][1] = matrix[0][1] + 1
                    branch.schedule_mat[comp]["transformed"] = True



    def apply_add02(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[0][2] = matrix[0][2] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[0][2] = matrix[0][2] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add03(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[0][3] = matrix[0][3] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[0][3] = matrix[0][3] + 1
                    branch.schedule_mat[comp]["transformed"] = True

        

    def apply_add04(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[0][4] = matrix[0][4] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[0][4] = matrix[0][4] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add12(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[1][2] = matrix[1][2] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[1][2] = matrix[1][2] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add13(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[1][3] = matrix[1][3] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[1][3] = matrix[1][3] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add14(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[1][4] = matrix[1][4] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[1][4] = matrix[1][4] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add23(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[2][3] = matrix[2][3] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[2][3] = matrix[2][3] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_add24(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[2][4] = matrix[2][4] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[2][4] = matrix[2][4] + 1
                    branch.schedule_mat[comp]["transformed"] = True
                    
    def apply_add34(self, action):
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            matrix[3][4] = matrix[3][4] + 1
            self.schedule_object.schedule_mat[comp]["transformed"] = True

            for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[3][4] = matrix[3][4] + 1
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss01(self, action):
        row_i = 0
        row_j = 1
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss02(self, action):
        row_i = 0
        row_j = 2
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True
            
        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss03(self, action):
        row_i = 0
        row_j = 3
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss04(self, action):
        row_i = 0
        row_j = 4
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss10(self, action):
        row_i = 1
        row_j = 0
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss12(self, action):
        row_i = 1
        row_j = 2
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss13(self, action):
        row_i = 1
        row_j = 3
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss14(self, action):
        row_i = 1
        row_j = 4
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss20(self, action):
        row_i = 2
        row_j = 0
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True
                    
    def apply_gauss21(self, action):
        row_i = 2
        row_j = 1
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss23(self, action):
        row_i = 2
        row_j = 3
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss24(self, action):
        row_i = 2
        row_j = 4
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss30(self, action):
        row_i = 3
        row_j = 0
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss31(self, action):
        row_i = 3
        row_j = 1
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True
                    
    def apply_gauss32(self, action):
        row_i = 3
        row_j = 2
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss34(self, action):
        row_i = 3
        row_j = 4
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss40(self, action):
        row_i = 4
        row_j = 0
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss41(self, action):
        row_i = 4
        row_j = 1
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss42(self, action):
        row_i = 4
        row_j = 2
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True

    def apply_gauss43(self, action):
        row_i = 4
        row_j = 3
        for comp in action.comps:
            matrix = self.schedule_object.schedule_mat[comp]["matrix"]
            for j in range(len(matrix[0])):
                matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
            self.schedule_object.schedule_mat[comp]["transformed"] = True

        for branch in self.branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                     # Update the schedule
                    matrix = branch.schedule_mat[comp]["matrix"]
                    matrix[row_i][j] = matrix[row_i][j] + matrix[row_j][j]
                    branch.schedule_mat[comp]["transformed"] = True