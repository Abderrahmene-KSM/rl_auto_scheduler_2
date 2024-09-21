import numpy as np , copy
from conf.config import Config
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.core.services.converting_service import ConvertService
from env_api.scheduler.models.action import *
from env_api.scheduler.models.representation import Representation

class Schedule:
    def __init__(self, program: TiramisuProgram):
        self.schedule_str = ""
        self.prog = program

        self.transformed = 0
        # List of computations of the program
        self.comps = self.prog.comps
        # Iterators dictionnary
        self.it_dict = {}
        # List of branches of the program tree
        self.branches = []
        # List of common iterators
        self.common_it = []
        # A dictionnary that has the types of schedule applied on the program with their representation in the cost model
        self.schedule_dict = {}
        # self.schedule_list is an array that contains a list of optimizations that has been applied on the program
        self.schedule_list = []
        # Additional loops when Tiling is applied
        self.additional_loops = 0
        # Dictionary of all comps with the matrice that represent all the transformations applied to them
        self.schedule_mat = {}

        self.repr : Representation = None

        self.actions_mask = None


        if((type(self).__name__) == "Schedule"):
            self.__init_schedule_dict_tags()
            self.__calculate_common_it()
            self.__init_representation()
            self.__set_action_mask()
            self.__form_iterators_dict()
            self.__form_branches()
            self.__init_schedule_mat()
        else : 
            self.__init_schedule_dict_tags()
            self.__init_representation()
            self.__set_action_mask()
            self.__form_iterators_dict()
            self.__init_schedule_mat()


    def __init_representation(self):
        self.repr = Representation(*ConvertService.get_representation_template(self.prog.annotations,self.schedule_dict))

    def __init_schedule_dict_tags(self):
        self.schedule_dict["fusions"] = None
        for comp in self.comps:
            self.schedule_dict[comp] = {
                "tiling": {},
                "unrolling_factor": None,
                "parallelized_dim": None,
                "shiftings": None,
                "transformations_list": []
            }
        self.schedule_dict["tree_structure"] = {
            "roots": [ConvertService.get_tree_structure(self.prog.annotations)]}

    def __calculate_common_it(self):
        if len(self.comps) != 1:  # Multi-computation program
            # comps_it is a list of lists of iterators of computations
            comps_it = []
            for comp in self.comps:
                comps_it.append(
                    self.prog.annotations["computations"][comp]["iterators"]
                )
            self.common_it = comps_it[0]
            for comp_it in comps_it[1:]:
                self.common_it = [it for it in comp_it if it in self.common_it]
        else:  # A single comp program
            self.common_it = self.prog.annotations["computations"][self.comps[0]][
                "iterators"
            ]
    
    def __set_action_mask(self):
        self.actions_mask = np.zeros(52)

        if((type(self).__name__) == "Branch"):
            nb_it = len(self.prog.annotations["computations"][self.comps[0]]["iterators"])
            # mask add and addrow actions that target an element not included in the transformation matrix (out of range)
            if nb_it == 2:
                self.actions_mask[32:41] = 1
                self.actions_mask[33:51] = 1
            elif nb_it == 3:
                self.actions_mask[34:41] = 1
                self.actions_mask[37:51] = 1
            elif nb_it == 4:
                self.actions_mask[37:41] = 1
                self.actions_mask[43:51] = 1

    def __form_iterators_dict(self):
        for comp in self.comps:
            comp_it_dict = {}
            iterators = list(self.prog.annotations["computations"][comp]["iterators"])
            for i in range(len(iterators)):
                comp_it_dict[i] = {}
                comp_it_dict[i]['iterator'] = iterators[i]
                comp_it_dict[i]['lower_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['lower_bound']
                comp_it_dict[i]['upper_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['upper_bound']
            self.it_dict[comp] = comp_it_dict
            
    def __form_branches(self):
        branches = []
        iterators = copy.deepcopy(self.prog.annotations["iterators"])
        computations = copy.deepcopy(self.prog.annotations["computations"])
        it = {}
        for computation in computations:
            iterators = copy.deepcopy(self.prog.annotations["computations"][computation]["iterators"])
            if iterators[-1] in it :
                it[iterators[-1]]["comps"].append(computation)
            else :
                it[iterators[-1]] = {
                    "comps" : [computation],
                    "iterators" : iterators
                }
        
        for iterator in it :
            branches.append({
                "comps" : it[iterator]["comps"],
                "iterators" : it[iterator]["iterators"],
                "annotations": {}
            })
                
        for branch in branches :
            branch_annotations = {
                "computations" : {},
                "iterators": {}
            }
            for comp in branch["comps"]:
                branch_annotations["computations"][comp] = copy.deepcopy(self.prog.annotations["computations"][comp])
            # extract the branch specific iterators annotations
            for iterator in branch["iterators"]:
                branch_annotations["iterators"][iterator] = copy.deepcopy(self.prog.annotations["iterators"][iterator])
                if (self.prog.annotations["iterators"][iterator]["parent_iterator"]):
                    # Making sure that the parent node has the actual node as the only child
                    # It may happen that the parent node has many children but in a branch it is only allowed
                    # to have a single child to form a straight-forward branch from top to bottom
                    parent = (branch_annotations["iterators"][iterator]["parent_iterator"])
                    branch_annotations["iterators"][parent]["child_iterators"] = copy.deepcopy([iterator])
                    branch_annotations["iterators"][parent]["computations_list"] = []
            branch["annotations"] = copy.deepcopy(branch_annotations)

        self.branches = branches


    def build_sched_string(self) -> str:
        # Prepare a dictionary of computations name to fill it with each action applied on every comp
        comps = {}
        # Map the schedules applied one by one
        for schedule in self.schedule_list : 
            # schedule has comps_schedule which includes the comps that was invloved in the optimisation
            for key in schedule.comps_schedule.keys():
                # Add the data from that schedule to the global comps dictionnary
                if(not key in comps or not comps[key]):
                    comps[key] = ""
                comps[key] += schedule.comps_schedule[key]
        # Prepare the string and form it from the comps dictionary
        schedule_string = ""
        for key in comps.keys():
            schedule_string+= "{"+key+"}:"+comps[key]
        return schedule_string
    
    def __init_schedule_mat(self):
        
        # nb_it: number of the iterators
        # transformed : whether a transformation has been applied to the computation or not
        # matrix: the global matrix that represent all the transformation applied to the computation
        
        for comp in self.comps:
            nb_it = len(self.prog.annotations["computations"][comp]["iterators"])
            self.schedule_mat[comp]={"nb_it":nb_it,
                                    "transformed": False,
                                    "matrices_list": []}

    
    def update_actions_mask(self, action : Action,applied : bool = True):
        if (action.env_id not in range(31, 51)):
            # Whether an action is legal or not we should mask it to not use it again
            self.actions_mask[action.env_id] = 1

        if self.transformed == 4 :
            self.actions_mask[0:12] = 1

        if applied and Config.config.experiment.beam_search_order:
            self.apply_beam_search_conditions(action=action)
        
    def apply_beam_search_conditions(self, action : Action):
        # The order of actions in beam search :
        # Fusion, add, [addrow, Interchange, reversal, skewing], parallelization, tiling, unrolling
        if (isinstance(action,Parallelization)):
            self.actions_mask[0:14] = 1
            self.actions_mask[31:51] = 1

        elif (isinstance(action,Tiling)) : 
            self.actions_mask[0:26] = 1
            self.actions_mask[31:51] = 1

        elif (isinstance(action,Unrolling)):
            self.actions_mask[0:31] = 1
            self.actions_mask[31:51] = 1
