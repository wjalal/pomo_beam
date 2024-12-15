
from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold
import matplotlib.pyplot as plt

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    BEAM_IDX: torch.Tensor = None
    # shape: (batch, pomo, beam)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo, beam)
    current_node: torch.Tensor = None
    # shape: (batch, pomo, beam)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, beam, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo, beam)
    logprob: torch.Tensor = None
    # shape: (batch, pomo, beam)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.beam_size = env_params['beam_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.BEAM_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        self.nd_0 = None
        self.nxy_0 = None
        self.dxy_0 = None

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        # state, reward, done
        self.reward = None
        self.done = False
        self.logprob = 0.0

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        first_done = False
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError
            
        
        if not first_done :
            # print (depot_xy[0])
            # print (node_xy[0])
            # print (node_demand[0])


            nxy = node_xy.cpu()[0]
            nd = node_demand.cpu()[0]
            dxy = depot_xy.cpu()[0]
            self.nxy_0 = nxy
            self.dxy_0 = dxy
            self.nd_0 = nd
        # Normalize node demand for color intensity
            normalized_demand = 0.3*nd # Normalize between 0 and 1

            # Plotting
            plt.figure(figsize=(6, 6))
            plt.axis([0, 1, 0, 1])  # Unit square
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.title("Nodes and Depots on Unit Square")
            plt.xlabel("X")
            plt.ylabel("Y")

            # Plot nodes as red circles with intensity based on demand
            plt.scatter(
                nxy[:, 0], nxy[:, 1],
                c=normalized_demand,  # Use normalized demand for color mapping
                cmap='copper',  # Red colormap
                s=15,  # Size of the markers
                label="Node"
            )

            # Plot depots as blue squares
            plt.scatter(
                dxy[:, 0], dxy[:, 1],
                color="blue", marker="s", s=40, label="Depot"
            )

            # Add legend
            plt.legend(loc="upper left")
            plt.savefig("plots/instance.png")

            first_done = True


        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None, None].expand(self.batch_size, self.pomo_size, self.beam_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :, None].expand(self.batch_size, self.pomo_size, self.beam_size)
        self.BEAM_IDX = torch.arange(self.beam_size)[None, None, :].expand(self.batch_size, self.pomo_size, self.beam_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.BEAM_IDX = self.BEAM_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo, beam)
        self.logprob = torch.zeros(size=(self.batch_size, self.pomo_size, self.beam_size))
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, self.beam_size, 0), dtype=torch.long)
        # shape: (batch, pomo, beam, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size, self.beam_size), dtype=torch.bool)
        # shape: (batch, pomo, beam)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size, self.beam_size))
        # shape: (batch, pomo, beam)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.beam_size, self.problem_size+1))
        # shape: (batch, pomo, beam, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.beam_size, self.problem_size+1))
        # shape: (batch, pomo, beam, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size, self.beam_size), dtype=torch.bool)
        # shape: (batch, pomo, beam)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.logprob = self.logprob

        reward = None
        done = False

        self.reward = reward
        self.done = done

        return self.step_state, reward, done
    
    def plot(self):
        # print (self.selected_node_list.shape)
        # print (self.selected_node_list[0].shape)
        # print (self.selected_node_list[0][0].shape)
        # print (self.selected_node_list[0][0])
        soln0 = self.selected_node_list[0][0].cpu()
        soln0_path = []
        for element in soln0:
            if element == 0:
                soln0_path.append(self.dxy_0[0].cpu().numpy())  # Add the value from self.dxy[0] if the element is 0
            else:
                soln0_path.append(self.nxy_0[element - 1].cpu().numpy())  # Add the value from self.nxy[element-1] otherwise
        # print (soln0_path)
        x_coords = [coord[0] for coord in soln0_path]
        y_coords = [coord[1] for coord in soln0_path]

        plt.figure(figsize=(6, 6))
        plt.axis([0, 1, 0, 1])  # Unit square
        plt.grid(True, linestyle="--", alpha=0.5)

        # Plot nodes as red circles with intensity based on demand
        plt.scatter(
            self.nxy_0[:, 0], self.nxy_0[:, 1],
            c=0.3*self.nd_0,  # Use normalized demand for color mapping
            cmap='copper',  # Red colormap
            s=15,  # Size of the markers
            label="Node",
            zorder=5,
        )

        # Plot depots as blue squares
        plt.scatter(
            self.dxy_0[:, 0], self.dxy_0[:, 1],
            color="blue", marker="s", s=40, label="Depot", zorder=5
        )

        # Join the coordinates with straight lines
        plt.plot(x_coords, y_coords, color="black", linestyle="-", linewidth=0.5, zorder=3, label="Path")

        # Setup the unit square
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.title("Solution Path on Unit Square")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("plots/soln.png")


    def step(self, selected, selected_beams, prob):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo, beam)
        # print("List before: ", self.selected_node_list[0,0])
        self.selected_node_list = self.selected_node_list[self.BATCH_IDX, self.POMO_IDX, selected_beams]
        # print("Selected beams: ", selected_beams[0,0])
        # print("Selected nodes: ", selected[0, 0])
        # print("List after reorder: ", self.selected_node_list[0,0])
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, :, None]), dim=3)
        # print("List after concatenation: ", self.selected_node_list[0,0])
        # shape: (batch, pomo, beam, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, None, :].expand(self.batch_size, self.pomo_size, self.beam_size, -1)
        # shape: (batch, pomo, beam, problem+1)
        gathering_index = selected[:, :, :, None]
        # shape: (batch, pomo, beam, 1)
        selected_demand = demand_list.gather(dim=3, index=gathering_index).squeeze(dim=3)
        # shape: (batch, pomo, beam)

        self.load = self.load[self.BATCH_IDX, self.POMO_IDX, selected_beams] - selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot


        self.visited_ninf_flag = self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected_beams]
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected[self.BEAM_IDX]] = float('-inf')
        # shape: (batch, pomo, beam,problem+1)
        self.visited_ninf_flag[:, :, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, beam, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, beam, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=3)
        # shape: (batch, pomo, beam)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo, beam)
        if self.selected_count > 115:
            for b in range(self.batch_size):
                for p in range(self.pomo_size):
                    for n in range(self.beam_size):
                        if not self.finished[b, p, n].item():
                            print(self.selected_node_list[b, p, n])

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, :, 0][self.finished] = 0

        self.logprob = self.logprob[self.BATCH_IDX, self.POMO_IDX, selected_beams] + torch.log(prob + 1e-5)

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.logprob = self.logprob

        # returning values
        done = self.finished.all()
        if done:                  
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        self.reward = reward
        self.done = done


        return self.step_state, reward, done


    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, :, None].expand(-1, -1, -1, -1, 2)
        # shape: (batch, pomo, beam, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, None, :, :].expand(-1, self.pomo_size, self.beam_size, -1, -1)
        # shape: (batch, pomo, beam, problem+1, 2)

        ordered_seq = all_xy.gather(dim=3, index=gathering_index)
        # shape: (batch, pomo, beam, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=3, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(4).sqrt()
        # shape: (batch, pomo, beam, selected_list_length)

        travel_distances = segment_lengths.sum(3)
        # shape: (batch, pomo)
        return travel_distances

