"""
This is the master file for all simulations related to RN-34. The module contains to
parts: 
    1) The class Simulator, which gives the high-level setup of the different stages in
        the simulation. 
    2) The main run script (see bottom of the file), which contains instructions to the
        Simulator object.
        
The detailed simulator setup is given in the module models.

A note on terminology: The code, and the documentation, is inconsistent on the usage of
'fault' and 'fracture'; the former is the geologically meaningful term, while the latter
is standard terminology in simulation development. Please bear with us.

"""
import numpy as np
import porepy as pp
import logging

from models import FlowModel, BiotMechanicsModel


logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, params=None):

        if params is None:
            params = {}

        # Different grid realizations with increasing resolution in the vertical
        # direction. The 2d mesh is the same for all cases. In the vertical direction,
        # the grid is refined around the depths where injection takes place.
        # Case 3 is used for the simulations reported in the paper, however, grids with
        # lower resolution is useful for experimentation runs.
        case = 0
        
        if case == 0:
            self.z_coord = np.array([0, -1500, -2200, -3000, -4000])
        elif case == 1:
            self.z_coord = np.array(
                [0, -1000, -1500, -1800, -2100, -2400, -2700, -3000, -3500, -4000]
            )
        elif case == 2:
            self.z_coord = np.array(
                [
                    0,
                    -1000,
                    -1700,
                    -2100,
                    -2200,
                    -2300,
                    -2400,
                    -2500,
                    -2600,
                    -2700,
                    -3000,
                    -3300,
                    -4000,
                ]
            )
        elif case == 3:
            self.z_coord = np.array(
                [
                    0,
                    -1000,
                    -1700,
                    -2100,
                    -2200,
                    -2300,
                    -2350,
                    -2400,
                    -2450,
                    -2500,
                    -2550,
                    -2600,
                    -2650,
                    -2700,
                    -2800,
                    -3000,
                    -3300,
                    -4000,
                ]
            )
            
        # Fractures to include in the simulation.
        self.included_fractures = [
            "Fault_1",
            "Fault_2",
            "Fault_3a",
            "Fault_3b",
            "Fault_4",
            "Fault_8",
        ]
        self.num_fracs = len(self.included_fractures)


    def simulate_leak_off_test(self, params=None):
        """ Tune the fracture and matrix permeability, as well as matrix porosity
        using data from the         

        """

        print("\n\n")
        print("-------- Start permeability calibration -------------")
        print("\n\n")

        if params is None:
            params = {}

        logger.setLevel(logging.CRITICAL)

        # Create a flow
        solver = FlowModel(self._standard_parameters())

        target = solver.calibration_run(return_values=True)
        print("Found the target value to be:", target)

        logger.setLevel(logging.INFO)

    def _standard_parameters(self):
        # utility function.
        return {
            "z_coordinates": self.z_coord,
            "fracture_files": self.included_fractures,
        }

    def initial_state_poro_mech(self):

        print("\n\n")
        print("-------- Poro-mechanical initialization -------------")
        print("\n\n")

        model_params = self._standard_parameters()
        # model_params["initial_mechanics_state"] = self.initial_mechanics_state

        # The injection lasts in total 6 hours. Also do half an hour of relaxation after this
        model_params["end_time"] = 100 * pp.YEAR
        model_params["time_step"] = model_params["end_time"] / 2
        model_params["num_loadsteps"] = 1

        model_params["export_folder"] = "model_initialization"
        model_params["export_file_name"] = "model_init"

        poro_model = BiotMechanicsModel(model_params)

        pp.run_time_dependent_model(
            poro_model, {"max_iterations": 500, "nl_convergence_tol": 1e-10}
        )
        poro_model.store_contact_state("reference")
        poro_model.store_contact_state("previous")

        logger.info(f"\n\n Model initialization completed\n\n")

        self.model = poro_model

    def simulate_march_29(self, params=None):

        if params is None:
            params = {}

        poro_model = self.model

        # Switch on the sources
        poro_model.activate_sources()

        time_step = 15 * pp.MINUTE

        poro_model.time = 0
        poro_model.time_step = time_step
        poro_model.init_time_step = time_step
        poro_model.end_time = 10.5 * pp.HOUR

        poro_model.export_folder = "march_29"
        poro_model.export_file_name = "stimulation"
        poro_model.prev_export_time = 0
        poro_model.set_export()

        pp.run_time_dependent_model(
            poro_model,
            {
                "prepare_simulation": False,
                "max_iterations": 100,
                "nl_convergence_tol": 1e-10,
            },
        )


if __name__ == "__main__":

    # Set the random seed. This ensures that the local coordinate system in the
    # fractures are always the same, which has been useful for debugging.
    # This is not really a necessary step, but has been kept for legacy / convenience
    # reasons.
    np.random.seed(0)

    # Initialize simulation object
    sim = Simulator()

    # Simulate the leak-off test on the morning of 29 March 2015. This is mainly done
    # to check that the difference in pressure (measured at the injection cell, which
    # is presumed to be proportional to the observed pressure at the measurement point
    # at 1400m depth) between the steady state and the plateau reached during injection
    # is consistent. On a more detailed level, the simulated and measured pressure are
    # not in agreement - for this, a more elaborate parameter calibration would have
    # been needed, but this was deemed not warrented given the scarcity of data.
    sim.simulate_leak_off_test()

    # Initialize the poro-mechanical simulation. This simulates the poro-mechanical
    # system to steady state, using zero injection rate, but with the remaining
    # parameters set to the same values as used for the stimulation.
    sim.initial_state_poro_mech()
    
    # Simulate the
    # IS: Finish comment
    sim.simulate_march_29()
