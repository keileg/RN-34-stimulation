"""
This module defines the simulation setup (including parameters, governing equations
and numerical methods) for the simulations. The model contains three classes:

RN34SimulationData:
    This is the class which specifies problem geometry and all parameters.
    The two model classes (below) takes the data specification from this class.

FlowModel:
    This model provides simulations of single phase flow. Specifically, the model
    is set up to replicate a fall-off test in the well RN-34 that was run in
    the morning of March 29, 2015, that is, prior to the hydraulic stimulation.
    
BiotMechanicsModel:
    This class gives a poro-mechanical simulation of the reservoir around the well
    RN-34. It is used both for initialization of the model, and simulation of the
    March 29 stimulation experiment.

NOTE: To experiment with different scenarios for formation and fault permeability,
    modify the function set_flow_parameters() in the class RN34SimulationData.

"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

# import pandas as pd
import logging, sys

import scipy.sparse.linalg as spla
import porepy.models.contact_mechanics_model
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot


import porepy as pp
import observations_RN34 as observations


root = logging.getLogger()
root.setLevel(logging.INFO)

if not root.hasHandlers():
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    root.addHandler(ch)

logger = logging.getLogger(__name__)


class RN34SimulationData:
    def __init__(
        self,
        scalar_parameter_key,
        mechanics_parameter_key,
        z_coord,
        fracture_files,
        force_scale=1,
        time_step=None,
    ):

        self.rock_density = 3.0 * pp.KILO / pp.METER ** 3
        self.mechanics_parameter_key = mechanics_parameter_key
        self.scalar_parameter_key = scalar_parameter_key
        self.force_scale = force_scale  # IS: You don't use a length scale?

        self.z_coord = z_coord
        self.fracture_files = fracture_files
        self.time_step = time_step

        self.DEBUG = False

    def _set_domain_corners(self):
        """ Get the corners of the domain.

        Returns:
            domain_corners (np.array, size 3x8): Coordinates of the domain corners.
                The first four columns give the SW, SE, NE, NW corners at the surface,
                the four next give the same placements at the bottom of ethe domain.

        """
        # The domain is set to 10x10km, with a vertical extent of 4000m. This is large
        # enough that the pressure perturbation at the boundaries is negligible.
        domain_size = np.array([10000, 10000, self.z_coord.min()])

        domain_corners = np.array(
            [
                [0, 0, 0],
                [domain_size[0], 0, 0],
                [domain_size[0], domain_size[1], 0],
                [0, domain_size[1], 0],
                [0, 0, domain_size[2]],
                [domain_size[0], 0, domain_size[2]],
                [domain_size[0], domain_size[1], domain_size[2]],
                [0, domain_size[1], domain_size[2]],
            ]
        )

        # Offset, used to center the domain at the origin
        lower_left_offset = np.array(
            [-domain_size[0] / 2, -domain_size[1] / 2, 0]
        ).reshape((-1, 1))

        # The domain center is set at 5000, 5000
        self.domain_center = domain_size.reshape((-1, 1)) - lower_left_offset
        self.domain_center[2] = 0

        # Rotation angle for the domain boundary.
        # If set to zero, the x-axis is east-west. Kept for legacy reasons
        xy_angle = np.deg2rad(0)

        domain_corners = np.vstack(
            (
                domain_corners[:, 0] * np.cos(xy_angle)
                - domain_corners[:, 1] * np.sin(xy_angle),
                domain_corners[:, 0] * np.sin(xy_angle)
                + domain_corners[:, 1] * np.cos(xy_angle),
                domain_corners[:, 2],
            )
        )
        domain_corners += lower_left_offset

        return domain_corners

    def _read_well_path(self):
        """

        """
        return np.genfromtxt("data_files/well_path_2.csv", delimiter=",")

    def _well_coordinate_surface(self):
        return self._read_well_path()[0, 5:]

    def _grid_by_extrusion(self, grid_to_vtu=False):
        """
        Create a 3d simulation grid by first meshing a 2d domain, with the faults as
        line fractures, and extrude the mesh in the vertical direction.

        Returns:
            gb_3d (TYPE): DESCRIPTION.

        """

        # Domain corners.
        domain_corners = self._set_domain_corners()
        # The fault coordinates are given in a special coordinate system, centered far
        # outside the simulation domain. For convenience, we translate the coordinates
        # to the top of the injection well.
        offset_in_data = self._well_coordinate_surface()

        # Data structure to store the fracture traces.
        frac_2d = np.empty((2, 0))

        # Loop over all files with data on fault geometry,
        for file in self.fracture_files:
            path = "data_files/fault_description/" + file
            # read data
            data = np.genfromtxt(path, comments="#")

            # The fault is specified by its surface trace, given as a polyline. In
            # pratice these are almost straight lines, and we approximate the fault by
            # a straight line between the extreme surface points.
            # The final column in the data files have values 1 for surface trace, 2 for
            # (postulated) bottom trace.
            col = data[:, 4]
            # The row index of the extreme points on the surface are 0, and the final
            # row with data[:, 4] == 1
            p_ind = np.array([0, np.where(col == 1)[0][-1]])

            # Get the trace of this fault
            trace = np.array(
                [
                    [
                        data[p_ind[0], 0] - offset_in_data[0],
                        data[p_ind[1], 0] - offset_in_data[0],
                    ],
                    [
                        data[p_ind[0], 1] - offset_in_data[1],
                        data[p_ind[1], 1] - offset_in_data[1],
                    ],
                ]
            )
            frac_2d = np.hstack((frac_2d, trace))

        # 2d domain
        domain_2d = domain_corners[:2, :4]

        # Define the faults by connections between the trace points
        num_pt = frac_2d.shape[1]
        edges = np.vstack((np.arange(0, num_pt, 2), np.arange(1, num_pt, 2)))

        # Make a fracture network object representation of the faults
        network_2d = pp.FractureNetwork2d(frac_2d, edges, domain_2d)
        # The 2d mesh is generated with fairly large cells towards the domain boundary,
        # and refinement towards the fracture network
        mesh_args = {"mesh_size_frac": 250, "mesh_size_min": 1, "mesh_size_bound": 1500}
        # Mesh generation in 2d
        gb = network_2d.mesh(mesh_args)
        # Extend to a 3d grid
        gb_3d, _ = pp.grid_extrusion.extrude_grid_bucket(gb, self.z_coord)

        # We also need to tag the faces of the 3d grid that are on the domain boundary
        g = gb_3d.grids_of_dimension(3)[0]
        fc = g.face_centers

        found = -np.ones(g.num_faces, dtype=np.int)
        # Tolerance, faces with center closer to the specified domain boundary than
        # this will be marked as boundary
        tol = 1e-3

        # Indices, referring to domain corners, of the sides of the domain. See
        # self._set_domain_corners() for comments.
        top = [0, 1, 2, 3]
        west = [0, 3, 7, 4]
        south = [0, 1, 5, 4]
        east = [1, 2, 6, 5]
        north = [2, 3, 7, 6]
        bottom = [4, 5, 6, 7]

        # Loop over the domain sides, assign numbers. Note that the domain boundary
        # tags are the same as the order in this list
        # IMPLEMENTATION NOTE: This could also have been achieved with
        # pp.bc.boundary_side, however, only if the rotation angle of the domain is 0,
        # see self._set_domain_corners()
        for num, side in enumerate([top, west, south, east, north, bottom]):
            dist, *rest = pp.distances.points_polygon(fc, domain_corners[:, side])
            found[dist < tol] = num

        # Sanity check
        assert np.all(found[g.get_boundary_faces()] >= 0)
        g.tags["domain_side"] = found

        logger.info(f"Generated extruded grid with {self.z_coord.size - 1} z-layers.")
        logger.info(f"Total number of cells in 3d grids is {g.num_cells}")

        if grid_to_vtu:
            # Generate a FractureNetwork3d object for the full network - useful for
            # visualization
            z_min = self.z_coord.min()
            frac_3d = []
            for fi in range(edges.shape[1]):

                p0 = frac_2d[:, edges[0, fi]]
                p1 = frac_2d[:, edges[1, fi]]
                p2 = np.hstack((p1, z_min))
                p3 = np.hstack((p0, z_min))
                p0 = np.hstack((p0, 0))
                p1 = np.hstack((p1, 0))

                frac_3d.append(pp.Fracture(np.vstack((p0, p1, p2, p3)).T))

            network_3d = pp.FractureNetwork3d(frac_3d)
            network_3d.to_vtk("fracture_network.vtu")

            viz = pp.Exporter(gb_3d, "grid_information")
            fluid = pp.Water()
            for g, d in gb_3d:
                d[pp.STATE] = {}
                d[pp.STATE]["cell_center_x"] = g.cell_centers[0]
                d[pp.STATE]["cell_center_y"] = g.cell_centers[1]
                d[pp.STATE]["cell_center_z"] = g.cell_centers[2]

                hydrostatic_pressure = (
                    -fluid.density()
                    * pp.GRAVITY_ACCELERATION
                    * g.cell_centers[2]
                    / self.force_scale
                )
                d[pp.STATE]["hydrostatic_pressure"] = (
                    hydrostatic_pressure
                )

            viz.write_vtk(
                data=[
                    "cell_center_x",
                    "cell_center_y",
                    "cell_center_z",
                    "hydrostatic_pressure",
                ]
            )

            for g, d in gb_3d:
                d[pp.STATE] = {}

        return gb_3d

    def create_iceland_grid(self, grid_to_vtu=False):
        """ Create the 3d mixed-dimensional grid, set projections needed for contact
        mechanics calculations
        """
        gb = self._grid_by_extrusion(grid_to_vtu)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        return gb

    def domain_boundary_sides(self, g, gb):
        """
        Get domain boundary faces for a given grid.
        
        If g is not the matrix grid, the index of all boundary faces is returend.
        If it is the matrix grid, we further return the index of faces on bottom,
        top, west, south, east and north, in that order.

        """
        all_bf = g.get_boundary_faces()

        if g.dim < gb.dim_max():
            return all_bf
        else:
            side_tags = g.tags["domain_side"]
            top = np.where(side_tags == 0)[0]
            west = np.where(side_tags == 1)[0]
            south = np.where(side_tags == 2)[0]
            east = np.where(side_tags == 3)[0]
            north = np.where(side_tags == 4)[0]
            bottom = np.where(side_tags == 5)[0]

            # Sanity check on the arrays top, bottom etc.
            # NOTE: This assumes that the domain is not rotated in
            # self._grid_by_extrusion
            xf = g.face_centers
            domain_corners = self._set_domain_corners()

            assert np.allclose(xf[0, west], domain_corners[0, 0])
            assert np.allclose(xf[0, east], domain_corners[0, 1])
            assert np.allclose(xf[1, north], domain_corners[1, 2])
            assert np.allclose(xf[1, south], domain_corners[1, 0])
            assert np.allclose(xf[2, top], domain_corners[2, 0])
            assert np.allclose(xf[2, bottom], domain_corners[2, 4])

            return all_bf, bottom, top, west, south, east, north

    def bc_type_mech(self, g, gb):
        """
        Set the type of boundary condition for the mechanics calculations.
        
        The conditions are Dirichlet at the bottom, Neumann on all other sides.

        """
        if g.dim < gb.dim_max():
            raise ValueError(
                "The mechanics problem should only be posed for the matrix domain"
            )
        all_bf, bottom, top, *rest = self.domain_boundary_sides(g, gb)

        # Default internal BC is Neumann, set Dirichlet at the bottom.
        bc = pp.BoundaryConditionVectorial(g)

        # Dirichlet condition at the top. Not sure about this part.
        bc.is_neu[:, top] = False
        bc.is_dir[:, top] = True
        fix_ind = np.argsort(np.sum(g.face_centers[:, top] ** 2, axis=0))[:2]

        bc.is_neu[:2, top[fix_ind]] = False
        bc.is_dir[:2, top[fix_ind]] = True
        #        import pdb
        #        pdb.set_trace()

        # We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        return bc

    def bc_values_mech(self, g, gb):
        """ Set boundary condition values for the mechanics problem.
        
        The conditions are homogeneous on the top and bottom of the domain,
        corresponding to free and fix boundary, respectively.
        
        On the vertical sides, stresses are set according to section 2.4 in the paper.
        
        """
        # Method should not be invoked for fracture and intersection probems
        if g.dim < gb.dim_max():
            raise ValueError(
                "The mechanics problem should only be posed for the matrix domain"
            )

        # The direction is estimated to be ~45 degrees in CW direction off north.
        # The angle here should be -pi/4
        angle = -np.pi / 4
        # Rotation matrix to the coordinate system of the principal axes
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])

        # Boundary faces, and division into the sides of the domain boundaries
        bf, bottom, top, west, south, east, north = self.domain_boundary_sides(g, gb)

        # Depth-dependent principal stresses.
        fc = g.face_centers
        fc_z = fc[2, bf]
        # Values read from Peter-Borie et al Geotherm Energy 2018

        # The maximum horizontal stress is assumed to be 1.5 * the lithostatic stress
        max_hor_stress = 180 / 134 * fc_z * self.rock_density * pp.GRAVITY_ACCELERATION
        # The minimum horizontal stress is assumed to be 5/8 * the lithostatic stress
        min_hor_stress = 60 / 134 * fc_z * self.rock_density * pp.GRAVITY_ACCELERATION

        # Stress scaling in the coordinate system of the stress tensor.
        # Note: Scale with the face areas
        diag_stress = np.vstack((max_hor_stress, min_hor_stress)) * g.face_areas[bf]

        # Normalized normal vectors.
        # Their direction is not clear from this expression, but this is anyhow handled
        # below.
        # We could also have dropped face area scaling above, and not normalized vectors,
        # but the current approach is sometimes more convenient for debugging.
        n = g.face_normals[:, bf]
        nn = np.array([nf / np.linalg.norm(nf) for nf in n.T]).T[:2]

        # Project normal vectors to the coordinate system of the principal stresses,
        # scales with stresses, and project back again
        proj_stress = R.T.dot(diag_stress * R.dot(nn))

        # Values for all Nd components, facewise
        values = np.zeros((gb.dim_max(), g.num_faces))
        # Scaling of the stresses
        values[:2, bf] = proj_stress / self.force_scale

        # Sanity check - only the lateral sides should have non-zero conditions
        assert np.sum(np.abs(values).sum(axis=0) > 1e-10) == (
            north.size + west.size + south.size + east.size
        )

        # Ensure that the boundary conditions on the lateral sides all are compressive.
        # From the directions of the principal stresses, and the definition of the
        # west, south, east and north boundary faces, we know that the boundary
        # condition should be pointing in the positive x-direction for west and south
        # and negative condition for east and north
        hit = values[0, west] < 0
        values[:, west[hit]] *= -1

        hit = values[1, south] < 0
        values[:, south[hit]] *= -1

        hit = values[0, east] > 0
        values[:, east[hit]] *= -1

        hit = values[1, north] > 0
        values[:, north[hit]] *= -1

        # Enforce free boundary at the top
        values[:, top] = 0
        # Zero conditions in x and y directions at the bottom
        values[:, bottom] = 0

        # Lithostatic stress in the z-directions.
        values[2, bottom] = -(
            pp.GRAVITY_ACCELERATION
            * self.rock_density
            * fc[2, bottom]
            * g.face_areas[bottom]
            / self.force_scale
        )
        # Lithostatic pressure should be downwards
        assert np.all(values[2, bottom] > 0)
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def bc_type_flow(self, g, gb):
        """ Set type of boundary conditions for the flow problem:
            Dirichlet at vertical sides and top, no-flow at bottom.
        """
        if g.dim < gb.dim_max():
            all_bf = self.domain_boundary_sides(g, gb)
            bc = pp.BoundaryCondition(g)
        else:
            all_bf, bottom, *rest = self.domain_boundary_sides(g, gb)
            bc = pp.BoundaryCondition(g, all_bf, "dir")
            bc.is_neu[bottom] = True
            bc.is_dir[bottom] = False

        # Neumann condition on fracture faces
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[frac_face] = True

        # On lower-dimensional grids, the default type, Neumann, is assigned.
        # We could have set Dirichlet conditions on fracture and intersection faces on
        # the surface.

        return bc

    def bc_values_flow(self, g, gb, gravity, time_step):
        """ Set values for flow boundary conditions: 0 at top and bottom, hydrostatic
        at lateral sides.
        """
        if g.dim < gb.dim_max():
            # No-flow conditions on the fracture tips.
            # Strictly speaking,
            return np.zeros(g.num_faces)
        else:
            values = np.zeros(g.num_faces)
            if gravity:
                all_bf, bottom, top, *lateral_sides = self.domain_boundary_sides(g, gb)
                fz = g.face_centers[2]

                # Hydrostatic pressure, given by \rho g z, but divided but the scaling
                # of the scalar variable
                for side in lateral_sides:
                    # Minus sign to enforce increasing pressure with increasing depth
                    values[side] = -(
                        fz[side]
                        * pp.Water().density()
                        * pp.GRAVITY_ACCELERATION
                        # The pressure is divided by force_scale throughout
                        / self.force_scale
                        # Divide by time step to compensate for the dt scaling in
                        # ImplicitMpfa.
                        / time_step
                    )
            return values

    def set_flow_parameters(self, gb, time_step, gravity=True, **kwargs):
        """
        Define the permeability, apertures, boundary conditions and sources.
        """
        # Values for matrix permeability, porosity
        matrix_permeability = 1e-12
        matrix_porosity = 0.1

        fracture_porosity = 1

        # Estimated water compressibility
        water_compressibility = 5 * 1e-10 / pp.PASCAL * self.force_scale

        # As the simulation tool has no well model, the pressure in the injection cell,
        # which is used to calibrate the flow properties, is highly dependent on the
        # aperture in the injection fracture, as well as on the area of the injection
        # cell. The latter again depends on the grid resolutoin in the vertical
        # direction, as described in simulation_master.Simulator.__init__()
        # Experimentation has shown the following reasonable values for the aperture
        # in the injection fracture:
        #   Case 0: 3.0e-3
        #   Case 1: 4.0e-3
        injection_cell_aperture = 1e-2

        # Map from fracture numbers to aperture values. This controls apertures, thus
        # permeabilities for all fractures (normal and tangential) and intersections.
        # Comments:
        #   1) Fractures 0 and 1 are parallel, and are assumed to have an aperutre of
        #       1cm.
        #   2) Fractures 2-4 also have equal, and fairly high permeabilities (they have
        #       almost equal orientation to the regional stress field).
        #   3) Fracture 5 is less favorably oriented, and is persumed to have lower
        #       aperture.
        fracture_aperture_map = {
            0: 1.0 * pp.CENTIMETER,
            1: 1.0 * pp.CENTIMETER,
            2: 1.0 * pp.CENTIMETER,
            3: 1.0 * pp.CENTIMETER,
            4: 1.0 * pp.CENTIMETER,
            5: 1.0 * pp.CENTIMETER,
        }

        # Index of blocking fracture
        BLOCKING_FRACTURE_INDEX = 5

        # Compute the fracture tangential permeability before the main asignment loop
        # (next for-loop). The permaebilities are used also for 1d intersection grids.
        # In this way, we do not run into trouble if a 1d grid is encountered before
        # its 2d neighbor.
        fracture_permeability_map = {}

        for frac_num, aperture in fracture_aperture_map.items():
            # The specific volume of a fracture equals its aperture
            specific_volume = aperture

            if frac_num == BLOCKING_FRACTURE_INDEX:
                # Explicitly set low permeability for fracture number 5
                # kxx = 1e-2 * matrix_permeability
                kxx = np.power(aperture, 2) / 12
            else:
                # Permeability by parallel plate model
                kxx = np.power(aperture, 2) / 12

            # Store the permeability of this fracture
            fracture_permeability_map[frac_num] = kxx

        # Store calculated aperture and tangential permeability of intersection grids.
        # NOTE: for keys in these maps, we will use the intersection grids (contrary
        # to the corresponding dictionaries for fractures, which use numbers so that
        # they are available without access to the fracture grid).
        intersection_aperture_map, intersection_permeability_map = {}, {}

        fluid = pp.Water()

        # Loop over all grids in the bucket, populate the parameter dictionary
        for g, d in gb:

            # First, assign specific volume and permeability to the grid

            # Divide by the dynamic viscosity of water, and introduce force scaling
            inverse_viscosity_force_scale = (
                np.ones(g.num_cells) / fluid.dynamic_viscosity() * self.force_scale
            )
            unit_vector = np.ones(g.num_cells)

            # Set specific volumes and permeabilities for all objects
            if g.dim == 3:
                specific_volume = 1 * unit_vector
                # Create a permeability tensor that also incorporates the fluid viscosity
                permeability = pp.SecondOrderTensor(
                    inverse_viscosity_force_scale
                    * matrix_permeability
                    * specific_volume
                )

            elif g.dim == 2:
                # Pull aperture from the specified values
                kxx = fracture_permeability_map[g.frac_num] * unit_vector
                specific_volume = fracture_aperture_map[g.frac_num] * unit_vector

                if g.frac_num == 1:
                    # Tune the pressure response by the aperture in the injection fracture
                    kxx[self.inj_cell] = np.power(injection_cell_aperture, 2) / 12
                    specific_volume[self.inj_cell] = injection_cell_aperture

                permeability = pp.SecondOrderTensor(
                    inverse_viscosity_force_scale * kxx * specific_volume
                )

            elif g.dim == 1:
                # Get the high-dimensional neighbors of g
                neighbors = gb.node_neighbors(g, only_higher=True)

                # If any of the neighbors has frac_num equal to BLOCKING_FRACTURE_INDEX,
                # the intersection is close to a barrier
                close_to_barrier = np.any(
                    [ng.frac_num == BLOCKING_FRACTURE_INDEX for ng in neighbors]
                )

                # Get apertures of all neighboring fractures
                aperture_of_high_dim_neigh = [
                    fracture_aperture_map[ng.frac_num] for ng in neighbors
                ]
                # The aperture of the 1d line is taken as the mean of the apertures of
                # the adjacent cells.
                aperture = np.mean(aperture_of_high_dim_neigh)

                # Specific volume is the product of neighboring apertures.
                specific_volume = np.power(aperture, gb.dim_max() - g.dim)

                if close_to_barrier:
                    # The permeability is inherited from the barrier fracture, no 5
                    kxx = fracture_permeability_map[BLOCKING_FRACTURE_INDEX]

                else:
                    # Standard permeability calculation
                    kxx = np.power(aperture, 2) / 12

                # Store calculated intersection aperture
                intersection_aperture_map[g] = aperture
                # NOTE: Store permeability without scaling with self.force_scale
                intersection_permeability_map[g] = kxx

                # Create a permeability tensor that also incorporates the fluid
                # viscosity and force scaling
                permeability = pp.SecondOrderTensor(
                    inverse_viscosity_force_scale * kxx * specific_volume
                )

            # Done with permeability and specific volumes

            # Set porosity in matrix, use unit value for fractures and intersections;
            # it is not really clear what else do do for g.dim < 3
            if g.dim == gb.dim_max():
                porosity = matrix_porosity * unit_vector
            else:
                porosity = fracture_porosity * unit_vector

            # Dummy source values. The real value is set in the time loop (see self.iterate()).
            source = np.zeros(g.num_cells)
            Nd = gb.dim_max()
            vector_source = np.zeros(g.num_cells * Nd)
            if gravity:
                vector_source[Nd - 1 :: Nd] = (
                    -pp.GRAVITY_ACCELERATION * fluid.density() / self.force_scale
                )

            # Set boundary values and conditions
            bc = self.bc_type_flow(g, gb)
            bc_val = self.bc_values_flow(g, gb, gravity, time_step)

            # Initialize flow problem
            specified_parameters = {
                "bc": bc,
                "bc_values": bc_val,
                "source": source,
                "second_order_tensor": permeability,
                "mass_weight": porosity * water_compressibility * specific_volume,
                "biot_alpha": self.biot_alpha(gb, g),
                "time_step": time_step,
                "ambient_dimension": Nd,
                "vector_source": vector_source,
            }
            pp.initialize_data(g, d, self.scalar_parameter_key, specified_parameters)
            # End of parameter assignment for this grid

        # Parameter assignment for the interfaces. This is simpler - we just need to
        # get the normal permeability.
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            mg = d["mortar_grid"]
            if g_l.dim == 2:
                # This is a fracture-matrix interface
                # Set the normal permeability from the tangential one
                aperture = fracture_aperture_map[g_l.frac_num]
                kt = fracture_permeability_map[g_l.frac_num]
                kn = kt / (0.5 * aperture) * np.ones(mg.num_cells)

            else:
                # This is an interface between a fracture intersection and the full
                # interface.
                # Detect whether this is an interface to an intersection line, and the
                # intersection is with the barrier fracture
                close_to_barrier = False
                for e2, d2 in gb.edges_of_node(g_l):
                    if gb.nodes_of_edge(e2)[1].frac_num == BLOCKING_FRACTURE_INDEX:
                        close_to_barrier = True

                if close_to_barrier:
                    # Use aperture of the low-permeable fracture only
                    aperture = fracture_aperture_map[BLOCKING_FRACTURE_INDEX]
                    # kt = fracture_permeability_map[BLOCKING_FRACTURE_INDEX]
                    # Low normal diffusivity for the blocking fracture
                    kt = 1e-2 * matrix_permeability
                    kn = kt / (0.5 * aperture) * np.ones(mg.num_cells)

                else:
                    aperture = intersection_aperture_map[g_l]
                    kt = intersection_permeability_map[g_l]
                    kn = kt / (0.5 * aperture) * np.ones(mg.num_cells)

                # The specific volume of the interface (really mortar grid) is
                # inherited from the fracture.
                # Specific volume equals aperture for fractures
                # This is not needed if g_l.dim == 2, since the specific volume of
                # g_h (the matrix grid) is unity.
                specific_volume_h = fracture_aperture_map[g_h.frac_num]
                kn *= specific_volume_h

            # Divide normal permeability by viscosity, and scale with force_scale
            kn *= self.force_scale / fluid.dynamic_viscosity()
            pp.initialize_data(
                mg, d, self.scalar_parameter_key, {"normal_diffusivity": kn}
            )

    def sources(self, gb):
        # Minor thing:
        # def source_scalar(self):
        #     gb = self.gb

        # I would consider splitting this into one method which identifies (and tags) the
        # injection cell, and one method (source_scalar) returning the rates. The former
        # would only be called once (after grid construction, in prepare_simulation or similar).

        # Find fracture grid and cell index of inlet
        well_path = self._read_well_path()
        well_path[:, 5:7] -= self._well_coordinate_surface()[:2]

        # Set the leakage point to 2500 meters, this corresponds to the region with
        # highest probability of leakage, according to televiewer imaging.
        z_coords_leakage_points = np.array([2500])

        # The well is described as a picewise linear curve in 3d. We plan to first
        # identify the segment of the leakage points, then find the corresponding
        # x and y coordinates of the segment, and finally find the cell in the injection
        # fracture that is closest to this xyz coordinate.

        # Depth of the measurement points, measured with 0 on the surface, and
        # increasing downwards
        z_well = well_path[:, 4]
        # Depth of the end of the segments
        z_well_shifted = np.hstack((z_well[1:], 5000))

        inj_frac, inj_cell = [], []

        for z in z_coords_leakage_points:
            # Find the segment where the fracture is located
            ind = np.where(np.logical_and(z > z_well, z < z_well_shifted))[0][0]

            # X and Y coordinates are set according to the start of the segments.
            # Interpolation could have been used.
            x = well_path[ind, 5]
            y = well_path[ind, 6]

            # Inlet point.
            p = np.array([x, y, -z]).reshape((-1, 1))

            # Next find the cell in the target fracture closest to the inlet point
            frac_ind = -1
            cell_ind = -1
            min_dist = np.inf

            for g, _ in gb:
                if g.dim != 2:
                    continue
                # We have decided to put the injection point at fracture number 5
                # (1-offset, thus frac_num=4), so disregard other fractures.
                # However, we may also use a single fracture for debugging purposes -
                # if so, we need to put the source in that fracture.
                if len(gb.grids_of_dimension(2)) > 1 and g.frac_num != 4:
                    continue

                dist = pp.distances.point_pointset(p, g.cell_centers)
                closest = np.argmin(dist)
                if np.min(dist) < min_dist:
                    min_dist = np.min(dist)
                    frac_ind = g.frac_num
                    cell_ind = closest

            inj_frac.append(frac_ind)
            inj_cell.append(cell_ind)

        # Injection rates (L/S) - this should be expanded if more inlets are included
        inj_rate = np.array([43])
        inj_rate = inj_rate / inj_rate.sum()

        self.inj_cell = inj_cell

        return {
            "rate_ratios": inj_rate / inj_rate.sum(),
            "fracture": np.asarray(inj_frac),
            "cell": inj_cell,
        }

    def biot_alpha(self, gb, g):
        if g.dim == gb.dim_max():
            return 0.8
        else:
            return 1

    def set_mechanics_parameters(self, gb, time_step):
        """ Set fault friction coefficients, elastic moduli
        """
        for g, d in gb:

            if g.dim == 3:

                # Define boundary condition on sub_faces
                bc = self.bc_type_mech(g, gb)

                bc_val = self.bc_values_mech(g, gb)

                # The elastic properties of the rock are available through the seismic
                # velocities, tabularized as functions of depth.

                # Depths for velocity estimates
                speed_depth = np.array([0, -1000, -2000, -3000, -4000, -6000])
                # Speed of the primary waves, at the depths in speed_depth
                p_speed = (
                    np.array([3.53, 4.47, 5.16, 5.60, 5.96, 6.50])
                    * pp.KILOMETER
                    / pp.SECOND
                )
                # Speed of the secondary waves, at the depths in speed_depth
                s_speed = (
                    np.array([1.98, 2.51, 2.90, 3.15, 3.35, 3.65])
                    * pp.KILOMETER
                    / pp.SECOND
                )

                # Cell center depth, we will evaluate the seismic velocity speed there
                cell_depth = g.cell_centers[2]

                # Convert seismic velocities to Lame parameters, evaluated values at
                # cell center depths
                # Zimmermann p333-334

                # mu = rho * Vs^2
                mu = self.rock_density * np.power(
                    np.interp(cell_depth, speed_depth[::-1], s_speed[::-1]), 2
                )
                # lmbda = rho * Vp^2 - 2 * mu
                lmbda = (
                    self.rock_density
                    * np.power(
                        np.interp(cell_depth, speed_depth[::-1], p_speed[::-1]), 2
                    )
                    - 2 * mu
                )

                # Scaling of parameters
                mu /= self.force_scale
                lmbda /= self.force_scale

                stiffness = pp.FourthOrderTensor(mu, lmbda)

                body_force = np.zeros((g.dim, g.num_cells))
                # Gravitational body force
                body_force[2] = (
                    self.rock_density
                    * pp.GRAVITY_ACCELERATION
                    * g.cell_volumes
                    / self.force_scale
                )
                body_force = body_force.ravel(order="F")

                biot_alpha = self.biot_alpha(gb, g)
                # Should there be a biot coefficient?
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": body_force,
                        "fourth_order_tensor": stiffness,
                        "biot_alpha": biot_alpha,
                        "max_memory": 7e7,
                        "time_step": time_step,
                    },
                )
            elif g.dim == 2:
                friction = 0.4 * np.ones(g.num_cells)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction, "time_step": time_step},
                )

        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key, {})


class FlowModel:
    """ This is a class dedicated to flow simulations for RN34.
    Could be stripped down and included in the core.
    """

    def __init__(self, params, target_date="march_29"):

        z_coord = params["z_coordinates"]
        fracture_files = params["fracture_files"]

        # Time step size
        self.time_step = 150 * pp.SECOND

        # Identifiers of terms in equation
        self.diffusion_term_flow = "pressure_diffusion"
        self.source_term_flow = "source_term_pressure"
        self.accumulation_term_flow = "accumulation_term_pressure"
        self.coupling_operator_term_flow = "coupling_operator_pressure"

        # Identifiers of parameter keywords
        self.scalar_parameter_key = "flow"
        self.transport_parameter_keyword = "transport"

        # Variables
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_p"

        self.sim_data = RN34SimulationData(
            scalar_parameter_key=self.scalar_parameter_key,
            mechanics_parameter_key="",
            z_coord=z_coord,
            fracture_files=fracture_files,
            force_scale=1,
        )

        # Load observation data
        # This is used for plotting of observed data, but also to control the injection
        # strength (on and off)
        if target_date == "apr_7":
            (
                inj_rate,
                length_periods,
                observed_pressure,
                observed_time,
                num_data_points_periods,
            ) = observations.load_pressure_observations_april_7()
        elif target_date == "march_29":
            t, p = observations.load_pressure_observations_march_29()
        else:
            raise ValueError("Unknown target date " + target_date)

        self.target_date = target_date
        self.observed_pressure = p
        self.observation_time = t

    def simulate(self, do_export=False, **kwargs):
        # Set up and simulate a flow experiment

        # Grid
        self.gb = self.sim_data.create_iceland_grid()
        # Well information. This contains some information that is redundant, due to the
        # relatively simple calibration approach we ended up using
        self.well_data = self.sim_data.sources(self.gb)

        # Injection rate, set in accordance with data from ISOR
        inj_rate = 43 * pp.KILOGRAM / pp.SECOND

        # Set variables, discretization schemes, and parameters
        self._set_variables_discretization()
        self.sim_data.set_flow_parameters(
            self.gb, self.time_step, gravity=False, **kwargs
        )
        # Discretize the full problem
        self.discretize(dt=self.time_step, direct_solver=True)
        # Further treatment of source parameters - needed for legacy reasons
        self.update_source_parameters()

        T_full = self.observation_time[0]
        time_step_counter = 0

        # Initialize pressure vector
        pressure = 0 * self.rhs_source

        if do_export:
            folder = kwargs.get("flow_export_folder", "flow_viz")
            self.exporter = pp.Exporter(self.gb, "flow_sim", folder_name=folder)
            self.export_time_steps = [T_full]
            self.assembler.distribute_variable(
                pressure,
                variable_names=[self.scalar_variable, self.mortar_scalar_variable],
            )
            self.exporter.write_vtk(
                self.scalar_variable, time_step=self.export_time_steps[-1]
            )

        # Store the pressure profile
        num_source_points = self.well_data["rate_ratios"].size
        pressure_in_sources = np.zeros((1, num_source_points))

        all_time_steps = []

        for time_step in range(self.observation_time.size - 1):

            # Sub time interval, between two time stamps where the pressure value is
            # available.
            # This is in part motivated by the (currently not used) approach of an
            # automatic calibration of model parameters
            final_T = (
                self.observation_time[time_step_counter + 1]
                - self.observation_time[time_step_counter]
            )
            # I would say all of this belongs in a time dependent source_scalar() method.
            # It is also quite difficult to follow what goes on here. If it is not used,
            # I suggest to simplify.
            # Adjust the injection source strength. This will in practice switch off
            # the source at the right time.
            self.rhs_source *= 0
            if self.target_date == "march_29":
                if self.observation_time[time_step_counter] < 0:
                    self.rhs_source[self.well_data["dof_injection"]] = (
                        inj_rate * self.well_data["rate_ratios"] * self.time_step
                    )
                else:
                    self.rhs_source[self.well_data["dof_injection"]] = 0
            else:
                raise ValueError(
                    "This model is tailored for simulating the March 29 event"
                )

            # Propagate in time through this sub interval
            pressure, pressure_in_sources_this_period = self._do_time_steps(
                pressure, self.time_step, end_time=final_T, do_export=do_export
            )

            # Monitor the pressure in the source cell for this interval.
            # Main motivation is the (unused) automatic calibration approach.
            pressure_in_sources = np.vstack(
                (pressure_in_sources, pressure_in_sources_this_period[-1])
            )
            time_step_counter += 1
            T_full += final_T

            all_time_steps.append(T_full)

        if do_export:
            self.exporter.write_pvd(self.export_time_steps)

        return pressure_in_sources

    def calibration_run(self, do_plot=True, **kwargs):
        """ Run the simulation, but also plot the pressure profile in the injection cell,
        together with the observed well pressure.
        Also compute the mismatch between observed and measured well pressure, used
        in (unused) automatic calibration.
        """
        logger.setLevel(logging.CRITICAL)

        pressure_in_sources = self.simulate(**kwargs)

        mean_computed_pressure = pressure_in_sources.mean(axis=1)
        # Why first multiply and then divide by bar??
        pressure_increase = (
            self.observed_pressure - self.observed_pressure[0]
        ) * pp.BAR

        pressure_error = np.linalg.norm(
            (mean_computed_pressure - pressure_increase) / pp.BAR
        )

        print("Pressure error " + str(pressure_error))

        if do_plot:
            plt.figure()
            plt.plot(self.observation_time, mean_computed_pressure / pp.BAR, "r")
            plt.plot(self.observation_time, pressure_increase / pp.BAR, "b")
            plt.show()

        pressure_error *= -1

        # This was needed at some point during automatic calibration, currently not
        # relevant.
        if not np.isfinite(pressure_error) or pressure_error < -1e10:
            pressure_error = -1e10

        return pressure_error

    def _set_variables_discretization(self, use_mpfa=True):
        """ Set keywords that control discretization
        """
        if use_mpfa:
            # This is too misleading. Rename to xpfa, FV_disc, flux_disc, fa or similar
            fv_discr = pp.Mpfa(self.scalar_parameter_key)
        else:
            fv_discr = pp.Tpfa(self.scalar_parameter_key)

        source_disrcetization = pp.ScalarSource(self.scalar_parameter_key)
        mass_discretization = pp.MassMatrix(self.scalar_parameter_key)

        edge_discretization = pp.RobinCoupling(
            self.scalar_parameter_key, fv_discr, fv_discr
        )

        # Loop over the nodes in the GridBucket, define primary variables and discretization schemes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES] = {self.scalar_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {
                self.scalar_variable: {
                    self.diffusion_term_flow: fv_discr,
                    self.source_term_flow: source_disrcetization,
                    self.accumulation_term_flow: mass_discretization,
                }
            }
        # Loop over the edges in the GridBucket, define primary variables and discretizations
        for e, d in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar_scalar_variable: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_operator_term_flow: {
                    g1: (self.scalar_variable, self.diffusion_term_flow),
                    g2: (self.scalar_variable, self.diffusion_term_flow),
                    e: (self.mortar_scalar_variable, edge_discretization),
                }
            }

    def discretize(self, dt, direct_solver=True, **kwargs):
        assembler = pp.Assembler(self.gb)
        assembler.discretize()

        # Assemble the linear system, using the information stored in the GridBucket
        A_terms, b = assembler.assemble_matrix_rhs(add_matrices=False)
        coupling_operator_term_flow = self.coupling_operator_term_flow + (
            "_"
            + self.mortar_scalar_variable
            + "_"
            + self.scalar_variable
            + "_"
            + self.scalar_variable
        )
        accumulation_term_flow = (
            self.accumulation_term_flow + "_" + self.scalar_variable
        )
        source_term_flow = self.source_term_flow + "_" + self.scalar_variable
        diffusion_term_flow = self.diffusion_term_flow + "_" + self.scalar_variable

        # Build matrix
        lhs = (
            A_terms[accumulation_term_flow]
            + A_terms[coupling_operator_term_flow]
            + dt * A_terms[diffusion_term_flow]
        )
        lhs = lhs.tocsc()
        rhs_source = dt * b[source_term_flow]

        # Use a factorized linear solver. This is feasible under the assumption that the
        # time step stays the same throughout the simulation.
        pressure_solve = spla.factorized(lhs)

        self.pressure_solve = pressure_solve
        self.rhs_source = rhs_source
        self.A_terms = A_terms
        self.assembler = assembler

    def update_source_parameters(self):
        """ This is mainly needed for (currently unused) functionality to calibrate the
        parameters automatically.
        Purge if not used
        """

        dof_start = np.cumsum(np.hstack((0, self.assembler.full_dof)))

        num_inj_cells = len(self.well_data["fracture"])
        dof_of_injection_cell = np.zeros(num_inj_cells, dtype=np.int)
        volume_of_injection_cell = 0 * np.zeros(num_inj_cells)

        counter = 0
        # Obtain dofs corresponding to the cells of the sources
        for key, block_ind in self.assembler.block_dof.items():
            g = key[0]
            if isinstance(g, pp.Grid) and g.dim == 2:
                hit = np.where(self.well_data["fracture"] == g.frac_num)[0]
                for ind in hit:
                    cell_ind = self.well_data["cell"][ind]
                    dof_of_injection_cell[counter] = dof_start[block_ind] + cell_ind
                    volume_of_injection_cell[counter] = g.cell_volumes[cell_ind]
                    counter += 1

        self.well_data.update(
            {
                "dof_injection": dof_of_injection_cell,
                "volume_injection_cell": volume_of_injection_cell,
            }
        )

    def _do_time_steps(self, pressure_prev, dt, end_time=None, do_export=False):
        """ Helper method for time stepping. Intended for time propagation between two
        points in time when a pressure observation is available.
        
        The time step size is set as 1) small, and 2) so that all observation times
        can be hit with a fixed time step size.
        
        Parameters:
            pressure_prev (np.array): Pressure state from the previous time stepping
            dt (double): time step size.
            end_time (double, optional): Length of this set of sub timesteps. If not 
                provided, it is set equal to the time step size, that is, a single step
                is taken.
            do_export (boolean, optional): If True, the time evolution is dumped to
                Paraview.
                
        Returns:
            np.array : Pressure solution at the end of this simulation period.
            np.array: Pressure in the feedpoints (as identified by self.well_data) for
                each time step.
        """

        if end_time is None:
            end_time = dt

        pressure = pressure_prev

        t = 0

        pressure_in_sources = []

        accumulation_term_flow = (
            self.accumulation_term_flow + "_" + self.scalar_variable
        )

        while t <= (end_time + 0.01 * dt):
            # Right hand side is formed by the sources and the accumulation term
            rhs = self.rhs_source + self.A_terms[accumulation_term_flow] * pressure

            # Left hand side is constant, use the precomputed matrix factorization
            pressure = self.pressure_solve(rhs)

            # Depending on the linear solver, the pressure solver may return more than
            # just the solution vector. Dump everything else.
            if isinstance(pressure, tuple):
                pressure = pressure[0]
            pressure_in_sources.append(pressure[self.well_data["dof_injection"]])

            if do_export:
                self.assembler.distribute_variable(
                    pressure,
                    variable_names=[self.scalar_variable, self.mortar_scalar_variable],
                )
                full_time = dt + self.export_time_steps[-1]
                self.exporter.write_vtk(self.scalar_variable, time_step=full_time)
                self.export_time_steps.append(full_time)

            t += dt

        return pressure, np.array(pressure_in_sources)


class BiotMechanicsModel(ContactMechanicsBiot):
    def __init__(self, params):
        super(BiotMechanicsModel, self).__init__(params)

        self.z_coord = params["z_coordinates"]
        self.fracture_files = params["fracture_files"]

        self.time_step = params["time_step"]
        self.scalar_scale = 1e9
        # Scaling coefficients
        self.length_scale = (
            1  # Consider using this if you have issues with condition numbers
        )

        self.sim_data = RN34SimulationData(
            self.scalar_parameter_key,
            self.mechanics_parameter_key,
            self.z_coord,
            self.fracture_files,
            self.scalar_scale,
            time_step=self.time_step,
        )

        self.high_rate = 0
        self.low_rate = 0

        self.time = 0
        self.do_export = params.get("do_export", True)
        self.export_every = params.get("export_every", 1)
        self.prev_export_time = 0

        self.min_time_step = 30 * pp.SECOND
        self.init_time_step = self.time_step

        if self.do_export:
            self.export_file_name = params.get("export_file_name", "sim_result")
            self.export_folder = params.get("export_folder", "biot_contact_viz")

        self.end_time = params["end_time"]

    def set_export(self):
        if not self.do_export:
            return

        self.displacement_exporter = pp.Exporter(
            self._nd_grid(),
            self.export_file_name + "_displacement",
            folder_name=self.export_folder,
        )
        self.export_times = [self.time]
        d = self.gb.node_props(self._nd_grid())

        u = d[pp.STATE][self.displacement_variable]

        export = {"u_x": u[::3], "u_y": u[1::3], "u_z": u[2::3]}

        self.displacement_exporter.write_vtk(export, time_step=self.export_times[-1])

        gb_2d = pp.GridBucket()
        gb_2d.add_nodes(self.gb.grids_of_dimension(2))
        for g, d in gb_2d:
            d[pp.STATE] = {}

        self.contact_exporter = pp.Exporter(
            gb_2d,
            self.export_file_name + "_fracture_jumps",
            folder_name=self.export_folder,
        )

        self.flow_exporter = pp.Exporter(
            self.gb, self.export_file_name + "_pressure", folder_name=self.export_folder
        )

        self.flow_exporter.write_vtk(
            data=self.scalar_variable, time_step=self.export_times[-1]
        )

    def export_step(self):
        if not self.do_export:
            return

        if self.time >= self.prev_export_time + self.export_every * self.time_step:
            d = self.gb.node_props(self._nd_grid())

            u = d[pp.STATE][self.displacement_variable]

            export = {"u_x": u[::3], "u_y": u[1::3], "u_z": u[2::3]}
            self.displacement_exporter.write_vtk(export, time_step=self.time)

            # Export flow state
            # NB: We could have
            self.flow_exporter.write_vtk(data=self.scalar_variable, time_step=self.time)

            # Export contact state
            self.store_contact_state(prefix="current")
            for e, d_e in self.gb.edges():
                g = e[1]
                if g.dim != self.Nd - 1:
                    continue

                d = self.gb.node_props(g)
                nc = g.num_cells
                proj = d_e["tangential_normal_projection"]

                difference_previous_disp = (
                    d[pp.STATE]["current_displacement_state"]
                    - d[pp.STATE]["previous_displacement_state"]
                )
                difference_reference_disp = (
                    d[pp.STATE]["current_displacement_state"]
                    - d[pp.STATE]["reference_displacement_state"]
                )

                contact_force = d[pp.STATE][self.contact_traction_variable]

                def nrm(vec):
                    # Cell-wise norm of a vector
                    return np.sqrt(np.sum(vec.reshape((nc, -1)) ** 2, axis=1))

                def tang_comp(vec):
                    # Tangential components of an Nd-vector. Assumed to be in the local
                    # coordinate system
                    vec_reshape = vec.reshape((self.Nd, nc), order="f")[: g.dim]
                    return vec_reshape.reshape((-1, 1), order="f")

                def norm_comp(vec):
                    # Normal component of an Nd-vector. Assumed to be in the local
                    # coordinate system
                    return vec[self.Nd - 1 :: self.Nd]

                max_jump = 0

                for ge, de in self.contact_exporter.gb:
                    if ge.frac_num == g.frac_num:

                        de[pp.STATE]["un_previous"] = (
                            proj.project_normal(nc) * difference_previous_disp
                        )
                        de[pp.STATE]["un_reference"] = (
                            proj.project_normal(nc) * difference_reference_disp
                        )

                        u_t_prev = (
                            proj.project_tangential(nc) * difference_previous_disp
                        )
                        u_t_ref = (
                            proj.project_tangential(nc) * difference_reference_disp
                        )

                        de[pp.STATE]["ut_previous_norm"] = nrm(u_t_prev)
                        de[pp.STATE]["ut_reference_norm"] = nrm(u_t_ref)

                        de[pp.STATE]["un"] = (
                            proj.project_normal(nc)
                            * d[pp.STATE]["current_displacement_state"]
                        )

                        de[pp.STATE]["contact_force_normal"] = norm_comp(contact_force)
                        de[pp.STATE]["contact_force_tangential_norm"] = nrm(
                            tang_comp(contact_force)
                        )

                        max_jump = np.maximum(max_jump, nrm(u_t_prev).max())

                        global_force = (
                            proj.project_tangential_normal(nc) * contact_force
                        )

                        de[pp.STATE]["contact_force_x"] = global_force[:: self.Nd]
                        de[pp.STATE]["contact_force_y"] = global_force[1 :: self.Nd]
                        de[pp.STATE]["contact_force_z"] = global_force[2 :: self.Nd]

                print(
                    f"\n Fracture number {g.frac_num}. Max tangential jump {max_jump}\n"
                )

            self.contact_exporter.write_vtk(
                data=[
                    "un_previous",
                    "un_reference",
                    "ut_previous_norm",
                    "ut_reference_norm",
                    "un",
                    "contact_force_normal",
                    "contact_force_tangential_norm",
                    "contact_force_x",
                    "contact_force_y",
                    "contact_force_z",
                ],
                time_step=self.time,
            )

            self.store_contact_state(prefix="previous")

            self.export_times.append(self.time)
            self.prev_export_time = self.time

    def export_pvd(self):
        self.displacement_exporter.write_pvd(np.array(self.export_times))
        self.flow_exporter.write_pvd(np.array(self.export_times))
        self.contact_exporter.write_pvd(np.array(self.export_times))

    def set_time_step(self, ts):
        self.time_step = ts
        self.sim_data.time_step = ts

    def set_parameters(self):
        """
        Set the parameters for the simulation.
        """
        # Set mechanics parameters
        self.sim_data.set_mechanics_parameters(self.gb, self.time_step)
        # Set flow parameters
        self.sim_data.set_flow_parameters(self.gb, self.time_step)

        # Finally, set the parameters for the sources

        # Special treatment of the source term
        # The scalar source alternates between high for 60 minutes and low for 30 minutes.
        length_cycle = 60 * pp.MINUTE + 30 * pp.MINUTE

        rest_time = self.time % length_cycle
        if self.time > 10 * pp.HOUR:
            rate = 0
        elif rest_time < 60.01 * pp.MINUTE:
            # This is safe also for the initialization, as both the high and low
            # rate is zero in that stage.
            rate = self.high_rate
        else:
            rate = self.low_rate

        logger.info(f"Injection rate set to {rate}")

        for g, d in self.gb:
            if g.dim == 2:
                source_vec = 0 * np.zeros(g.num_cells)
                hit = np.where(self.well_data["fracture"] == g.frac_num)[0]
                for ind in hit:
                    source_vec[self.well_data["cell"][ind]] = (
                        rate * self.well_data["rate_ratios"][ind] * self.time_step
                    )
            else:
                source_vec = 0 * np.zeros(g.num_cells)

            d[pp.PARAMETERS][self.scalar_parameter_key]["source"] = source_vec

    def initial_condition(self):
        super().initial_condition()

        rho_g = pp.GRAVITY_ACCELERATION * pp.Water().density() / self.scalar_scale

        for g, d in self.gb:
            # Initialize the pressure to hydrostatic
            initial_scalar_value = -rho_g * g.cell_centers[2]
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})

        self.store_contact_state(prefix="reference")
        self.store_contact_state(prefix="previous")

    def store_contact_state(
        self, prefix=None, mortar_displacement=None, contact_force=None
    ):
        # Store various information on the contact state, in a format that is well
        # suited for visualization.
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_l = e[1]
            if g_l.dim < self.Nd - 1:
                continue
            d_l = self.gb.node_props(g_l)

            if mortar_displacement is None:
                displacement_state = d[pp.STATE][self.mortar_displacement_variable]
            else:
                displacement_state = mortar_displacement[g_l.frac_num]

            displacement_jump = (
                mg.mortar_to_slave_avg(self.Nd)
                * mg.sign_of_mortar_sides(nd=3)
                * displacement_state
            )

            if contact_force is None:
                contact_state = d_l[pp.STATE][self.contact_traction_variable]
            else:
                contact_state = contact_force[g_l.frac_num]

            d_l[pp.STATE][prefix + "displacement_state"] = displacement_jump
            d_l[pp.STATE][prefix + "contact_force_state"] = contact_state

    def activate_sources(self):
        # RN data
        # injection rates are either 100 or 20 L/s in the stimulation experiment
        self.high_rate = 100
        self.low_rate = 20

    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self.gb = self.sim_data.create_iceland_grid(grid_to_vtu=True)
        self.Nd = self.gb.dim_max()
        self.well_data = self.sim_data.sources(self.gb)
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_export()

    def after_newton_convergence(self, solution, errors, iteration_counter):
        self.assembler.distribute_variable(solution)
        self.save_mechanical_bc_values()
        self.export_step()

    def assign_discretizations(self):
        super().assign_discretizations()
        for e, d in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                couplings = d[pp.COUPLING_DISCRETIZATION][self.friction_coupling_term]
                contact_discr = couplings[(g_h, g_l)][1].discr_slave
                contact_discr.tol = 1e-6

    def before_newton_loop(self):
        # Set new parameters. This will also adjust the source term
        self.set_parameters()

        logger.info("\n ---------------------- \n")
        logger.info(f"Start new newton step. Time: {self.time}\n\n")

    def after_simulation(self):
        self.export_pvd()

    def assemble_and_solve_linear_system(self, tol):

        A, b = self.assembler.assemble_matrix_rhs()
        logger.debug("Max element in A {0:.2e}".format(np.max(np.abs(A))))
        logger.debug(
            "Max {0:.2e} and min {1:.2e} A sum.".format(
                np.max(np.sum(np.abs(A), axis=1)), np.min(np.sum(np.abs(A), axis=1))
            )
        )
        import time

        tic = time.time()
        A.indices = A.indices.astype(np.int64)
        A.indptr = A.indptr.astype(np.int64)

        x = spla.spsolve(A, b, use_umfpack=True)
        print(f"      UMFPACK time: {time.time() - tic}\n")
        return x

        if self.linear_solver == "direct":
            return spla.spsolve(A, b)
        elif self.linear_solver == "iterative":
            raise ValueError("Not available.")
