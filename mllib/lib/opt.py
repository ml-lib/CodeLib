"""
Optimization module.

**Available routines:**

- class ``TSP``: Solve the Traveling salesman problem using integer
  programming/nearest neighbour algorithm.
- class ``Transport``: Solve the  Transportation problem using integer
  programming.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 28, 2021
"""

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

# pylint: disable=invalid-name
# pylint: disable-msg=too-many-arguments
# pylint: disable=too-many-instance-attributes

# =============================================================================
# --- Import libraries
# =============================================================================

from typing import List, Tuple, Dict

import copy
import math
import pandas as pd
import numpy as np
import pulp

# =============================================================================
# --- User defined functions
# =============================================================================


class TSP:
    """
    Traveling salesman problem.

    Module for `Traveling salesman problem
    <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ using
    integer programming or nearest neighbour algorithm.

    Methods
    -------
    :func:`~opt.TSP.integer_program`
      Determining optimal path using integer programming.

    :func:`~opt.TSP.nearest_neighbour`
      Determining optimal path using nearest neighbour algorithm.

    :func:`~opt.TSP.solve`
      Determining optimal path using integer programming or nearest neighbour
      algorithm based on count of locations.

    """

    _paired_loc = None

    @staticmethod
    def haversine_np(lon1: List[float],
                     lat1: List[float],
                     lon2: List[float],
                     lat2: List[float],
                     dist: str = "mi"
                     ) -> np.ndarray:
        """
        Haversine distance formula.

        Calculate the euclidean distance in miles between two points
        specified in decimal degrees using
        `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_.

        Parameters
        ----------
        lon1, lat1, lon2, lat2 : float

            Pair of Latitude and Longitude. All args must be of equal length.

        dist : str, `optional`

            Output distance in miles ('mi') or kilometers ('km')
            (the default is mi)

        Returns
        -------
        numpy.ndarray

            Euclidean distance between two points in miles.

        """
        if dist == "km":  # pragma: no cover
            R = 6372.8
        else:
            R = 3959.87433
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (np.sin(dlat / 2.0) ** 2
             + np.cos(lat1) * np.cos(lat2)
             * np.sin(dlon / 2.0) ** 2)
        op = R * 2 * np.arcsin(np.sqrt(a))
        return op

    @staticmethod
    def pair_dist(loc: List[str],
                  lat: List[float],
                  lon: List[float]
                  ) -> Tuple[List[str],
                             Dict[Tuple[str, str],
                                  float]]:
        """
        Create pairwise euclidean distance in miles between all locations.

        Parameters
        ----------
        loc : list

            A list containing the location names.

        lat : list

            List of latitude in decimal degrees.

        lon : list

            List of longitude in decimal degrees.

        Returns
        -------
        loc : list

            A list containing the location names.

        dist : dict

            A dictionary containg the pairwise locations as Key and distances
            as values.

        """
        df = pd.DataFrame(data={'loc': loc, 'x': lat, 'y': lon})
        df["key"] = 1
        df = pd.merge(df,
                      df,
                      how="outer",
                      on="key",
                      copy=False)
        df = df.drop(labels=["key"], axis=1)
        df = df[df["loc_x"] != df["loc_y"]]
        df["dist"] = TSP.haversine_np(df["y_x"], df["x_x"],
                                      df["y_y"], df["x_y"])
        df = df[["loc_x", "loc_y", "dist"]]
        df = dict(zip(zip(df.loc_x, df.loc_y), df.dist))
        return (loc, df)

    @staticmethod
    def integer_program(loc: List[str],
                        dist: Dict[Tuple[str, str],
                                   float],
                        debug_mode: bool = False
                        ) -> Tuple[str,
                                   float,
                                   zip]:
        """
        Travelling Salesman Problem using integer programming.

        Parameters
        ----------
        loc : list

            A list containing the location names.

        dist : dict

            A dictionary containg the pairwise locations as Key and distances
            as values.

        debug_mode : bool, optional, default : False

            Print log in console.

        Returns
        -------
        tuple

            Containing the following::

              Optimization solution status : str
              Objective function value : float
              Optimal path with distance : zip object containing
                                           - list of tuples
                                           - locations : str
                                           - distances : float

        """
        # Initiate IP formulation model
        model = pulp.LpProblem("Travelling_Salesman_Problem", pulp.LpMinimize)
        # Decision variables - 1 if chosen, 0 otherwise
        dv_leg = pulp.LpVariable.dicts("Decision variable leg", dist,
                                       lowBound=0, upBound=1, cat='Binary')
        # Objective function - Minimize total distance
        model += pulp.lpSum([dv_leg[(i, j)] * dist[(i, j)] for (i, j) in dist])
        # Constraints
        # Con 01 - 02: Each node should be entered and exited exactly once
        for k in loc:
            model += pulp.lpSum([dv_leg[(i, k)]
                                 for i in loc if (i, k) in dv_leg]) == 1
            model += pulp.lpSum([dv_leg[(k, i)]
                                 for i in loc if (k, i) in dv_leg]) == 1
        # Con 03: Eliminate subtours
        u = pulp.LpVariable.dicts("Relative position of each tour leg", loc,
                                  lowBound=0, upBound=len(loc)-1,
                                  cat='Integer')
        for i in loc:
            for j in loc:
                if (i != j) and (i != loc[0] and j != loc[0]) and\
                   (i, j) in dv_leg:
                    model += u[i] - u[j] <= (len(loc))*(1 - dv_leg[(i, j)]) - 1
        # Solve
        debug_mode = int(debug_mode is True)
        pulp.LpSolverDefault.msg = debug_mode
        model.solve()
        # Generate optimal path
        loc_left = copy.copy(loc)
        org = loc[0]
        tour = []
        tour.append(loc_left.pop(loc_left.index(org)))
        while loc_left:
            for k in loc_left:
                if dv_leg[(org, k)].varValue == 1:
                    tour.append(loc_left.pop(loc_left.index(k)))
                    org = k
                    break
        tour.append(loc[0])
        tour_legs = [dist[(tour[i-1], tour[i])] for i in range(1, len(tour))]
        return (pulp.LpStatus[model.status],
                sum(tour_legs),
                zip(tour, tour_legs))

    @staticmethod
    def nearest_neighbour(loc_dict: Dict[Tuple[str, str],
                                         float]
                          ) -> Tuple[str,
                                     float,
                                     zip]:
        """
        Travelling Salesman Problem using nearest neighbour algorithm.

        Parameters
        ----------
        loc_dict : dict

            A dictionary containg the pairwise locations as Key and distances
            as values.

        Returns
        -------
        tuple containing the following::

          Algorithm used : str
          Objective function value : float
          Optimal path with distance : zip object containing
                                      - list of tuples
                                      - locations : str
                                      - distances : float

        """
        loc_df = pd.concat([pd.DataFrame(loc_dict.keys(), columns=["loc1",
                                                                   "loc2"]),
                            pd.DataFrame(loc_dict.values(), columns=["dist"])],
                           axis=1, verify_integrity=True)
        loc_df = loc_df.sort_values(["loc1", "dist"])
        loc_dist = loc_df["loc1"].unique().tolist()
        loc_df = loc_df.set_index("loc1")
        obj_val = math.inf
        op_dist = None
        op_visited = None
        for loc in range(len(loc_dist)):
            unvisited = copy.copy(loc_dist)
            sp = unvisited[loc]
            visited = []
            dist = []
            i = sp
            while unvisited != []:
                if len(unvisited) > 1:
                    tmp = loc_df[loc_df.index == i]
                    tmp = tmp[tmp["loc2"].isin(unvisited)].reset_index()
                else:
                    tmp = loc_df[(loc_df.index == i)
                                 & (loc_df["loc2"] == sp)].reset_index()
                visited.append(i)
                dist.append(tmp["dist"][0])
                unvisited.remove(i)
                i = tmp["loc2"][0]
            if sum(dist) < obj_val:
                obj_val = sum(dist)
                op_dist = dist
                op_visited = visited
        return ("Nearest Neighbour", sum(op_dist), zip(op_visited, op_dist))

    def solve(self,
              loc: List[str],
              lat: List[float],
              lon: List[float],
              debug: bool = False
              ) -> Tuple[str,
                         float,
                         zip]:
        """
        Solve for TSP.

        Solve Travelling Salesman Problem using Integer Programming if
        locations are less than 50 else solve using Nearest neighbour
        algorithm.

        Parameters
        ----------
        loc : list

            A list containing the location names.

        lat : list

            List of latitude in decimal degrees.

        lon : list

            List of longitude in decimal degrees.

        debug : bool, optional, default : False

            Print log in console.

        Returns
        -------
        tuple containing the following::

          Algorithm used : str
          Objective function value : float
          Optimal path with distance : zip object containing
                                      - list of tuples
                                      - locations : str
                                      - distances : float

        """
        self._paired_loc = self.pair_dist(loc, lat, lon)
        if len(loc) < 50:
            op = self.integer_program(self._paired_loc[0],
                                      self._paired_loc[1],
                                      debug)
        else:
            op = self.nearest_neighbour(self._paired_loc[1])
        return op


class Transport():
    """
    Transportation problem.

    Module for `Transportation Problem
    <https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)>`_
    using integer programming.

    Parameters
    ----------
    loc : list

        A list containing the locations/nodes.

    demand : list

        A list containing the demand quantity for each node.

    supply : list

        A list containing the supply quantity for each node.

    lat : list

        A list containing the latitude for each node.

    lon : list

        A list containing the longitude for each node.

    cost : float

        Cost per unit per mile

    """

    def __init__(self,
                 loc: List[str],
                 demand: List[int],
                 supply: List[int],
                 lat: List[float],
                 lon: List[float],
                 cost: int):
        """Initialize variables."""
        self._loc = loc
        self._demand = demand
        self._supply = supply
        self._lat = lat
        self._lon = lon
        self._cost = cost
        self._ori_demand = demand
        self._ori_supply = supply
        self._inputs = None
        self.output = None

    @staticmethod
    def haversine_np(lon1: List[float],
                     lat1: List[float],
                     lon2: List[float],
                     lat2: List[float],
                     dist: str = "mi"
                     ) -> np.ndarray:
        """
        Haversine distance formula.

        Calculate the euclidean distance in miles between two points
        specified in decimal degrees using
        `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_.

        Parameters
        ----------
        lon1, lat1, lon2, lat2 : float

            Pair of Latitude and Longitude. All args must be of equal length.

        dist : str, `optional`

            Output distance in miles ('mi') or kilometers ('km')
            (the default is mi)

        Returns
        -------
        numpy.ndarray

            Euclidean distance between two points in miles.

        """
        if dist == "km":  # pragma: no cover
            R = 6372.8
        else:
            R = 3959.87433
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (np.sin(dlat / 2.0) ** 2
             + np.cos(lat1) * np.cos(lat2)
             * np.sin(dlon / 2.0) ** 2)
        op = R * 2 * np.arcsin(np.sqrt(a))
        return op

    def _opt_ip(self) -> Tuple[Dict[str, int],
                               Dict[str, int],
                               Dict[Tuple[str, str], float]]:
        """
        Create inputs dicts for :func:`transport_prob`.

        Returns
        -------
        tuple
            A tuple containing the following::

                demand : dict
                        Dict containing location, demand as key value pair.

                supply : dict
                        Dict containing location, supply as key value pair.

                costs : dict
                        Dict containing (suppy_node, demand_node), costs
                        as key value pair.

        """
        df = pd.DataFrame(zip(self._loc, self._demand, self._supply,
                              self._lat, self._lon, [1] * len(self._loc)),
                          columns=["loc", "demand", "supply",
                                   "lat", "lon", "key"])
        df["loc"] = df["loc"].astype(str)
        df_demand = df[df["demand"] > 0]
        df_supply = df[df["supply"] > 0]
        # Balance demand and supply
        tot_demand = sum(df_demand["demand"])
        tot_supply = sum(df_supply["supply"])
        if tot_demand > tot_supply:
            df_supply = df_supply.append({'loc': "Dummy",
                                          "demand": 0,
                                          "supply": tot_demand - tot_supply,
                                          "lat": 0,
                                          "lon": 0,
                                          "key": 1},
                                         ignore_index=True)
        elif tot_supply > tot_demand:
            df_demand = df_demand.append({"loc": "Dummy",
                                          "demand": tot_supply - tot_demand,
                                          "supply": 0,
                                          "lat": 0,
                                          "lon": 0,
                                          "key": 1},
                                         ignore_index=True)
        df_wt = pd.merge(df_supply,
                         df_demand,
                         on="key",
                         how="outer",
                         copy=False)
        df_wt = df_wt.drop(labels=["key"], axis=1)
        df_wt["dist"] = Transport.haversine_np(df_wt["lon_x"],
                                               df_wt["lat_x"],
                                               df_wt["lon_y"],
                                               df_wt["lat_y"])
        df_wt = df_wt.sort_values(["loc_x", "loc_y"])
        df_wt.loc[(df_wt["loc_x"] == "Dummy") | (df_wt["loc_y"] == "Dummy"),
                  "dist"] = 0
        df_wt["cost"] = df_wt["dist"] * self._cost
        ip_supply = dict(zip(df_supply["loc"].astype(str),
                             df_supply["supply"]))
        ip_demand = dict(zip(df_demand["loc"].astype(str),
                             df_demand["demand"]))
        df_wt = df_wt[["loc_x", "loc_y", "cost"]]
        ip_costs = df_wt.set_index(["loc_x", "loc_y"]).cost.to_dict()
        return (ip_demand, ip_supply, ip_costs)

    @staticmethod
    def integer_program(demand: Dict[str, int],
                        supply: Dict[str, int],
                        costs: Dict[Tuple[str, str], float],
                        debug: bool = False
                        ) -> Tuple[str,
                                   float,
                                   List[Tuple[str,
                                              str,
                                              int]]]:
        """
        Transportation Problem using Integer programming.

        Parameters
        ----------
        demand : dict

            Dict containing location, demand as key value pair.

        supply : dict

            Dict containing location, supply as key value pair.

        costs : dict

                Dict containing (suppy_node, demand_node), costs
                as key value pair.

        debug : bool, optional

            Print log in console (the default is False)

        Returns
        -------
        list
            Containing the following::

              Optimization solution status : str
              Objective function value in real number : float
              Optimal path with costs : list of tuples
                  (supply_node, demand_node, units)

        Example
        -------
        >>> optimal_op = Transport.integer_program(demand_dict,
                                                   supply_dict,
                                                   costs_list,
                                                   1)

        """
        supply_node = supply.keys()
        demand_node = demand.keys()
        # Integer programming
        model = pulp.LpProblem("Transportation_problem", pulp.LpMinimize)
        Routes = [(s, d) for s in supply_node for d in demand_node]
        route_vars = pulp.LpVariable.dicts("Route",
                                           (supply_node, demand_node),
                                           lowBound=0,
                                           upBound=None,
                                           cat=pulp.LpInteger)
        # Objective function
        model += pulp.lpSum([route_vars[s][d] * costs[(s, d)]
                             for (s, d) in Routes])
        # Con 01: Supply maximum constraints are added to model for each
        #         supply node
        for s in supply_node:
            model += pulp.lpSum([route_vars[s][d]
                                 for d in demand_node]) <= supply[s]
        # Con 02: The demand minimum constraints are added to model for each
        #         demand node
        for d in demand_node:
            model += pulp.lpSum([route_vars[s][d]
                                 for s in supply_node]) >= demand[d]
        # Solve
        debug = int(debug is True)
        pulp.LpSolverDefault.msg = debug
        model.solve()
        route_list = [(v.name.split("_")[1], v.name.split("_")[2], v.varValue)
                      for v in model.variables() if v.varValue > 0]
        ip_op = (pulp.LpStatus[model.status],
                 model.objective.value(),
                 route_list)
        return ip_op

    def solve(self, debug: bool = False) -> List[Tuple[str, str, int]]:
        """
        Transportation Problem using Integer programming.

        Parameters
        ----------
        :debug: int (binary), optional

            Print log in console when 1, 0 otherwise (the default is 0)

        Returns
        -------
        list of tuples
            Containing the following::

              Optimal path with costs : (supply_node, demand_node, units)

        """
        self._inputs = self._opt_ip()
        self.output = Transport.integer_program(demand=self._inputs[0],
                                                supply=self._inputs[1],
                                                costs=self._inputs[2],
                                                debug=debug)
        return self.output[2]

    def summary(self):  # pragma: no cover
        """Print summary for optimization results."""
        print("Summary:\n")
        print("Total demand:", sum(self._ori_demand))
        print("Total supply:", sum(self._ori_supply))
        if sum(self._ori_demand) > sum(self._ori_supply):
            print("Deficit:", sum(self._ori_demand) - sum(self._ori_supply))
        elif sum(self._ori_supply) > sum(self._ori_demand):
            print("Surplus:", sum(self._ori_supply) - sum(self._ori_demand))
        print("Optimization status:", self.output[0])
        if self.output[0] == "Optimal":
            print("Total cost:", np.round(self.output[1], decimals=2))
