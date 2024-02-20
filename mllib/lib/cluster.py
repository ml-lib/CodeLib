"""
Clustering module.

Objective:
    - Determine optimal number of clusters using
      `Gap statistic <https://web.stanford.edu/~hastie/Papers/gap.pdf>`_.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 05, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,R0914

from typing import List

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


class Cluster():
    """
    Clustering module.

    Objective:
        - Determine optimal number of clusters using
          `Gap statistic <https://web.stanford.edu/~hastie/Papers/gap.pdf>`_.

    Parameters
    ----------
    df : pd.DataFrame

        Dataframe containing all clustering variables i.e. `x_var`.

    x_var : List[str]

        List of clustering variables.

    max_cluster : int, optional

        Maximum number of clusters (the default is 20).

    nrefs : int, optional

        Number of random references to be created (the default is 20).

    seed : int, optional

        Random seed (the default is 1).

    method : str, optional

        Stopping criterion: `one_se` or `gap_max` (the default is "one_se").

    Returns
    -------
    optimal_k : int

        Optimal number of clusters.

    df_summary : pandas.DataFrame

        DataFrame containing Gap statistic and standard error for all clusters.

    Methods
    -------
    opt_k

    Example
    -------
    >>> clus_sol = Cluster(df=df_ip, x_var=["x1"], max_cluster=6, nrefs=5)
    >>> clus_op = clus_sol.opt_k()

    Notes
    -----
    Points to be noted for `method`:

    - Default method is `one_se`.

    - In case the clustering variables `x_var` contains `any` categorical or
      boolean values, the default method switches to `gap_max`

    Points to be noted for `max_cluster`:

    - Maximum number of clusters are defined as

        min(`max_cluster`, number of unique records)

    - In case of categorical variables, when `max_cluster` is lesser than the
      number of unique records, the final solution may be a single cluster.
      Since the module uses Gap statistic, this phenomenon is expected.
      Hence, the end user must research alternate ways to determine optimal
      number of clusters.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 x_var: List[str],
                 max_cluster: int = 20,
                 nrefs: int = 20,
                 seed: int = 1,
                 method: str = "one_se"):
        """Initialize variables for module ``Cluster``."""
        self.df = df
        self.x_var = x_var
        self.max_cluster = max_cluster
        self.nrefs = nrefs
        self.seed = seed
        self.clus_op: pd.DataFrame = None
        self.optimal_k: int = None
        self.df_summary: pd.DataFrame = None
        self.df = self.df[self.x_var]
        self.method = method
        self._pre_processing()

    def _pre_processing(self):
        self.max_cluster = min(self.max_cluster,
                               len(self.df.drop_duplicates()))
        x_cat = self.df.select_dtypes(include=['object', 'bool'])
        if not x_cat.empty:
            self.method = "gap_max"

    def _nref(self):
        """Create random reference data."""
        df = self.df
        x_cat = df.select_dtypes(include=['object', 'bool'])
        x_num = df.select_dtypes(include=['int', 'float64'])
        if not x_cat.empty:
            for _, cat_col in enumerate(x_cat.columns):
                cat_val_list = df[cat_col].unique()
                uniqu_val = len(cat_val_list)
                temp_cnt = 0
                while temp_cnt != uniqu_val:
                    temp_d = np.random.choice(cat_val_list,
                                              size=len(df),
                                              p=[1.0/uniqu_val] * uniqu_val)
                    temp_cnt = len(set(temp_d))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [cat_col]
                if _ == 0:
                    x_cat_d = temp_d
                else:
                    x_cat_d = x_cat_d.join(temp_d)
            df_sample = x_cat_d
        if not x_num.empty:
            for _, num_col in enumerate(x_num.columns):
                temp_d = np.random.uniform(low=min(df[num_col]),
                                           high=max(df[num_col]),
                                           size=len(df))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [num_col]
                if _ == 0:
                    x_cont_d = temp_d
                else:
                    x_cont_d = x_cont_d.join(temp_d)
            if not x_cat.empty:
                df_sample = df_sample.join(x_cont_d)
            else:
                df_sample = x_cont_d
        df_sample = pd.get_dummies(data=df_sample, drop_first=True)
        df_sample = pd.DataFrame(scale(df_sample))
        return df_sample

    def _gap_statistic(self):
        """Compute optimal number of clusters using gap statistic."""
        df = self.df
        # One hot encoding of categorical variables
        df_clus = pd.get_dummies(data=df, drop_first=True)
        # Scale the data
        df_clus_ip = pd.DataFrame(scale(df_clus))
        # Create arrays for gap and sk
        gaps = np.zeros(self.max_cluster)
        sks = np.zeros(self.max_cluster)
        # Create results dataframe
        df_summary = pd.DataFrame({"cluster": [], "gap": [], "sk": []})
        # Create new random reference set
        dict_nref = {}
        for nref_index in range(self.nrefs):
            dict_nref[nref_index] = self._nref()
        # Compute gap statistic
        for gap_index, k in enumerate(range(1, self.max_cluster + 1)):
            # Holder for reference dispersion results
            ref_disps = np.zeros(self.nrefs)
            # For n references, generate random sample and perform kmeans
            # getting resulting dispersion of each loop
            for nref_index in range(self.nrefs):
                # Create new random reference set
                random_ref = dict_nref[nref_index]
                # Fit to it
                kmeans_ref = KMeans(k, random_state=self.seed, n_init=10)
                kmeans_ref.fit(random_ref)
                ref_disp = kmeans_ref.inertia_
                ref_disps[nref_index] = ref_disp
            # Fit cluster to original data and create dispersion
            kmeans_orig = KMeans(k, random_state=self.seed, n_init=10)
            kmeans_orig.fit(df_clus_ip)
            orig_disp = kmeans_orig.inertia_
            # Calculate gap statistic
            gap = np.inf
            if orig_disp > 0.0:
                gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
            # Compute standard error
            sk = 0.0
            if sum(ref_disps) != 0.0:
                sdk = np.std(np.log(ref_disps))
                sk = sdk * np.sqrt(1.0 + 1.0 / self.nrefs)
            # Assign this loop's gap statistic and sk to gaps and sks
            gaps[gap_index] = gap
            sks[gap_index] = sk
            df_summary = pd.concat([df_summary,
                                    pd.DataFrame({"cluster": [k],
                                                  "gap": [gap],
                                                  "sk": [sk]})],
                                   ignore_index=True)
            # Stopping criteria
            if self.method == "one_se":
                if k > 1 and gaps[gap_index-1] >= gap - sk:
                    opt_k = k-1
                    km = KMeans(opt_k, random_state=self.seed, n_init=10)
                    km.fit(df_clus_ip)
                    clus_op = km.labels_
                    break
            opt_k = np.argmax(gaps) + 1
            km = KMeans(opt_k, random_state=self.seed, n_init=10)
            km.fit(df_clus_ip)
            clus_op = km.labels_
        self.df_summary = df_summary
        self.optimal_k = opt_k
        self.clus_op = pd.concat([self.df,
                                  pd.DataFrame(data=clus_op,
                                               columns=["cluster"])],
                                 axis=1)

    def opt_k(self) -> pd.DataFrame:
        """Compute optimal number of clusters using gap statistic.

        Returns
        -------
        pd.DataFrame

            pandas dataframe containing::

                x_var
                cluster

        """
        self._gap_statistic()
        return self.clus_op
