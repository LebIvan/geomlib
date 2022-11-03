import pandas as pd
import numpy as np
import typing

from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram, plot_point_cloud
from gtda.diagrams import PersistenceEntropy
from gtda.time_series import Stationarizer

import gudhi as gd
import gudhi.representations

import plotly.express as px


class TDA():
    # This class uses topological data analysis to count features for time series, graphs

    # ADD PERSISTENCE ENTROPY FEATURES

    def make_cloud_from_ts(self, data: pd.Series, window: int, points_to_point: int, k: int) -> np.array:
        """
        :param data: time series with observations
        :param window: size of window for points cloud
        :param points_to_point: dimension of vector multiplied by dimension of random variable
        :param k: start point for window
        :return: np.array with points of cloud
        """

        cloud = []
        for i in range(window):
            point = [data[k - i - j] for j in range(points_to_point)]
            cloud.append(point)

        return np.array(cloud)

    # ------------------------------------------------------------------------------

    def make_cloud_from_df(self, data: pd.DataFrame, window: int, k: int) -> np.array:
        """
        :param data: DataFrame which rows are observations
        :param window: size of window for points cloud
        :param points_to_point: dimension of vector multiplied by dimension of random variable
        :param k: start point for window
        :return: np.array with points of cloud
        """

        cloud = []
        for i in range(window):
            point = list(data.iloc[k - i])
            cloud.append(point)

        return np.array(cloud)

    # ------------------------------------------------------------------------------

    def point_clouds(self, data: typing.Union[pd.DataFrame, pd.Series], points_to_point: int,
                     window: int) -> np.array:
        """
        :param data: time series or DataFrame which rows are observations
        :param window: size of window for points cloud
        :param points_to_point: dimension of vector multiplied by dimension of random variable
        :return: np.array with points of cloud
        """

        clouds = []

        if (type(data) == pd.Series):
            diff = window + points_to_point
        elif (type(data) == pd.DataFrame):
            diff = window

        else:
            return

        for i in range(diff, len(data)):
            if (type(data) == pd.Series):
                cloud = self.make_cloud_from_ts(data, window, points_to_point, i)
            if (type(data) == pd.DataFrame):
                cloud = self.make_cloud_from_df(data, window, i)
            clouds.append(cloud)

        return clouds

    # ------------------------------------------------------------------------------

    def add_pers_entropy_ts(self, data: typing.Union[pd.DataFrame, pd.Series], points_to_point: int = 4,
                            window: int = 30, n_homologies: int = 1) -> pd.DataFrame:
        # counts persistence entropy for every window and concat it to time series
        """
        :param df: initial time series or DataFrame which rows are observations
        :param points_to_point: dimension of vector multiplied by dimension of random variable
        :param window: size of window for points cloud
        :param n_homologies: max i for H_{i} (maximum dimension j for j-d homologies)
        :return: pd.DataFrame with new features
        """

        # make clouds and its persistence diagrams:
        clouds = self.point_clouds(data, points_to_point, window)
        VR = VietorisRipsPersistence(homology_dimensions=list(range(n_homologies + 1)))
        diagrams = VR.fit_transform(clouds)

        # count persistence entropy:
        PE = PersistenceEntropy()
        features = PE.fit_transform(diagrams)

        # print(features)

        # concat new features:
        if (type(data) == pd.Series):
            diff = window + points_to_point
            data = data.to_frame()
        elif (type(data) == pd.DataFrame):
            diff = window

        # result = pd.DataFrame(series)
        for i in range(n_homologies + 1):
            f_i = [np.nan] * diff + [features[j][i] for j in range(len(features))]
            data[f'H{i}'] = f_i
            # result = pd.concat(result, f_i)

        return data.dropna()

    # ==============================================================================

    def pct_change(self, data: pd.Series) -> pd.Series:
        # turns time series of observations to time series of changes:
        # x_t -> (x_t-x_{t-1})/x_t
        """
        :param data: time series of observations
        :return: time series of changes
        """
        return res.pct_change()

    # ==============================================================================

    # def PD_pairwise_distances_features():

    # ==============================================================================

    def persistence_landscape(self, cloud: np.array, ):
        #
        """
        :param cloud:
        :param :
        :return:
        """
        acX = gd.AlphaComplex(points=cloud).create_simplex_tree()
        dgmX = acX.persistence()

        LS = gd.representations.Landscape(num_landscapes=3, resolution=1000)
        L = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])

    # def persistence_landscapes_features():

    # ==============================================================================

    # def persistence_images_features():
