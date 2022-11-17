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

from math import fabs


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
            point = [data.iloc[k - i - j] for j in range(points_to_point)]
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
        if (type(data) == pd.DataFrame):
            diff = window

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
        :param data: initial time series or DataFrame which rows are observations
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

    #     def pct_change(self, data: pd.Series) -> pd.Series:
    #         # turns time series of observations to time series of changes:
    #         # x_t -> (x_t-x_{t-1})/x_t
    #         """
    #         :param data: time series of observations
    #         :return: time series of changes
    #         """
    #         return res.pct_change()

    # ==============================================================================

    # def PD_pairwise_distances_features():

    # ==============================================================================

    #     def persistence_landscape(self, cloud: np.array, ):
    #         #
    #         """
    #         :param cloud:
    #         :param :
    #         :return:
    #         """
    #         acX = gd.AlphaComplex(points=cloud).create_simplex_tree()
    #         dgmX = acX.persistence()

    #         LS = gd.representations.Landscape(num_landscapes=3, resolution=1000)
    #         L = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])

    # ==============================================================================

    def square_integratetion(self, left_bottom_corner: typing.Union[list, tuple], square_side: float,
                             x_step: float, y_step: float, func) -> float:
        # Integrates given function in a square using grid
        """
        :param left_bottom_corner: coordinates of left_bottom_corner
        :param square_side: value of square's size
        :param x_step: integrating step on variable x
        :param y_step: integrating step on variable y
        :param func: function to integrate
        :return: value of integral
        """
        result = 0
        x_dots = np.arange(left_bottom_corner[0], left_bottom_corner[0] + square_side, x_step)
        y_dots = np.arange(left_bottom_corner[1], left_bottom_corner[1] + square_side, y_step)

        for x0 in x_dots:
            for y0 in y_dots:
                result += func(x0 + x_step / 2, y0 + y_step / 2)

        return result * x_step * y_step

    def Gaussian_kernel(self, x, y, sigma, point):
        return 1. / (2 * np.pi * sigma ** 2) * np.exp(-((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * sigma ** 2))

    #     def points_funcs_arr(self, points: typing.Union[list, np.array], sigma: float, weight_func):
    #         # sums points functions multiplied by weight function
    #         """
    #         :param points: set of points
    #         :param sigma: bandwidth parameter for Gaussian kernel
    #         :param weight_func: weight function
    #         :return: sum of points function multiplied by weight function
    #         """

    #         funcs = []
    #         for point in points:
    #             funcs.append(lambda x, y: self.Gaussian_kernel(x, y, sigma, point))
    #         return lambda x, y: sum(f(x, y) for f in funcs) * weight_func(x, y)

    def count_colour(self, cell_x: float, cell_y: float, square_side: float, weight_func, sigma: float,
                     x_step: float, y_step: float, points: typing.Union[np.array, list, tuple]) -> float:
        # counts cells colour with Gaussian kernel
        """

        """
        colour = 0
        for point in points:
            # if(region_includes(point, cell_x, cell_y, square_side)):
            point_f = lambda x, y: weight_func(x, y) * self.Gaussian_kernel(x, y, sigma, point)  # / points.max()
            colour += self.square_integratetion(left_bottom_corner=[cell_x, cell_y], square_side=square_side,
                                                x_step=x_step, y_step=y_step, func=point_f)

        return colour

    def colours_table(self, points: typing.Union[list, np.array], resolution: list, x_step: float, y_step: float,
                      weight_func, sigma: float) -> np.array:
        #
        """

        """

        table = np.zeros(shape=(resolution[0], resolution[1]))
        # points = points/points.max()
        points[:, 0] = points[:, 0] / (points[:, 0].max())
        points[:, 1] = points[:, 1] / (points[:, 1].max())
        square_side = points.max() / resolution[0]
        # print(points)
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                table[resolution[0] - i - 1][j] = self.count_colour(cell_x=i * square_side, cell_y=j * square_side,
                                                                    square_side=square_side,
                                                                    weight_func=weight_func, sigma=sigma, x_step=x_step,
                                                                    y_step=y_step, points=points)
                # print(i*square_side)

        return np.flip(table.T)

    #     def colours_table(self, points: typing.Union[list, np.array], resolution: list, x_step: float, y_step: float, weight_func, sigma: float) -> np.array:
    #         #
    #         """
    #         :param points: list of points in persistance diagram with (x,y) -> (x,y-x)
    #         :param resolution: resolution of persistance image
    #         :param x_step: integrating step for variable x
    #         :param y_step: integrating step for variable y
    #         :param weight_func: weight function
    #         :param sigma: bandwidth parameter for Gaussian kernel
    #         :return: table with colours for persistance image
    #         """

    #         table = np.zeros(shape=(resolution[0], resolution[1]))
    #         # points = points/points.max()
    #         # print(points)

    #         # if(len(points) != 0):
    #         points[:,0] = points[:,0]/(points[:,0].max())
    #         points[:,1] = points[:,1]/(points[:,1].max())
    #         square_side = points.max()/resolution[0]
    #         func = self.points_funcs_arr(points, sigma, weight_func)
    #         for i in range(resolution[0]):
    #             for j in range(resolution[1]):

    #                 table[resolution[0]-i-1][j] = self.square_integratetion(left_bottom_corner=[i*square_side, j*square_side],
    #                                                                         square_side=square_side, x_step=x_step,
    #                                                                         y_step=y_step, func=func)
    #                 # if(len(points) == 0):
    #                     # table[resolution[0]-i-1][j] = 0

    #         return np.flip(table.T)

    #     def drop_bad_pixels(self, features):

    def persistence_image_features(self, data: typing.Union[pd.DataFrame, pd.Series], points_to_point: int = 4,
                                   window: int = 30, n_homologies: int = 1, resolution: int = 10, x_step: float = 0.1,
                                   y_step: float = 0.1, weight_func=lambda x, y: x * np.square(y),
                                   sigma: float = 0.01, drop_const: bool = True) -> pd.DataFrame:
        # Counts persistence landscape image
        """
        :param data: initial time series or DataFrame which rows are observations
        :param points_to_point: dimension of vector multiplied by dimension of random variable
        :param window: size of window for points cloud
        :param n_homologies: max i for H_{i} (maximum dimension j for j-d homologies)
        :param resolution: resolution of persistance image
        :param x_step: integrating step for variable x
        :param y_step: integrating step for variable y
        :param weight_func: weight function
        :param sigma: bandwidth parameter for Gaussian kernel
        :return: pd.DataFrame with new features
        """
        clouds = self.point_clouds(data, points_to_point, window)
        # print(clouds)
        features = []
        for cloud in clouds:
            acX = gd.AlphaComplex(points=cloud).create_simplex_tree()
            dgmX = acX.persistence()

            df = pd.DataFrame(np.array(dgmX, dtype=object), columns=['H_i', 'x,y'])
            df['x'] = df['x,y'].apply(lambda x: x[0])
            df['y'] = df['x,y'].apply(lambda x: x[1])
            df['y-x'] = df['y'] - df['x']
            # print(df)
            # df.drop(columns=['x,y'])
            pointsT = np.array(df[df['H_i'] != 0][['x', 'y-x']])
            # print(pointsT)
            table = self.colours_table(points=pointsT, resolution=[resolution, resolution], x_step=x_step,
                                       y_step=y_step,
                                       weight_func=weight_func, sigma=sigma)
            features.append(table.flatten())

        #         if(drop_const):
        #             features = self.drop_bad_pixels(features)

        # print(len(features))
        if (type(data) == pd.Series):
            diff = window + points_to_point
        elif (type(data) == pd.DataFrame):
            diff = window

        features_df = pd.DataFrame(features)
        # print(features_df)
        extra = pd.DataFrame(np.zeros(shape=(diff, len(features_df.iloc[0]))))
        features_df = pd.concat([extra, features_df], ignore_index=True)
        features_df.index = data.index
        # print(data.to_frame(), features_df)

        if (type(data) == pd.Series):
            data = data.to_frame()

        result = pd.concat([data, features_df], axis=1)[diff:]

        return result

    # ==============================================================================

    # def persistence_images_features():


def test_integration():
    tda = TDA()

    integral = tda.square_integratetion([0, 0], 1, 0.01, 0.01, lambda x, y: x * y)
    if fabs(integral - 0.25) > 1e-8:
        print('Test square_integration failed on function x*y')
        return -1
    integral = tda.square_integratetion([0, 0], 1, 0.01, 0.01, lambda x, y: 0)
    if fabs(integral) > 1e-8:
        print('Test square_integration failed on function 0')
        return -1
    integral = tda.square_integratetion([0, 0], 1, 0.01, 0.01, lambda x, y: 1)
    if fabs(integral - 1) > 1e-8:
        print('Test square_integration failed on function x*y')
        return -1
    integral = tda.square_integratetion([0, 0], 1, 0.01, 0.01, lambda x, y: x + y)
    if (fabs(integral - 1) > 1e-8):
        print('Test square_integration failed on function x+y')
        return -1
    integral = tda.square_integratetion([1, 1], 1, 0.01, 0.01, lambda x, y: x + y)
    if (fabs(integral - 477.237) > 1e-8):
        print('Test square_integration failed on function x+y (1-2)')
        return -1

    return 0


def test_count_colour():
    tda = TDA()

    # Negative colour tests
    points = [[0, 0], [0, 1]]
    colour = tda.count_colour(cell_x=-1, cell_y=-1, square_side=0.5, weight_func=lambda x, y: 1, sigma=0.1, x_step=0.01,
                              y_step=0.01, points=points)
    if fabs(colour < 0):
        print('Test count_colour failed on negative colour test 1')
        return -1

    points = [[1, -1], [-1, 1]]
    colour = tda.count_colour(cell_x=0, cell_y=0, square_side=0.5, weight_func=lambda x, y: 1, sigma=0.1, x_step=0.1,
                              y_step=0.1, points=points)
    if fabs(colour < 0):
        print('Test count_colour failed on negative colour test 2')
        return -1

    points = [[0, 0], [0, 1], [1, 0], [1, 1]]
    colour = tda.count_colour(cell_x=-1, cell_y=-1, square_side=1, weight_func=lambda x, y: 1, sigma=0.1, x_step=0.01,
                              y_step=0.01, points=points)
    if fabs(colour < 0):
        print('Test count_colour failed on negative colour test 3')
        return -1

    points = [[0, 0], [0, 1], [1, 0], [1, 1]]
    colour = tda.count_colour(cell_x=-1, cell_y=-1, square_side=0.5, weight_func=lambda x, y: x*y, sigma=0.1,
                              x_step=0.01, y_step=0.01, points=points)
    if fabs(colour < 0):
        print('Test count_colour failed on negative colour test 4')

    # Other tests
    points = [[0, 0], [0.25, 0.25], [0.5, 0.5], [1, 1]]
    colour = tda.count_colour(cell_x=0, cell_y=0, square_side=1, weight_func=lambda x, y: x*y, sigma=1, x_step=0.01,
                              y_step=0.01, points=points)
    if fabs(colour - 1.55375) > 1e-8:
        print('Test count_colour failed on circle (square) example')
        return -1

    points = [[0, 0]]
    colour = tda.count_colour(cell_x=0, cell_y=0, square_side=1, weight_func=lambda x, y: 1, sigma=1, x_step=0.01,
                              y_step=0.01, points=points)
    if fabs(colour - 0.340488) > 1e-8:
        print('Test count_colour failed on 1 point example')
        return -1

    points = [[0, 0], [1, 1]]
    colour = tda.count_colour(cell_x=0, cell_y=0, square_side=1, weight_func=lambda x, y: 1, sigma=1, x_step=0.01,
                              y_step=0.01, points=points)
    if fabs(colour - 0.780031) > 1e-8:
        print('Test count_colour failed on 2 points example')
        return -1

    return 0

def test_point_clouds():
    tda = TDA()

    data = pd.Series([0, 1, 2, 3])
    clouds = tda.point_clouds(data=data, points_to_point=1, window=1)
    if clouds == -1:
        print('Test point_clouds failed on pandas.Series')
        return -1

    data = pd.DataFrame([0, 1, 2], [3, 4, 5], [6, 7, 8])
    clouds = tda.point_clouds(data=data, points_to_point=1, window=1)
    if clouds == -1:
        print('Test point_clouds failed on pandas.DataFrame')
        return -1

    data = pd.Series([0, 1, 2, 3])
    clouds = tda.point_clouds(data=data, points_to_point=1, window=1)
    if clouds != [data]:
        print('Test point_clouds failed on ptp1 w1 test')
        return -1

    data = pd.Series([0, 1, 2, 3])
    clouds = tda.point_clouds(data=data, points_to_point=3, window=5)
    if clouds != -1:
        print('Test point_clouds failed on big parameters')
        return -1

    data = pd.Series([0, 1, 2, 3])
    clouds = tda.point_clouds(data=data, points_to_point=2, window=2)
    if clouds != [[[0, 1], [1, 2]], [[1, 2], [2, 3]]]:
        print('Test point_clouds failed on 0123 2,2')
        return -1

    return 0


def test_colours_table():
    tda = TDA()
    points = [[0,0], [0,1]]
    table = tda.colours_table(points, resolution=[10, 10], x_step=0.1, y_step=0.1, weight_func=lambda x, y: 1, sigma=1)
    if table != -1:
        print('Test colours_table failed on no homologies bigger than 0')
        return -1
    return 0


def run_tests():
    ti = test_integration()
    tcc = test_count_colour()
    tpc = test_point_clouds()
    tct = test_colours_table()
    if ti == -1:
        print('test_integration failed. The error is described above')
        return -1
    if tcc == -1:
        print('test_count_colour failed. The error is described above')
        return -1
    if tpc == -1:
        print('test_point_clouds failed. The error is described above')
        return -1
    if tct == -1:
        print('test_point_table failed. The error is described above')
        return -1


