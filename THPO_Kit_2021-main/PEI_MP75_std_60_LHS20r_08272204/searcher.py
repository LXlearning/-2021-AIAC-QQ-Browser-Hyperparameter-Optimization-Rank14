# coding=utf-8
import copy
import random
import warnings

import numpy as np

from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm

from lhs import lhs
# from sko.DE import DE

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


class UtilityFunction(object):
    """
    This class mainly implements the collection function
    """

    def __init__(self, kind, kappa, x_i):
        self.kappa = kappa
        self.x_i = x_i

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi', 'PseudoEI', 'mp', 'std']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x_x, g_p, y_max, point_add):
        if self.kind == 'ucb':
            return self._ucb(x_x, g_p, self.kappa)
        if self.kind == 'ei':
            return self._ei(x_x, g_p, y_max, self.x_i)
        if self.kind == 'poi':
            return self._poi(x_x, g_p, y_max, self.x_i)
        if self.kind == 'PseudoEI':
            return self._Pseudoei(x_x, g_p, y_max, point_add)
        if self.kind == 'mp':
            return self._mp(x_x, g_p)
        if self.kind == 'std':
            return self._std(x_x, g_p)

    @staticmethod
    def _std(x_x, g_p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)
        return std
    @staticmethod
    def _mp(x_x, g_p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        return mean

    @staticmethod
    def _Pseudoei(x_x, g_p, y_max, point_add):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        a_a = (mean - y_max)
        z_z = a_a / std
        EI = a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

        def corr_gauss(d, theta=1):
            td = theta * np.power(d, 2)
            return np.exp(-np.sum(td, axis=1))

        if len(point_add) > 0:
            correlation = np.zeros((x_x.shape[0], len(point_add)))
            # 计算影响函数
            for i in range(len(point_add)):
                dx = x_x - point_add[i]
                correlation[:, i] = corr_gauss(dx)

            EI = EI * np.prod(1 - correlation, axis=1)

        return EI

    @staticmethod
    def _ucb(x_x, g_p, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        return a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

    @staticmethod
    def _poi(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        z_z = (mean - y_max - x_i) / std
        return norm.cdf(z_z)


class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        # def optimizer(obj_func, initial_theta, bounds):
        #     return theta_opt, func_min

        kernel = C(1.0, (1e-3, 1e3)) * RBF(2, (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=np.random.RandomState(1),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=3,
        )
        self.gp = gp


    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def train_gp(self, x_datas, y_datas):
        """ train gp

        Args:
            x_datas: Parameters
            y_datas: Reward of Parameters

        Return:
            gp: Gaussian process regression
        """
        self.gp.fit(x_datas, y_datas)

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample

    def get_bounds(self):
        """ Get sorted parameter space

        Return:
            _bounds: The sorted parameter space
        """

        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds

    def acq_max(self, f_acq, gp, y_max, bounds, num_warmup, num_starting_points, point_add):
        """ Produces the best suggested parameters

        Args:
            f_acq: Acquisition function
            gp: GaussianProcessRegressor
            y_max: Best reward in suggestions history
            bounds: The parameter boundary of the acquisition function
            num_warmup: The number of samples randomly generated for the collection function
            num_starting_points: The number of random samples generated for scipy.minimize

        Return:
            Return the current optimal parameters
        """
        # 1=======================Warm up with random points
        x_tries = np.array([self.random_sample() for _ in range(int(num_warmup))])
        # 删除已有样本点
        x_tries = np.array(
            list(set(tuple(x) for x in x_tries.tolist()).difference(set(tuple(x) for x in gp.X_train_.tolist()))))
        ys = f_acq(x_tries, g_p=gp, y_max=y_max, point_add=point_add)

        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        # 2==========================================DE算法搜索最优值==========================================

        # de = DE(func=lambda x: -f_acq(x, g_p=gp, y_max=y_max, point_add=point_add),
        #         n_dim=gp.X_train_.shape[1], size_pop=40, max_iter=50,
        #         lb=bounds[0,:], ub=bounds[1,:])
        # x_max, best_y = de.run()

        # ==================================================================================================
        # 3======= Explore the parameter space more throughly(5个随机点作为起始点用L-BFGS-B搜索最大EI位置)
        # x_seeds = np.array([self.random_sample() for _ in range(int(num_starting_points))])
        # for x_try in x_seeds:
        #     # Find the minimum of minus the acquisition function
        #     res = minimize(lambda x: -f_acq(x.reshape(1, -1), g_p=gp, y_max=y_max, point_add=point_add),
        #                    x_try.reshape(1, -1),
        #                    bounds=bounds,
        #                    method="L-BFGS-B")
        #     # See if success
        #     if not res.success:
        #         continue
        #     # Store it if better than previous minimum(maximum).
        #     if max_acq is None or -res.fun[0] >= max_acq:
        #         x_max = res.x
        #         max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])

    def parse_suggestions(self, suggestions):
        """ Parse the parameters result

        Args:
            suggestions: Parameters

        Return:
            suggestions: The parsed parameters
        """

        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestions = [{p_names[index]: suggestion[index] for index in range(len(suggestion))}
                       for suggestion in suggestions]

        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions

    def init_param_group(self, init_suggestions=5):
        """ Suggest n_suggestions parameters in random form

        Args:
            init_suggestions: number of parameters to suggest in every iteration

        Return:
            next_suggestions: n_suggestions Parameters in random form
        """
        # 拉丁方采样替代随机生成样本点
        _bounds = self.get_bounds()
        x = lhs(_bounds.shape[0], samples=init_suggestions, criterion='cm')
        # 修正样本点
        lhs_value = (_bounds[:, 1] - _bounds[:, 0]) * x + _bounds[:, 0]

        return lhs_value

    def init_param_group_random(self, n_suggestions):
        next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
                             for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

        return next_suggestions

    def suggest(self, suggestions_history, n_suggestions=1):

        num_point = len(suggestions_history)
        # if (suggestions_history is None) or (num_point <= 0):
        #     next_suggestions = self.init_param_group_random(n_suggestions)
        # =================================LHS采样方法=================================
        lhs_sample = 20
        if (suggestions_history is None) or (len(suggestions_history) == 0 ):
            lhs_value = self.init_param_group(init_suggestions=lhs_sample)
            np.savetxt('lhs_point.txt',lhs_value)
            lhs_suggestions = self.parse_suggestions(lhs_value)
            
            next_suggestions = []
            for index in range(n_suggestions):
                suggestion = lhs_suggestions[len(suggestions_history) + index]
                next_suggestions.append(suggestion)            
        elif 0 < len(suggestions_history) < lhs_sample:
            lhs_value = np.loadtxt('lhs_point.txt')
            lhs_suggestions = self.parse_suggestions(lhs_value)
            
            next_suggestions = []
            for index in range(n_suggestions):
                suggestion = lhs_suggestions[len(suggestions_history) + index]
                next_suggestions.append(suggestion)
        # ===========================================================================
        else:
            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            self.train_gp(x_datas, y_datas)
            _bounds = self.get_bounds()
            suggestions = []
            for index in range(n_suggestions):
                ei_num = 4
                if (num_point <= 60) & (index == ei_num):
                # if index == 4:
                    utility_function = UtilityFunction(kind='std', kappa=(index + 1) * 2.576, x_i=index * 3)
                elif (num_point >= 75) & (index == ei_num):
                # elif (num_point > 60) & (index == ei_num):
                    utility_function = UtilityFunction(kind='mp', kappa=(index + 1) * 2.576, x_i=index * 3)
                else:
                    utility_function = UtilityFunction(kind='PseudoEI', kappa=(index + 1) * 2.576, x_i=index * 3)
                suggestion = self.acq_max(
                    f_acq=utility_function.utility,
                    gp=self.gp,
                    y_max=y_datas.max(),
                    bounds=_bounds,
                    num_warmup=min(1000 * np.power(2, x_datas.shape[1]), 30000),
                    num_starting_points=5,
                    point_add=suggestions
                )
                suggestions.append(suggestion)

            suggestions = np.array(suggestions)
            # ===============================================================================================
            # import matplotlib.pyplot as plt
            # plt.ion()  # 打开交互模式
            # plt.clf()  # 清除当前的Figure对象
            # plt.xlim([-0.5, 5.5])
            # plt.ylim([-0.5, 5.5])
            # plt.scatter(x_datas[:, 0], x_datas[:, 1], color='r', s=30)
            # if num_point > 60:
            #     plt.scatter(suggestions[:ei_num, 0], suggestions[:ei_num, 1], marker='*', color='b', s=150)
            #     plt.scatter(suggestions[ei_num, 0], suggestions[ei_num, 1], marker='^', color='y', s=150)
            # elif num_point < 50:
            #     plt.scatter(suggestions[:ei_num, 0], suggestions[:ei_num, 1], marker='*', color='b', s=150)
            #     plt.scatter(suggestions[ei_num, 0], suggestions[ei_num, 1], marker='<', color='y', s=150)
            # else:
            #     plt.scatter(suggestions[:, 0], suggestions[:, 1], marker='*', color='b', s=150)
            # plt.title('max:' + str(round(max(y_datas), 1)) + '===' + str(num_point), fontsize=24)
            # for i in np.arange(len(y_datas)):
            #     plt.text(x_datas[i, 0], x_datas[i, 1], str(round(y_datas[i], 1)), fontsize=8)
            # plt.pause(3)  # 暂停功能
            # plt.ioff()
            # import sys
            # if max(y_datas) > -0.3:
            #     print('收敛到最大值', max(y_datas))
            #     sys.exit(0)
            # import time
            # if num_point == 95:
            #     plt.savefig('./fig/max' + str(round(max(y_datas), 1)) + '_' + str(time.localtime().tm_hour)
            #                 + str(time.localtime().tm_min) + str(time.localtime().tm_sec) + '.png')
            # ===============================================================================================
            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions

        return next_suggestions
