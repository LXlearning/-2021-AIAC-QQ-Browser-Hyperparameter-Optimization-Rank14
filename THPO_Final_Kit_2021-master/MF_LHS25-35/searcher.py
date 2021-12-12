# coding=utf-8
import warnings
warnings.filterwarnings("ignore")
import copy
import random
import numpy as np

from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm
from lhs import lhs
from sklearn.model_selection import KFold

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
            mean, std = g_p.predict_ensemble(x_x)
        return std

    @staticmethod
    def _mp(x_x, g_p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict_ensemble(x_x)

        return mean

    @staticmethod
    def _Pseudoei(x_x, g_p, y_max, point_add):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict_ensemble(x_x)

        a_a = (mean - y_max)
        z_z = a_a / std
        EI = a_a * norm.cdf(z_z) + std * norm.pdf(z_z)
        # VIX = std*std*((z_z*z_z+1)*norm.cdf(z_z) + z_z*norm.pdf(z_z)) - EI*EI
        # EI = EI / np.sqrt(VIX)
        
        def corr_gauss(d, theta=0.5):
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
            mean, std = g_p.predict_ensemble(x_x)

        return mean + kappa * std

    @staticmethod
    def _ei(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict_ensemble(x_x)

        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        return a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

    @staticmethod
    def _poi(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict_ensemble(x_x)

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

        self.surrogate_weight = dict()
        self.surrogate_container = dict()
        # self.surrogate_r = list()
        self.power_num = 3

    def train_gp(self, x_datas, y_datas, r):
        """ train gp

        Args:
            x_datas: Parameters
            y_datas: Reward of Parameters

        Return:
            gp: Gaussian process regression
        """

        self.surrogate_container[r] = GaussianProcessRegressor(
            kernel=1.0 * RBF(length_scale=1, length_scale_bounds=(0.5, 1e3)),
            alpha=1e-6,
            random_state=np.random.RandomState(1),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=5, )
        self.surrogate_container[r].fit(x_datas, y_datas)

    def predict_ensemble(self, X):
        means, vars, vars1 = np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0])
        for i, r in enumerate(self.surrogate_r):
            mean, std = self.surrogate_container[r].predict(X, return_std=True)
            # vars += self.surrogate_weight[i] * self.surrogate_weight[i] * std * std
            vars1 += self.surrogate_weight[i] / (std * std)
        stds = np.sqrt(1/vars1)
        for i, r in enumerate(self.surrogate_r):
            mean, std = self.surrogate_container[r].predict(X, return_std=True)
            means += self.surrogate_weight[i] * mean * np.power(stds/std, 2)

        return means, stds

    def update_weight(self):
        K = 3
        preserving_order_p = list()
        preserving_order_nums = list()
        if len(self.Y) > 3:
            for i, r in enumerate(self.surrogate_r):
                if r != K - 1:
                    mean = self.surrogate_container[r].predict(self.X)
                    preorder_num, pair_num = self.calculate_preserving_order_num(mean, self.Y)
                    preserving_order_p.append(preorder_num / pair_num)
                    preserving_order_nums.append(preorder_num)
                else:
                    kfold = KFold(n_splits=5)
                    cv_pred = np.array([0] * len(self.Y))
                    for train_idx, valid_idx in kfold.split(self.X):
                        train_configs, train_y = self.X[train_idx], self.Y[train_idx]
                        valid_configs, valid_y = self.X[valid_idx], self.Y[valid_idx]
                        _surrogate = GaussianProcessRegressor(
                            kernel=1.0 * RBF(length_scale=1, length_scale_bounds=(0.5, 1e3)),
                            alpha=1e-6,
                            random_state=np.random.RandomState(1),
                            optimizer='fmin_l_bfgs_b',
                            n_restarts_optimizer=5,)
                        _surrogate.fit(train_configs, train_y)
                        pred = _surrogate.predict(valid_configs)
                        cv_pred[valid_idx] = pred
                    preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, self.Y)
                    preserving_order_p.append(preorder_num / pair_num)
                    preserving_order_nums.append(preorder_num)

            trans_order_weight = np.array(preserving_order_p)
            power_sum = np.sum(np.power(trans_order_weight, self.power_num))
            new_weights = np.power(trans_order_weight, self.power_num) / power_sum
            self.surrogate_weight = new_weights

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def init_param_group(self, init_suggestions=5):

        # 拉丁方采样替代随机生成样本点
        _bounds = self.get_bounds()
        x = lhs(_bounds.shape[0], samples=init_suggestions, criterion='cm')
        # 修正样本点
        lhs_value = (_bounds[:, 1] - _bounds[:, 0]) * x + _bounds[:, 0]

        return lhs_value

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

    def acq_max(self, f_acq, Xdata, y_max, bounds, num_warmup, num_starting_points, point_add):
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
            list(set(tuple(x) for x in x_tries.tolist()).difference(set(tuple(x) for x in Xdata.tolist()))))
        ys = f_acq(x_tries, g_p=self, y_max=y_max, point_add=point_add)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
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

    def suggest_old(self, suggestions_history, suggestions_history_dict, iteration_number, run_history,  n_suggestions=1):
        """ Suggest next n_suggestion parameters, old implementation of preliminary competition.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}

                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        num_point = len(suggestions_history)
        # =================================LHS采样方法=================================
        if len(self.parameters_config) >= 5:
            self.lhs_sample = 35
        elif 3 <= len(self.parameters_config) < 5:
            self.lhs_sample = 30
        else:
            self.lhs_sample = 25
        next_suggestions = []
        if (suggestions_history is None) or (len(suggestions_history) == 0):
            lhs_value = self.init_param_group(init_suggestions=self.lhs_sample)
            np.savetxt('lhs_pointAAA.txt', lhs_value)
            lhs_suggestions = self.parse_suggestions(lhs_value)
            for index in range(n_suggestions):
                suggestion = lhs_suggestions[len(suggestions_history) + index]
                next_suggestions.append(suggestion)
        elif 0 < len(suggestions_history) < self.lhs_sample:
            lhs_value = np.loadtxt('lhs_pointAAA.txt')
            lhs_suggestions = self.parse_suggestions(lhs_value)
            for index in range(n_suggestions):
                if len(suggestions_history) + len(run_history) + index <= self.lhs_sample - 1:
                    suggestion = lhs_suggestions[len(suggestions_history) + len(run_history) + index]
                    next_suggestions.append(suggestion)
                else:
                    # print(111)
                    continue
        # ===========================================================================
        if len(next_suggestions) < n_suggestions:
            suggestions = [[item[1] for item in sorted(suggestion.items(), key=lambda x: x[0])]
                           for suggestion in next_suggestions]
            # ===============================集成模型=======================================
            self.surrogate_r = []
            for r, suggestions_history_list in suggestions_history_dict.items():
                if len(suggestions_history_list) > 3:
                    self.surrogate_r.append(r)
                    x_datas, y_datas = self.parse_suggestions_history(suggestions_history_list)
                    self.X, self.Y = self.parse_suggestions_history(suggestions_history_list)
                    self.train_gp(x_datas, y_datas, r)
            self.update_weight()
            # ===========================================================================
            Xdata_all, Ydata_all = self.parse_suggestions_history(suggestions_history)
            pointadd, ytemp = self.parse_suggestions_history(run_history)
            pointadd = pointadd.tolist()
            _bounds = self.get_bounds()
            for index in range(n_suggestions-len(suggestions)):
                ei_num = 0
                if (num_point <= 50) & (index == 0) & (iteration_number % 2 == 0):
                    utility_function = UtilityFunction(kind='std', kappa=(index + 1) * 2.576, x_i=index * 3)
                elif (num_point >= 50) & (index == 0) & (iteration_number % 2 == 0):
                    utility_function = UtilityFunction(kind='mp', kappa=(index + 1) * 2.576, x_i=index * 3)
                # elif (num_point >= 65) & (index == 0):
                #     utility_function = UtilityFunction(kind='mp', kappa=(index + 1) * 2.576, x_i=index * 3)
                else:
                    utility_function = UtilityFunction(kind='PseudoEI', kappa=(index + 1) * 2.576, x_i=index * 3)
                suggestion = self.acq_max(
                    f_acq=utility_function.utility,
                    Xdata=Xdata_all,
                    y_max=y_datas.max(),
                    bounds=_bounds,
                    num_warmup=min(2000 * np.power(2, x_datas.shape[1]), 30000),
                    num_starting_points=5,
                    point_add=pointadd
                )
                suggestions.append(suggestion)
                pointadd.append(suggestion)

            suggestions = np.array(suggestions)
            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions
        return next_suggestions

    def get_my_score(self, suggestion):
        """ Get the most trusted reward of all iterations.

        Returns:
            most_trusted_reward: float
        """
        if len(suggestion) == 14 or len(suggestion) < 3:
            score = suggestion['reward'][-1]['value']
        else:
            score1 = suggestion['reward'][-1]['value']
            score2 = suggestion['reward'][-2]['value']
            score3 = suggestion['reward'][-3]['value']
            score = (score1 + score2 + score3) / 3
        return score

    def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters. new implementation of final competition

        Args:
            iteration_number: int ,the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        new_suggestions_history, new_suggestions_history2 = [], []
        suggestions_history_dict = dict()
        for i in range(5):
            suggestions_history_dict[i] = []
        # 历史数据
        for suggestion in suggestion_history:
            iterations_of_suggestion = len(suggestion['reward'])
            cur_score = self.get_my_score(suggestion)
            new_suggestions_history.append([suggestion["parameter"], cur_score])
            # 查看历史得分
            if iterations_of_suggestion <= 6:
                suggestions_history_dict[0].append([suggestion["parameter"], cur_score])
            elif 6 < iterations_of_suggestion <= 12:
                suggestions_history_dict[1].append([suggestion["parameter"], cur_score])
            # elif 5 < iterations_of_suggestion <= 9:
            #     suggestions_history_dict[1].append([suggestion["parameter"], cur_score])
            elif iterations_of_suggestion == 14:
                suggestions_history_dict[2].append([suggestion["parameter"], cur_score])
                new_suggestions_history2.append([suggestion["parameter"], cur_score])
        # run数据
        run_history = []
        for suggestion in running_suggestions:
            iterations_of_suggestion = len(suggestion)
            cur_score = self.get_my_score(suggestion)
            run_history.append([suggestion["parameter"], cur_score])
            if iterations_of_suggestion <= 6:
                suggestions_history_dict[0].append([suggestion["parameter"], cur_score])
            elif 6 < iterations_of_suggestion <= 12:
                suggestions_history_dict[1].append([suggestion["parameter"], cur_score])
            else:
                suggestions_history_dict[2].append([suggestion["parameter"], cur_score])

        # 更新样本点
        next_suggestions = self.suggest_old(new_suggestions_history, suggestions_history_dict, iteration_number, run_history, n_suggestions)
        x_datas, y_datas = self.parse_suggestions_history(new_suggestions_history)
        x_datas_all, y_datas_all = self.parse_suggestions_history(new_suggestions_history2)

        if iteration_number >= 135:
            print('score:', max(y_datas_all), 'point', len(new_suggestions_history))
        # ===============================================================================================
        # ===============================================================================================
        return next_suggestions

    def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
        """ Decide whether to stop the running suggested parameter experiment.

        Args:
            iteration_number: int, the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestions_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

        Returns:
            stop_list: list of bool, indicate whether to stop the running suggestions.
                    len(stop_list) must be the same as len(running_suggestions), for example:
                        len(running_suggestions) = 3, stop_list could be :
                            [True, True, True] , which means to stop all the three running suggestions
        """

        # Early Stop algorithm demo 2:
        res = [False] * len(running_suggestions)
        Lbounds, sug_ranges, cur_scores = [], [], []
        for suggestion in suggestion_history:
            Lbounds.append(suggestion['reward'][-1]['lower_bound'])
            sug_range = suggestion['reward'][-1]['value'] - suggestion['reward'][-1]['lower_bound']
            cur_scores.append(self.get_my_score(suggestion))
            sug_ranges.append(sug_range)
        Lbounds = np.array(Lbounds)
        cur_scores = np.array(cur_scores)
        sug_ranges = np.array(sug_ranges)
        # ========================early_stop策略============================
        if 5 <= len(suggestion_history) < self.lhs_sample:
            for idx, suggestion in enumerate(running_suggestions):
                run_cur_score = self.get_my_score(suggestion)#['reward']
                # if run_cur_score < np.quantile(cur_scores, 0.6):
                #     if len(suggestion['reward']) >= 7:
                #         res[idx] = True
                if np.quantile(cur_scores, 0.6) <= run_cur_score < np.quantile(cur_scores, 0.9):
                    if len(suggestion['reward']) >= 9:
                        res[idx] = True
                elif np.quantile(cur_scores, 0.6) <= run_cur_score < np.quantile(cur_scores, 0.6):
                    if len(suggestion['reward']) >= 6:
                        res[idx] = True
                elif run_cur_score < np.quantile(cur_scores, 0.4):
                    if len(suggestion['reward']) >= 3:
                        res[idx] = True
        elif len(suggestion_history) >= self.lhs_sample:
            # 利用历史数据定义更新策略
            for idx, suggestion in enumerate(running_suggestions):
                run_cur_score = self.get_my_score(suggestion)
                # if run_cur_score >= np.quantile(cur_scores, 0.9):
                #     res[idx] = False
                if np.quantile(cur_scores, 0.6) <= run_cur_score < np.quantile(cur_scores, 0.9):
                    if len(suggestion['reward']) >= 9:
                        res[idx] = True
                elif np.quantile(cur_scores, 0.4) <= run_cur_score < np.quantile(cur_scores, 0.6):
                    if len(suggestion['reward']) >= 6:
                        res[idx] = True
                elif run_cur_score < np.quantile(cur_scores, 0.4):
                    if len(suggestion['reward']) >= 3:
                        res[idx] = True
        return res
