from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as st
from autorank import autorank, plot_stats
import scikit_posthocs as sp

# define Python user-defined exceptions
class MaxRuntimeHit(Exception):
    "Raised when the input value is less than 18"
    pass

class QAP_Instance:

    sqrt_2 = np.sqrt(2)

    def __init__(self, N, max_runtime):
        self.N = N

        #Matriz de distancias
        self.D = np.random.randint(1, 50, size=(N, N))
        np.fill_diagonal(self.D, 0)

        #Matriz de flujo
        self.F = np.random.randint(1, 50, size=(N, N))
        np.fill_diagonal(self.F, 0)

        self.results = {}
        self.max_runtime = max_runtime

        self._num_evals = 0
        self._init_time = None

    def __str__(self):
        output = '-----QAP INSTANCE BEGIN--\n'
        output += str(self.N) + '\n'
        output += str(self.D) + '\n'
        output += str(self.F) + '\n'
        output += str(self.results) + '\n'
        output += str(self.max_runtime) + '\n'
        output += '-----QAP INSTANCE END__--\n'
        return output

    def clear_results(self):
        self.results = {}

    def get_results(self, alg):
        if alg in self.resutls:
            return self.results[alg]
        else:
            return [(1,self.sqrt_2 * self.size())]

    def size(self):
        return len(self.D)
    
    def get_num_evals(self):
        return self._num_evals

    def _reset_num_evals(self, max_evals=None):
        self._num_evals = 0

        if max_evals is not None:
            self.max_evals = max_evals

    def _reset_init_time(self):
        self._init_time = datetime.now()

    def cost(self, permutation):

        # print(permutation)
        if 0 in permutation:
            permutation_aux = permutation
        else:
            permutation_aux = [i-1 for i in permutation]
            
        cost = 0
        for i in range(len(permutation_aux)):
            for j in range(len(permutation_aux)):
                cost += self.D[permutation_aux[i]][permutation_aux[j]] * self.F[i][j]
        
        return cost

    def evaluate_and_resgister(self, permutation, alg_id):

        current_cost = self.cost(permutation)

        if not alg_id in self.results:
            self._reset_init_time()
            self._reset_num_evals()
            self.results[alg_id] = [(1,current_cost)]
        else:
            now_time = datetime.now()

            if now_time - self._init_time > timedelta(seconds=self.max_runtime):
                raise MaxRuntimeHit

            self._num_evals += 1
            value_to_be_registered = min(self.results[alg_id][-1][1], current_cost)

            if value_to_be_registered < self.results[alg_id][-1][1]:
                self.results[alg_id].append((self._num_evals, value_to_be_registered))

        return current_cost

class QAP_Experiment_Generator:

    def __init__(self, num_problems, min_num_facilities, max_num_facilities, max_runtime):
        self.min_num_facilities = min_num_facilities
        self.max_num_facilities = max_num_facilities
        self.set_of_facilities = []
        self.max_runtime = max_runtime
        self.initialise_problems(num_problems)

    def __str__(self):
        output = '------------QAP GENERATOR BEGIN---\n'
        output += str(len(self.set_of_facilities)) + " " + str(self.min_num_facilities) + " " + str(self.max_num_facilities) + " " + str(self.max_runtime)
        for i_problem in self.set_of_facilities:
            output += '\n' + str(i_problem) + '\n'
        output += '------------QAP GENERATOR END----\n'
        return output

    def introduce_a_new_instance(self):
        self.set_of_facilities.append(
            QAP_Instance(
                np.random.choice(np.arange(self.min_num_facilities,
                                           self.max_num_facilities)), self.max_runtime))
        
    def initialise_problems(self, num_problems):
        for _ in range(num_problems):
            self.introduce_a_new_instance()

    def set_size_axes(self,w,h,ax):
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    def plot_convergence_graphs(self, axes_or_filename, family):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                _, ax = plt.subplots()
        else:
            ax = axes_or_filename

        ax.set_xscale('log')

        # Normalize the results of the algorithms on every problem
        normalised_results = []
        for i_problem in self.set_of_facilities:
            normalised_results.append(pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()])))
            normalised_results[-1].ffill(axis=0, inplace=True)
            max_value = normalised_results[-1].max().max()
            min_value = normalised_results[-1].min().min()
            normalised_results[-1] -= min_value
            if max_value > min_value:
                normalised_results[-1] /= (max_value - min_value)

        # Executed algs
        algs = list(set([i for j in self.set_of_facilities for i in j.results]))

        for i in family:
            # Results of algorithm i on all the problems
            df = pd.DataFrame(dict([(index, j[i]) for index, j in enumerate(normalised_results) if i in j.columns]))
            df.ffill(axis=0, inplace=True)
            try:
                ax.step(df.index, np.mean(df, axis=1), where='post', label=i)
            except ValueError:
                print(normalised_results[0].index)
                print(normalised_results[1].index)
                print(normalised_results[2].index)
                print(list(df.index))
                print(df.loc[:20])
                raise
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=np.mean(df, axis=1), scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            # ax.fill_between(df.index, np.min(df, axis=1), np.max(df, axis=1), alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        self.set_size_axes(10,5,ax)
        ax.legend()

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def plot_rank_evolution_graph(self, axes_or_filename, family):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        ax.set_xscale('log')

        results_ranks = []

        # Executed algs
        algs = list(set([i for j in self.set_of_facilities for i in j.results]))

        ax.set_ylim(1,len(algs))

        for i_problem in self.set_of_facilities:
            data_results = pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()]))

            for i_alg in algs:
                if i_alg not in data_results.columns:
                    data_results = pd.concat((data_results, pd.DataFrame({i_alg: dict([(1,i_problem.size()*QAP_Instance.sqrt_2)])})), axis=1)

            data_results.ffill(axis=0, inplace=True)
            data_results = data_results.round(decimals=6)
            results_ranks.append(data_results.rank(axis=1))

        for i in family:
            df = pd.DataFrame({str(index): j[i] for index, j in enumerate(results_ranks)})
            df.ffill(axis=0, inplace=True)
            mean = np.mean(df, axis=1)
            std = np.std(df, axis=1)
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            ax.step(df.index, mean, where='post', label=i)
            # ax.fill_between(df.index, mean - std, mean + std, alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        self.set_size_axes(10,5,ax)
        ax.legend()
        
        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def clear_results(self):
        for i in self.set_of_facilities:
            i.clear_results()

    def nemenyi(self, family):
        results = []
        for i_problem in self.set_of_facilities:
            results.append(pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()])))
            results[-1].ffill(axis=0, inplace=True)

        algs = list(set([i for j in self.set_of_facilities for i in j.results]))

        for i_problem in self.set_of_facilities:
            data_results = pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()]))

            for i_alg in algs:
                if i_alg not in data_results.columns:
                    data_results = pd.concat((data_results, pd.DataFrame({i_alg: dict([(1,i_problem.size()*QAP_Instance.sqrt_2)])})), axis=1)

        df_final = pd.DataFrame()
        
        for i in family:
            df = pd.DataFrame(dict([(index, j[i]) for index, j in enumerate(results) if i in j.columns]))
            df.fillna(np.inf, inplace=True)
            best_results = df.min(axis=0)
            new_data = pd.DataFrame(best_results, columns=[str(i)])
            df_final = pd.concat([df_final, new_data], axis=1)

        return df_final
