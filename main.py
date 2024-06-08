import pickle
import os
from os import path
import sys
from hashlib import sha256
import logging
from inspect import getmembers, isfunction, signature
from tqdm import tqdm
import numpy as np
import random
from shutil import move
import scikit_posthocs as sp 

sys.path.append('.')
from QAP_generator import *
import mhs

def reset_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

def dict_2_str_recurive(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    output = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            output += '%s%s%s%s' %(ident,braces*'[',key,braces*']')
            output += dict_2_str_recurive(value, ident+'  ', braces+1)
        else:
            output += ident+'%s = %s' %(key, value)

    return output

def framework_configurator():

    #Se piden por entrada valores para variables. Si falla salta una excepción y se cierra el programa
    num_qap_instances = input('Introduce the initial number of QAP instances for the experiment: (integer)\n')
    try:
        num_qap_instances = int(num_qap_instances)
    except ValueError:
        print('\'', num_qap_instances, '\' could not be converted to integer', sep='')
        sys.exit(1)

    min_num_facilities = input('Introduce the minimum number of facilities for the QAP instances: (integer)\n')
    try:
        min_num_facilities = int(min_num_facilities)
    except ValueError:
        print('\'', min_num_facilities, '\' could not be converted to integer', sep='')
        sys.exit(1)

    max_num_facilities = input('Introduce the maximum number of facilities for the QAP instances: (integer)\n')
    try:
        max_num_facilities = int(max_num_facilities)

        while max_num_facilities <= min_num_facilities:
            print('The maximum number of facilities must be estrictly greater than the minimum number of facilities')
            max_num_facilities = input('Introduce the maximum number of facilities for the QAP instances: (integer)\n')
            max_num_facilities = int(max_num_facilities)

    except ValueError:
        print('\'', max_num_facilities, '\' could not be converted to integer', sep='')
        sys.exit(1)

    execution_max_time = input('Introduce the maximum running time of the MHs in seconds: (integer)\n')
    try:
        execution_max_time = int(execution_max_time)
    except ValueError:
        print('\'', execution_max_time, '\' could not be converted to integer', sep='')
        sys.exit(1)

    #Se crea "framework_data", con el campo 'QAP_generator' que contiene datos generados por la función QAP_Experiment_Generator
    framework_data = {}
    # framework_data['num_qap_instances'] = num_tsp_instances
    # framework_data['min_num_facilities'] = min_num_facilities
    # framework_data['max_num_facilities'] = max_num_facilities
    # framework_data['execution_max_time'] = execution_max_time
    framework_data['QAP_generator'] = QAP_Experiment_Generator(num_qap_instances, min_num_facilities, max_num_facilities, execution_max_time)
    hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()
    all_data = {'framework_data': framework_data, 'hash_string': hash_string}
    with open('data.pickle', 'wb') as f:
        pickle.dump(all_data, f)

def update_data_pickle(framework_data):
    hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()
    all_data = {'framework_data': framework_data, 'hash_string': hash_string}
    with open('data_new.pickle', 'wb') as f:
        pickle.dump(all_data, f)

    # os.rename('data_new.pickle', 'data.pickle')
    move('data_new.pickle', 'data.pickle')
    logging.info(str(datetime.now())+': DATABASE UPDATED: data.pickle')

def load_and_check_data_pickle():

    # BASIC LOAD AND CHECK CONSISTENCY
    # Se intenta abrir el archivo data en formato pickle, si no se encuentra se imprime un error
    if not path.exists('data.pickle'):
        print('Data file data.pickle not found. Running comparison framework configurator')

        #Se llama a una función para crear el framework pickle con los datos que introduzca el usuario
        framework_configurator()

    with open('data.pickle', 'rb') as f:
        all_data = pickle.load(f)
        framework_data = all_data['framework_data']
        read_hash_string = all_data['hash_string']

        #Comprobamos que el archivo no esté corrupto con un código hash
        real_hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()

        if real_hash_string != read_hash_string:
            print('Corrupt file \'data.pickle\'. Please, remove it and run again the program')
            sys.exit(1)

    experimenter = framework_data['QAP_generator']
    print('Comparison framework with the following parameter settings:')
    print('Num of QAP instances:', len(experimenter.set_of_facilities))
    print('Min number of facilities:', experimenter.min_num_facilities)
    print('Max number of facilities:', experimenter.max_num_facilities)
    print('Max MH run time:', experimenter.max_runtime)
    # print('QAP_generator:', framework_data['QAP_generator'])

    # CHECK THAT ALL MHs have been run on all the problems
    # experimenter = framework_data['QAP_generator']
    # algs = list(set([i for j in experimenter.set_of_facilities for i in j.results]))
    #
    # for i in algs:
    #     for j in experimenter.set_of_facilities:
    #         if i not in j.results:
    #             print('MH not runned:', i, j)
    #             sys.exit(1)

    return framework_data

#Función para el menú que aparece al arrancar main
def menu(data):

    #experimenter será un objeto de la clase QAP_Experiment_Generator con los datos del framework
    experimenter = data['QAP_generator']
    algs = list(set([i for j in experimenter.set_of_facilities for i in j.results]))

    results_allMHs_on_allInstances = True

    #Se imprimen los resultados para cada algoritmo
    print('\nThere are results for:')
    for index, j in enumerate(experimenter.set_of_facilities):
        print('Instance ', index, ': ', sep='', end='')
        for i_alg in algs:
            if i_alg in j.results:
                if len(j.results[i_alg]) >= 1:
                    print(i_alg, ', ', sep='', end='')
                # else:
                #     print('Error: Too little data for a MH. Please remove this data: ', i_alg, '(', len(j.results[i_alg]),')', sep='')
            else:
                results_allMHs_on_allInstances = False
        print('')

    print('\nMENU')
    print('1: Continue running MHs')
    print('2: Remove all the results of a MH')
    print('3: Increase the number of instances by one')
    print('4: Generate plots')
    print('^C: exit')
    try:
        option = input('Please, introduce your choice (1/2/3/4):\n')
    except KeyboardInterrupt:
        return 'exit'

    if option == '1':
        return option
    elif option == '2':
        mh_name = input('Please, introduce the name of the MH:\n')
        any_deletion = False
        confirmation = input('We are removing the results of ' + mh_name + '. Proceed? (y/n)\n')
        if confirmation == 'y':
            for index_instance, i_qap_instance in enumerate(experimenter.set_of_facilities):
                if mh_name in i_qap_instance.results:
                    del i_qap_instance.results[mh_name]
                    logging.info(str(datetime.now())+': RESULTS REMOVED (pickle update pending): '+mh_name+' on TSP instance '+str(index_instance))
                    any_deletion = True
            if not any_deletion:
                print('No results found (nor removed)')
            else:
                update_data_pickle(data)
                print('Deletion completed')
        else:
            print('Deletion cancelled')

    elif option == '3':
        experimenter.introduce_a_new_instance()
        logging.info(str(datetime.now()) + ': NEW QAP INSTANCE (pickle update pending): ' + str(len(experimenter.set_of_facilities) - 1))
        update_data_pickle(data)
        print('Instance created and added')
    elif option == '4':
        if not results_allMHs_on_allInstances:
            print('Plots not generated. All the MHs must have been applied on all the TSP instance.')
        else:
            print('Generating plots:')
            with tqdm(total=26) as pbar:
                family = ['random_search','hill_climb','steepest_ascent_hill_climbing','simulated_annealing','tabu_search','mcts_permutation','ILS','vns','grasp','iterated_greedy','LPSO','ABC_optimize','ant_colony_optimization','bat_algorithm_shuffled_array','firefly_algorithm','chc','pufferfish_optimization']
                data['QAP_generator'].plot_convergence_graphs('TODOS_CONVERGENCIA.png', family)
                pbar.update()

                data['QAP_generator'].plot_rank_evolution_graph('TODOS_RANKING.png', family)
                pbar.update()

                family = ['random_search']
                data['QAP_generator'].plot_convergence_graphs('RANDOM_SEARCH.png', family)
                pbar.update()

                family = ['random_search','hill_climb']
                data['QAP_generator'].plot_convergence_graphs('HILLCLIMB.png', family)
                pbar.update()

                family = ['random_search','steepest_ascent_hill_climbing']
                data['QAP_generator'].plot_convergence_graphs('steepest_ascent_hill_climbing.png', family)
                pbar.update()

                family = ['random_search','simulated_annealing']
                data['QAP_generator'].plot_convergence_graphs('simulated_annealing.png', family)
                pbar.update()

                family = ['random_search','mcts_permutation']
                data['QAP_generator'].plot_convergence_graphs('mcts_permutation.png', family)
                pbar.update()

                family = ['random_search','ILS']
                data['QAP_generator'].plot_convergence_graphs('ILS.png', family)
                pbar.update()

                family = ['random_search','vns']
                data['QAP_generator'].plot_convergence_graphs('vns.png', family)
                pbar.update()

                family = ['random_search','grasp']
                data['QAP_generator'].plot_convergence_graphs('grasp.png', family)
                pbar.update()

                family = ['random_search','iterated_greedy']
                data['QAP_generator'].plot_convergence_graphs('iterated_greedy.png', family)
                pbar.update()

                family = ['random_search','LPSO']
                data['QAP_generator'].plot_convergence_graphs('LPSO.png', family)
                pbar.update()

                family = ['random_search','ABC_optimize']
                data['QAP_generator'].plot_convergence_graphs('ABC_optimize.png', family)
                pbar.update()

                family = ['random_search','ant_colony_optimization']
                data['QAP_generator'].plot_convergence_graphs('ant_colony_optimization.png', family)
                pbar.update()

                family = ['random_search','bat_algorithm_shuffled_array']
                data['QAP_generator'].plot_convergence_graphs('bat_algorithm_shuffled_array.png', family)
                pbar.update()

                family = ['random_search','firefly_algorithm']
                data['QAP_generator'].plot_convergence_graphs('firefly_algorithm.png', family)
                pbar.update()

                family = ['random_search','chc']
                data['QAP_generator'].plot_convergence_graphs('chc.png', family)
                pbar.update()

                family = ['random_search','pufferfish_optimization']
                data['QAP_generator'].plot_convergence_graphs('pufferfish_optimization.png', family)
                pbar.update()

                family = ['random_search','tabu_search']
                data['QAP_generator'].plot_convergence_graphs('tabu_search.png', family)
                pbar.update()

                family = ['hill_climb','steepest_ascent_hill_climbing','simulated_annealing','tabu_search','mcts_permutation','ILS','vns','iterated_greedy']
                data['QAP_generator'].plot_convergence_graphs('local.png', family)
                pbar.update()

                family = ['LPSO','ABC_optimize','ant_colony_optimization','bat_algorithm_shuffled_array','firefly_algorithm','chc','pufferfish_optimization']
                data['QAP_generator'].plot_convergence_graphs('poblaciones.png', family)
                pbar.update()
                
                family = ['steepest_ascent_hill_climbing','tabu_search','ILS','vns','bat_algorithm_shuffled_array']
                data['QAP_generator'].plot_convergence_graphs('5mejores.png', family)
                pbar.update()
                
                family = ['tabu_search','ILS','vns']
                data['QAP_generator'].plot_convergence_graphs('3mejores.png', family)
                pbar.update()
                
                family = ['hill_climb','mcts_permutation','pufferfish_optimization','firefly_algorithm','simulated_annealing']
                data['QAP_generator'].plot_convergence_graphs('5peores.png', family)
                pbar.update()
                
                family = ['hill_climb','mcts_permutation','pufferfish_optimization','firefly_algorithm','simulated_annealing','iterated_greedy','chc','LPSO','ant_colony_optimization']
                data['QAP_generator'].plot_convergence_graphs('peoresaleatorio.png', family)
                pbar.update()
                
                family = ['steepest_ascent_hill_climbing','tabu_search','ILS','vns','bat_algorithm_shuffled_array','ABC_optimize']
                data['QAP_generator'].plot_convergence_graphs('mejoresaleatorio.png', family)
                pbar.update()
                
                family = ['hill_climb','steepest_ascent_hill_climbing','simulated_annealing','tabu_search','mcts_permutation','ILS','vns','iterated_greedy']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['hill_climb'], dt['steepest_ascent_hill_climbing'], dt['steepest_ascent_hill_climbing'], dt['tabu_search'], dt['mcts_permutation'], dt['ILS'], dt['vns'], dt['iterated_greedy']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

                family = ['LPSO','ABC_optimize','ant_colony_optimization','bat_algorithm_shuffled_array','firefly_algorithm','chc','pufferfish_optimization']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['LPSO'], dt['ABC_optimize'], dt['ant_colony_optimization'], dt['bat_algorithm_shuffled_array'], dt['firefly_algorithm'], dt['chc'], dt['pufferfish_optimization']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

                family = ['steepest_ascent_hill_climbing','tabu_search','ILS','vns','bat_algorithm_shuffled_array']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['steepest_ascent_hill_climbing'], dt['tabu_search'], dt['ILS'], dt['bat_algorithm_shuffled_array'], dt['vns']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

                family = ['tabu_search','ILS','vns']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['tabu_search'], dt['ILS'], dt['vns']])
                print(sp.el(dt_final.T))

                family = ['hill_climb','mcts_permutation','pufferfish_optimization','firefly_algorithm','simulated_annealing']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['hill_climb'], dt['mcts_permutation'], dt['pufferfish_optimization'], dt['firefly_algorithm'], dt['simulated_annealing']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

                family = ['hill_climb','mcts_permutation','pufferfish_optimization','firefly_algorithm','simulated_annealing','iterated_greedy','chc','LPSO','ant_colony_optimization']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['hill_climb'], dt['mcts_permutation'], dt['pufferfish_optimization'], dt['firefly_algorithm'], dt['simulated_annealing'], dt['iterated_greedy'], dt['chc'], dt['LPSO'], dt['ant_colony_optimization']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

                family = ['steepest_ascent_hill_climbing','tabu_search','ILS','vns','bat_algorithm_shuffled_array','ABC_optimize']
                dt = data['QAP_generator'].nemenyi(family)
                dt_final = np.array([dt['steepest_ascent_hill_climbing'], dt['tabu_search'], dt['ILS'], dt['vns'], dt['bat_algorithm_shuffled_array'], dt['ABC_optimize']])
                print(sp.posthoc_nemenyi_friedman(dt_final.T))

    else:
        print('Not a valid option: ', option)

    return None

def run(mh_function, qap_instance, index):
    
    #Corremos el algoritmo solo si no tenemos resultados para el
    if mh_function.__name__ not in qap_instance.results:

        #Creamos una función para obtener la función de evaluación del algoritmo
        def get_fitness_function(name):

            #Esta función llama a evaluate_and_resgister(), que devuelve un coste cuando se le pasa un algoritmo y una permutación
            def fitness_function(solution):
                return qap_instance.evaluate_and_resgister(solution, name)
            return fitness_function
        # mh_function(i.size(), i.evaluate_and_resgister)

        #Obtenemos la función de fitness para más tarde
        ffunction = get_fitness_function(mh_function.__name__)
        num_facilities = qap_instance.size()
        try:
            logging.info(str(datetime.now())+': EXECUTION BEGINS: ' + mh_function.__name__ + ' on QAP intsance ' + str(index))
            mh_function(num_facilities, ffunction)
        except MaxRuntimeHit:
            logging.info(str(datetime.now())+': EXECUTION ENDED (pickle update pending): ' + mh_function.__name__ + ' on QAP intsance ' + str(index))
            update_data_pickle(data)
        except KeyboardInterrupt:
            print('Executions interrupted')
            logging.info(str(datetime.now())+': EXECUTION INTERRUPTED: ' + mh_function.__name__ + ' on QAP intsance ' + str(index))
            raise
        except Exception as e:
            print('ERROR: Something went wrong with a MH. See log.log', mh_function.__name__)
            logging.error(str(datetime.now())+':\n'+str(e), exc_info=True)
            sys.exit(1)
        else:
            print('No MAX_RUNTIME_HIT: The following algorithm terminated before the maximal runtime and results were not saved: ' + mh_function.__name__)
            print('Did you replace the main loop with While True:?')
            sys.exit(1)

if __name__ == '__main__':

    #Función para guardar los logs en el archivo log.log (a nivel de INFO)
    logging.basicConfig(level=logging.INFO, filename='log.log', filemode='a')

    data = load_and_check_data_pickle()
    experimenter = data['QAP_generator']

    while True:
        try:
            option = menu(data)
        except KeyboardInterrupt:
            option = 'None'

        if option == '1':

            # Get the list of executions not runned
            pending_executions = []

            #name es el nombre de la función del algoritmo, y mh_function es la propia función en sí
            for name, mh_function in getmembers(mhs, isfunction):

                #Comprobamos que la función reciba solamente dos parámetros (N y f)
                sig = signature(mh_function)
                if len(sig.parameters) == 2:
                    for index, instance in enumerate(experimenter.set_of_facilities):
                        if name not in instance.results:
                            pending_executions.append((mh_function, instance, index))
                else:
                    print('There are functions in mhs.py with less or more than two arguments. This is discouraged! ', name+str(sig))
                    print(
                        'mhs.py should just contains a function per MH and every one should recieve exactly two arguments: ', end='')
                    print('(N, f), with N the size of the problem and f the evaluation function')
                    print('In case some operators of the MH are defined as another function, include them into the function of the MH. For instance:')
                    print('def simulated_annealing(N, f):')
                    print('   def neigh_operator(solution):')
                    print('      ...')
                    print('   ...')
                    print('   new_solution = neigh_operator(solution)')
                    print('   ...\n')
                    logging.warning(str(datetime.now()) + ': NO MH in mhs.py: ' + name + str(sig))

            print('EXECUTING MHs (you can kill the process with ^C whenever you want. Results of the executions would be stored incrementally. Those of the last execution would be discarded)')
            for i in tqdm(pending_executions):
                try:
                    run(i[0], i[1], i[2])
                except KeyboardInterrupt:
                    option = 'exit'
                    break

        if option == 'exit':
            break

        _ = input('Press Enter to continue')
        sys.stdin.flush()
