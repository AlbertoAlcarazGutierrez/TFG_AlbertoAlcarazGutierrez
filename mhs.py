import numpy as np
import random
from random import shuffle, sample
import math
import time
np.seterr(all='raise')

def random_search(N, f):
  """
  Performs random search to find the best permutation for a given function.

  Args:
      N: The number of elements in the permutation.
      f: The function to evaluate the fitness of a permutation.

  Returns:
      A tuple containing the best permutation found and its fitness value.
  """

  # Generate a random initial permutation
  best_permutation = random.sample(range(N), N)
  best_fitness = f(best_permutation)

  # Loop for a fixed number of iterations (replace with your stopping criterion)
  while True:
    # Generate a new random permutation
    candidate_permutation = random.sample(range(N), N)

    # Evaluate the candidate permutation
    candidate_fitness = f(candidate_permutation)

    # Update the best solution if found a better one
    if candidate_fitness < best_fitness:
      best_permutation = candidate_permutation
      best_fitness = candidate_fitness

def simulated_annealing(N, f):
  """
  Performs simulated annealing to find a near-optimal permutation for function f.

  Args:
      N: Number of elements in the permutation (1 to N).
      f: Function to evaluate the quality of a permutation.
          Takes a list of size N as input and returns a numeric value.
          Lower value indicates better solution.
      min_temperature: Minimum temperature threshold (default: 0.001).
      start_temp: Initial temperature (default: 100).
      cooling_rate: Rate at which temperature cools down (default: 0.95).

  Returns:
      A list representing the best permutation found and its corresponding score.
  """
  # Generate initial solution (random permutation)
  current_solution = list(range(1, N + 1))
  random.shuffle(current_solution)
  current_score = f(current_solution)
  best_solution = current_solution.copy()
  best_score = current_score

  min_temperature=0.00001
  start_temp=10000
  cooling_rate=0.995

  temperature = start_temp

  while True:
    # Generate a new neighbor solution (swap two elements)
    new_solution = current_solution.copy()
    i, j = random.sample(range(N), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

    # Evaluate new solution
    new_score = f(current_solution)

    # Decide whether to accept the new solution
    delta_e = new_score - current_score
    if delta_e < 0:  # New solution is better, always accept
      current_solution = new_solution
      current_score = new_score
    else:
      # Probability of accepting a worse solution based on temperature
      p = math.exp(-min(delta_e, 10) / temperature)  # Limit delta_e
      if random.random() < p:
        current_solution = new_solution
        current_score = new_score

    # Update best solution if found a better one
    if current_score < best_score:
      best_solution = current_solution.copy()
      best_score = current_score

    # Cool down the temperature (bounded by min_temperature)
    temperature = max(temperature * cooling_rate, min_temperature)

import random

def LPSO(N, f):
  """
  Local Particle Swarm Optimization for permutation problems

  Args:
      N: Number of elements in the permutation (1 to N)
      f: Function to evaluate a permutation (higher value is better)
      max_iter: Maximum number of iterations
      c1: Cognitive learning rate
      c2: Social learning rate (local neighborhood)
      w: Inertia weight
      swap_prob: Probability of performing a random swap
      vel_max: Maximum velocity (positive for larger movements)
      vel_min: Minimum velocity (negative for smaller movements)

  Returns:
      Best solution found and its fitness value
  """

  c1=2
  c2=1
  w=0.7
  swap_prob=0.2
  vel_max=1.0
  vel_min=-1.0

  # Initialize swarm
  swarm = []
  for _ in range(N):
    perm = list(range(1, N+1))  # Permutation from 1 to N
    random.shuffle(perm)  # Randomly shuffle for initial diversity
    swarm.append({
      "position": perm,
      "pbest": perm.copy(),
      "pbest_val": f(perm),
      "velocity": [0] * N
    })

  # Neighborhood size (consider a small radius for local exploration)
  neighborhood_size = 2

  # Main loop
  while True:
    for i, particle in enumerate(swarm):
      # Update local best

      if f(particle["position"]) > particle["pbest_val"]:
        particle["pbest"] = particle["position"].copy()
        particle["pbest_val"] = f(perm)

      # Select neighborhood
      neighborhood = [swarm[j] for j in range(max(0, i-neighborhood_size//2), min(N, i+neighborhood_size//2+1))]

      # Find local best in neighborhood
      local_best = max(neighborhood, key=lambda p: p["pbest_val"])

      # Update velocity (with diversity bias)
      for j in range(N):
        cognitive = c1 * random.random() * (particle["pbest"][j] - particle["position"][j])
        social = c2 * random.random() * (local_best["pbest"][j] - particle["position"][j])
        particle["velocity"][j] = w * particle["velocity"][j] + cognitive + social
        # Clamping with bias towards larger movements
        particle["velocity"][j] = max(min(particle["velocity"][j], vel_max), vel_min)

     # Update position (ensure unique elements and valid permutation)
      for j in range(N):
        available_values = set(range(1, N+1))  # All possible values
        available_values.difference_update(set(particle["position"]))  # Remove all used values
        # Find a valid index for the new element
        valid_index = (particle["position"][j] + int(particle["velocity"][j])) % N
        clip_value = 0.5  # Clip velocity to a reasonable range (adjust as needed)
        particle["velocity"][j] = min(max(particle["velocity"][j], -clip_value), clip_value)
        while valid_index in particle["position"]:
          valid_index = (valid_index + 1) % N
        # Iterative swap if available_values is empty
        if not available_values:
          for swap_index in range(j+1, N):  # Iterate from next position onwards
            if valid_index in particle["position"][swap_index:]:
              continue  # Skip if already used in the remaining elements
            particle["position"][j], particle["position"][swap_index] = particle["position"][swap_index], particle["position"][j]
            break  # Stop after finding a valid swap
        # Update position
        particle["position"][j] = available_values.pop() if available_values else particle["position"][j]

def ABC_optimize(N, f):
  """
  Implements the Artificial Bee Colony algorithm to find the optimal permutation among N integers.

  Args:
      N: Number of elements in the permutation (e.g., N=3 for [1, 2, 3]).
      f: Function to evaluate the fitness of a permutation.
      limit: Threshold for abandoning an employed bee's food source (default 10).

  Returns:
      The best permutation found and its corresponding fitness value.
  """
  limit=10
  
  # Colony sizes (can be adjusted for different problems)
  employed_bees = N
  onlooker_bees = employed_bees

  # Employed bee solutions (initially random permutations)
  employed = []
  for _ in range(employed_bees):
    employed.append(random.sample(range(1, N+1), N))

  # Fitness values for employed bees
  fitness_values = [f(perm) for perm in employed]

  # Global variables to track best solution
  best_solution = employed[0]
  best_fitness = fitness_values[0]

  iter_ = 0
  while True:
    # Onlooker bee phase
    for i in range(onlooker_bees):
      # Select a source based on roulette wheel selection (probability based on fitness)
      p = random.random()
      total = sum(fitness_values)
      selected_source = 0
      while p > 0:
        p -= fitness_values[selected_source] / total
        selected_source += 1
      selected_source -= 1

      # Employed bee's solution for reference
      origin = employed[selected_source]

      # Modify the solution slightly (neighbor search)
      neighbor = origin.copy()
      i, j = random.sample(range(N), 2)  # Select two random indexes
      neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

      # Evaluate the neighbor solution
      neighbor_fitness = f(neighbor)

      # Select the better solution (exploitation)
      if neighbor_fitness > fitness_values[selected_source]:
        employed[selected_source] = neighbor.copy()
        fitness_values[selected_source] = neighbor_fitness

    # Scout bee phase
    for i in range(employed_bees):
      # Check if solution hasn't improved in 'limit' iterations (stagnation)
      if iter_ - employed.index(employed[i]) > limit:
        # Become a scout and find a new random solution
        employed[i] = random.sample(range(1, N+1), N)

        fitness_values[i] = f(employed[i])

    # Update best solution if necessary
    for i in range(employed_bees):
      if fitness_values[i] < best_fitness:
        best_solution = employed[i].copy()
        best_fitness = fitness_values[i]

    iter_ += 1

#ERROR CON RHO
def ant_colony_optimization(N, f):
  """
  Implements Ant Colony Optimization for permutation problems.

  Args:
      N: Number of elements in the permutation (problem size).
      f: Function to evaluate the quality of a permutation.
      rho: Pheromone evaporation rate.
      iterations: Number of iterations.
      ants: Number of ants to simulate.

  Returns:
      A tuple containing the best permutation found and its fitness value.
  """
  rho=0.1
  ants=10
  min_pheromone = 1e-6

  # Initialize pheromone trails
  pheromone = [[0.01 for _ in range(N)] for _ in range(N)]

  # Main loop
  best_solution = None
  best_fitness = float('inf')
  while True:
    # Generate ant solutions
    solutions = []
    for _ in range(ants):
      solution = list(range(N))
      random.shuffle(solution)

      fitness = f(solution)
      solutions.append((solution, fitness))

    # Update pheromone trails
    for solution, fitness in solutions:
      for i in range(N - 1):
        for j in range(i + 1, N):
          delta_tau = rho * fitness if fitness > best_fitness else 0
          pheromone[solution[i]][solution[j]] += delta_tau
          pheromone[solution[j]][solution[i]] += delta_tau

    # Evaporate pheromone trails
    for i in range(N):
      for j in range(N):
        pheromone[i][j] = max(min_pheromone, pheromone[i][j] * (1 - rho))

    # Update best solution
    for solution, fitness in solutions:
      if fitness < best_fitness:
        best_solution = solution
        best_fitness = fitness

def bat_algorithm_shuffled_array(N, f):
  """
  Bat-Inspired Algorithm for generating shuffled arrays of [1, N].

  Args:
      N: Number of elements (length of the array).
      f: Objective function to evaluate a permutation (lower is better).
      max_iterations: Maximum number of iterations for the algorithm.
      population_size: Number of virtual bats in the population.
      alpha: Initial value for loudness (typically between 0.5 and 1.0).
      gamma: Minimum value for loudness (typically 0.0).

  Returns:
      The best shuffled array found and its corresponding objective function value.
  """
  population_size=20
  alpha=0.9
  gamma=0.0

  # Initialize bat population
  population = []
  for _ in range(population_size):
    available_elements = [i for i in range(1, N+1)]  # List containing elements from 1 to N
    permutation = []
    for _ in range(N):
      index = random.randrange(len(available_elements))  # Random index within available options
      permutation.append(available_elements.pop(index))  # Append and remove from available options
    population.append({
      "position": permutation,
      "velocity": [0 for _ in range(N)],
      "frequency": random.uniform(0, 1),  # Random frequency
      "loudness": 1.0,
      "fitness": f(permutation)  # Evaluate initial fitness
    })

  # Bat Algorithm loop
  best_bat = min(population, key=lambda bat: bat["fitness"])
  while True:

  # Update frequency and loudness
    for bat in population:
      bat["frequency"] = min(bat["frequency"] + random.uniform(-1, 1), 1.0)
      bat["loudness"] = alpha + (gamma - alpha) * random.uniform(0, 1)

  # Update velocity based on frequency, loudness, best solution, and current velocity
    for bat in population:
      for i in range(N):
        avg_velocity = sum(p["velocity"][i] for p in population) / population_size
        new_velocity = bat["velocity"][i] + (bat["frequency"] * (best_bat["position"][i] - bat["position"][i])) * bat["loudness"] + random.gauss(0, 1) * avg_velocity
        bat["velocity"][i] = new_velocity

    # Update position (ensure it remains a permutation of [1, N])
    for bat in population:
      for i in range(N):
        new_velocity = bat["velocity"][i]
        # Update position with a swap operation based on velocity
        swap_index = int(round(bat["position"][i] + new_velocity)) % N
        if swap_index != i:
          bat["position"][i], bat["position"][swap_index] = bat["position"][swap_index], bat["position"][i]

    # Evaluate fitness of updated positions
    for bat in population:
      bat["fitness"] = f(bat["position"])

    # Update best solution
    current_best = min(population, key=lambda bat: bat["fitness"])
    if current_best["fitness"] < best_bat["fitness"]:
      best_bat = current_best

#MODIFICADA Y 1 CAMBIO
def firefly_algorithm(N, f):
  """
  Firefly algorithm for permutation optimization (infinite loop).

  Args:
      N: Number of elements in the permutation (1 to N).
      f: Objective function to evaluate permutations.
      alpha: Light attraction coefficient (0.0 to 1.0).
      beta: Attraction coefficient (0.0 to 1.0).
  """
  alpha=0.2
  beta=0.4

  # Generate initial population of fireflies (permutations)
  population = [random.sample(range(1, N + 1), N) for _ in range(N)]  # Sample N elements from 1 to N

  # Convert elements to integers (optional)
  for perm in population:
    for i in range(N):
      perm[i] = int(perm[i])  # Convert each element to integer

  values = [f(p) for p in population]  # Evaluate fitness of each firefly

  # Main loop (infinite)
  while True:
    for i in range(N):
      for j in range(N):
        if values[j] < values[i]:  # Brighter firefly (smaller objective value)
          distance = sum((a - b)**2 for a, b in zip(population[i], population[j]))
          attraction = alpha * math.exp(-beta * distance)
          
          # Update movement with integer bounds
          for k in range(N):
            step = int(random.uniform(0, 1) * attraction * (population[j][k] - population[i][k]))
            new_value = population[i][k] + step

            # Ensure new value stays within 1 to N (inclusive)
            population[i][k] = max(1, min(N, new_value))  # Clamp to valid range

      # Evaluate fitness after movement
      values = []
      for p in population:
        values.append(f(p))
      #values = [f(p) for p in population]

    # Find the brightest firefly (best solution with smallest value)
    best_index = values.index(min(values))  # Find index of permutation with minimum value
    best_solution = population[best_index]
    best_value = values[best_index]

#MODIFICADA
def tabu_search(N, f):
  """
  Performs Tabu Search to find the best permutation for function f.

  Args:
      N: Number of elements in the permutation (1 to N).
      f: Function to evaluate the quality of a permutation.
          Takes a list of size N as input and returns a float.
          Lower values indicate better solutions.
      tabu_size: Size of the tabu list (default: 10).

  Returns:
      A list representing the best permutation found and its evaluation.
  """
  
  # Generate initial random permutation
  current_solution = random.sample(range(1, N + 1), N)
  best_solution = current_solution.copy()
  best_score = f(best_solution)
  tabu_size = 5

  # Initialize empty tabu list
  tabu_list = []
  
  while True:

    # Generate neighborhood (swap two elements)
    neighbors = []
    for i in range(N):
      for j in range(i + 1, N):
        neighbor = current_solution.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)

    # Find best non-tabu neighbor
    best_neighbor = None
    best_neighbor_score = float('inf')  # Initialize with worst possible score
    for neighbor in neighbors:
      if neighbor not in tabu_list:

        neighbor_score = f(neighbor)
        if neighbor_score < best_neighbor_score:
          best_neighbor = neighbor
          best_neighbor_score = neighbor_score

    # Aspiration criteria: Update even if tabu if better than global best
    if best_neighbor and f(best_neighbor) < best_score:
      best_solution = best_neighbor.copy()
      best_score = f(best_neighbor)
    
    # Update current solution
    current_solution = best_neighbor

    # Dynamic tabu list update: Only add if space and not already there
    if len(tabu_list) < tabu_size:
      tabu_list.append(current_solution.copy())
    elif current_solution not in tabu_list:
      tabu_list.pop(0)  # Remove oldest entry if full and not already there
      tabu_list.append(current_solution.copy())

def mcts_permutation(N, f):
  """
  Performs MCTS to find the best permutation for a given function f.

  Args:
      N: Number of elements in the permutation (e.g., N=3 for [1, 2, 3]).
      f: Function to evaluate the quality of a permutation.
      tabu_tenure: Number of iterations a solution stays in the tabu list (default: 10).
      acceptance_rate: Probability of accepting a worse solution (default: 0.9).

  Returns:
      The best permutation found and its corresponding function value.
  """
  tabu_tenure=10
  acceptance_rate=0.9

  # Initial solution (random permutation)
  current_state = list(range(N))
  random.shuffle(current_state)
  best_state = current_state.copy()
  best_value = f(best_state)

  # Tabu list to store recently explored solutions
  tabu_list = []

  while True:
    # Generate candidate neighbors
    neighbors = [(current_state[:i] + [current_state[j]] + 
                  current_state[i+1:j] + [current_state[i]] + current_state[j+1:]) 
                 for i in range(N) for j in range(i+1, N)]

    # Filter out tabu neighbors and potentially add some randomness
    allowed_neighbors = [n for n in neighbors if n not in tabu_list]

    # Calculate and use scaled weights (avoiding overflow)
    if allowed_neighbors:
      max_weight = max(f(n) for n in allowed_neighbors)  # Initialize and calculate max_weight
      scaled_weights = [f(n) / max_weight for n in allowed_neighbors]
      next_state = random.choices(allowed_neighbors, weights=scaled_weights)[0]
    else:
      # If all neighbors are tabu, choose a random one with some acceptance chance
      next_state = random.choices(neighbors, k=1)[0]
      if random.random() > acceptance_rate:
        continue  # Stay in current state with low probability

    # Update tabu list
    tabu_list.append(current_state)
    tabu_list = tabu_list[-tabu_tenure:]  # Keep only the most recent tabu entries

    # Evaluate and update best solution

    next_value = f(next_state)
    if next_value < best_value:
      best_state = next_state.copy()
      best_value = next_value

    # Move to the next state
    current_state = next_state

def ILS(N, f):
  """
  Iterative Local Search for finding the best permutation (N first integers).

  Args:
      N: Number of elements in the permutation (e.g., N=3 for [0, 1, 2]).
      f: Objective function to evaluate a permutation (lower is better).
      max_local_search: Maximum number of steps in local search.

  Returns:
      The best permutation found and its corresponding objective function value.
  """
  max_local_search=100

  # Generate initial random permutation
  current_solution = list(range(N))
  random.shuffle(current_solution)
  current_value = f(current_solution)
  best_solution = current_solution.copy()
  best_value = current_value

  while True:
    # Local Search
    for _ in range(max_local_search):
      # Generate a neighbor by swapping two elements
      neighbor = current_solution.copy()
      i, j = random.sample(range(N), 2)
      neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

      neighbor_value = f(neighbor)
      if neighbor_value < current_value:
        current_solution = neighbor
        current_value = neighbor_value

    # Perturbation (random swap)
    i, j = random.sample(range(N), 2)
    current_solution[i], current_solution[j] = current_solution[j], current_solution[i]

    # Update best solution
    if current_value < best_value:
      best_solution = current_solution.copy()
      best_value = current_value

#1 MODIFICACION, 2 CAMBIOS
def vns(N, f):
  """
  Performs Variable Neighbourhood Search to find the optimal permutation (N first integers)
  using function f to evaluate solutions.

  **Warning:** Using while True loop can lead to infinite loops if stop_signal is not implemented.

  Args:
      N: Number of elements in the permutation (e.g., N=4 for [0, 1, 2, 3])
      f: Function to evaluate the quality of a permutation (lower is better)
      max_local_search: Maximum number of iterations for local search within a neighborhood
      stop_signal: Optional function that checks for a stop signal (e.g., time limit)

  Returns:
      The best permutation found and its corresponding evaluation
  """
  max_local_search=100
  stop_signal=None

  # Generate random initial solution
  current_solution = list(range(N))
  shuffle(current_solution)
  best_solution = current_solution.copy()
  best_value = f(current_solution)

  def swap_neighborhood(solution, start=None, end=None):
    """
    Performs a swap operation on a permutation.

    Args:
        solution: The permutation to modify (list of integers)
        start: Optional starting index for swap (default: None for random swap)
        end: Optional ending index for swap (default: None for random swap)
    """
    if start is None or end is None:
      i, j = random.sample(range(N), 2)  # Random swap if no start/end provided
    else:
      i, j = start, end  # Swap within specified indices
    solution[i], solution[j] = solution[j], solution[i]
    return solution

  def reverse_neighborhood(solution, start, end):
    while start < end:
      solution[start], solution[end] = solution[end], solution[start]
      start += 1
      end -= 1
    return solution

  neighborhood_funcs = [swap_neighborhood, reverse_neighborhood]

  while True:
    current_neighborhood = 0
    improvement = True

    # Loop through neighborhoods
    while improvement and current_neighborhood < len(neighborhood_funcs):
      improvement = False

      # Local search within current neighborhood
      for _ in range(max_local_search):
        # Generate random start and end positions within valid range
        start = random.randint(0, N-2)  # Ensures end can be larger than start
        end = random.randint(start + 1, N-1)
        neighbor = neighborhood_funcs[current_neighborhood](current_solution.copy(), start, end)

        neighbor_value = f(neighbor)
        if neighbor_value < best_value:
          current_solution = neighbor.copy()
          best_solution = current_solution.copy()
          best_value = neighbor_value
          improvement = True

      current_neighborhood += 1

def grasp(N, f):
  """
  Implements the GRASP metaheuristic for finding the best permutation
  using function f to evaluate candidate solutions (minimization problem).

  Args:
      N: Number of elements in the permutation (e.g., N first integers).
      f: Function to evaluate a candidate solution (permutation).
      max_time: Maximum allowed execution time in seconds.

  Returns:
      The best solution (permutation) found and its corresponding evaluation.
  """

  def local_search(candidate):
    """
    Performs a simple 2-opt local search on the candidate permutation.

    Args:
        candidate: A list representing the current permutation solution.

    Returns:
        A new permutation potentially better than the input candidate.
    """
    N = len(candidate)
    improved = True

    while improved:
      improved = False
      for i in range(1, N - 2):
        for j in range(i + 2, N):
          # Reverse the sub-segment from i+1 to j (inclusive)
          new_candidate = candidate[:i+1] + candidate[j:i:-1] + candidate[j+1:]
          
          # Check if the reversed permutation improves the score
          if f(new_candidate) < f(candidate):
            candidate = new_candidate
            improved = True
            break  # Exit inner loop after finding improvement
    return candidate
  
  best_solution = None
  best_score = float('inf')  # Initialize with positive infinity for minimization

  start_time = time.time()
  while True:

    # 1. Construction with Greedy Twist
    candidate = list(range(N))  # Initial candidate (sorted permutation)
    shuffle(candidate)  # Randomize initial order slightly
    
    # Greedy selection of elements to keep
    for i in range(N):
      # Evaluate current candidate with and without element i
      score_with = f(candidate[:i] + candidate[i+1:])
      score_without = f(candidate[:i] + candidate[i+1:])
      
      # Keep element if it improves the score (lower for minimization)
      if score_with <= score_without:
        continue
      else:
        del candidate[i]
        break  # Stop after one element removal

    # 2. Local Search (optional, comment out if not needed)
    candidate = local_search(candidate)

    # 3. Update Best Solution
    current_score = f(candidate)
    if current_score < best_score:
      best_solution = candidate
      best_score = current_score

def iterated_greedy(N, f):
  """
  Iterated Greedy metaheuristic for finding the best permutation (N first integers)
  evaluated by function f.

  Args:
      N: Number of elements in the permutation.
      f: Function to evaluate a candidate solution (permutation).
      destroy_ratio: Ratio of elements to destroy in each iteration (default: 0.2).

  Returns:
      The best permutation found and its corresponding evaluation by f.
  """
  destroy_ratio=0.3

  # Generate initial random permutation
  current_solution = list(range(N))
  shuffle(current_solution)
  best_solution = current_solution.copy()
  best_value = f(best_solution)

  while True:
    # Destroy a portion of the current solution
    destroy_count = int(N * destroy_ratio)
    destroyed_indices = list(range(N))
    shuffle(destroyed_indices)
    for i in range(destroy_count):
      index = destroyed_indices[i]
      temp = current_solution[0]
      current_solution[0] = current_solution[index]
      current_solution[index] = temp

    # Rebuild using greedy approach (swapping adjacent elements)
    for i in range(N - 1):
      neighbor = current_solution.copy()
      neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
      neighbor_value = f(neighbor)
      if neighbor_value > best_value:
        current_solution = neighbor
        best_value = neighbor_value

    # Update best solution if found a better one
    if best_value > f(current_solution):
      best_solution = current_solution.copy()
      best_value = f(current_solution)

  return best_solution, best_value

def hill_climb(N, f):

  permutation = list(range(1, N + 1))
  shuffle(permutation)
  while True:
    for i in range(1, N):
      neighbor = permutation.copy()
      neighbor[i], neighbor[i-1] = neighbor[i-1], neighbor[i]

      neighbor_score = f(neighbor)

      if neighbor_score > f(permutation):
        permutation = neighbor.copy()
        break

def steepest_ascent_hill_climbing(N, f):
  """
  Performs steepest ascent hill climbing (for minimization) using function f.

  Args:
      N: The size of the permutation (number of elements).
      f: A function that takes a permutation (list of integers) and returns its score.
      time_limit: Optional time limit for the search (in seconds).

  Prints the best solution found so far and continues searching until timeout.
  """
  # Generate a random initial permutation
  current_state = list(range(N))
  random.shuffle(current_state)

  # Keep track of the best solution found so far
  best_state = current_state
  best_score = f(current_state)

  while True:
    # Flag to track if any improvement was found
    improved = False
    best_neighbor = None

    # Explore all neighbors (swaps)
    for i in range(N - 1):
      for j in range(i + 1, N):
        # Create a new state with a swap
        new_state = current_state.copy()
        new_state[i], new_state[j] = new_state[j], new_state[i]

        # Evaluate the neighbor
        new_score = f(new_state)

        # Update potential best neighbor if lower score found
        if new_score < best_score and (best_neighbor is None or new_score < best_neighbor_score):
          improved = True
          best_neighbor = new_state.copy()
          best_neighbor_score = new_score

    # Update current state and best state if improvement found
    if improved:
      current_state = best_neighbor
      best_state = best_neighbor.copy()
      best_score = best_neighbor_score

def chc(N, f):
  """
  Implements the CHC algorithm for permutation optimization problems.

  Args:
      N: The number of elements in the permutation.
      f: The fitness function to evaluate permutations.
      max_generations: The maximum number of generations to run the algorithm.
      population_size: The size of the population in each generation.
      restart_prob: The probability of restarting an individual in the population.

  Returns:
      The best permutation found and its fitness value.
  """
  population_size=100
  restart_prob=0.1

  # Generate initial population (ensure all permutations are unique)
  population = []
  while len(population) < population_size:
    perm = random.sample(range(N), N)
    if len(set(perm)) == N:  # Check for unique elements
      population.append(perm)

  # Evaluate fitness of each individual
  fitness = [f(p) for p in population]

  # Main loop
  while True:
    # Elitism - Select top individuals for next generation
    elite = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:population_size // 2]
    next_gen = [p for p, _ in elite]

    # Fill remaining slots with offspring
    while len(next_gen) < population_size:
      # Select parents
      parent1, parent2 = random.choices(elite, k=2)

      # Half Uniform Crossover with guaranteed unique elements
      offspring = []
      seen = set()  # Track seen elements during crossover
      for i in range(N):
        gene = random.choice([parent1[0][i], parent2[0][i]])
        while gene in seen:  # Ensure unique element in offspring
          gene = (gene + 1) % N  # Wrap around if necessary
        offspring.append(gene)
        seen.add(gene)

      # Restart with a small probability
      if random.random() < restart_prob:
        offspring = random.sample(range(N), N)

      next_gen.append(offspring)

    # Evaluate fitness of offspring
    fitness = [f(p) for p in next_gen]

    # Update population
    population = next_gen

import random

def pufferfish_optimization(N, f):
  """
  Implements the Pufferfish Optimization Algorithm for permutations with unique elements.

  Args:
      N: Number of elements in the permutation (N first integers).
      f: Function to evaluate the quality of a permutation.
          Lower values indicate better solutions.
      max_iter: Maximum number of iterations (default: 100).
      archive_size: Size of the archive to store promising solutions (default: 10).

  Returns:
      The best permutation found and its corresponding fitness value.
  """

  archive_size=100

  # Initialize population with unique permutations
  population = []
  while len(population) < archive_size:
    perm = random.sample(range(N), N)  # Sample N unique elements
    population.append(perm)

  fitness = [f(perm) for perm in population]

  # Main loop
  while True:
    for i in range(archive_size):
      # Exploration (Predator Attack)
      new_perm = population[i].copy()
      attack_pos1 = random.randint(0, N-1)
      attack_pos2 = random.randint(0, N-1)
      while attack_pos1 == attack_pos2:
        attack_pos2 = random.randint(0, N-1)
      new_perm[attack_pos1], new_perm[attack_pos2] = new_perm[attack_pos2], new_perm[attack_pos2]

      # Exploitation (Predator Escape)
      escape_prob = random.random()
      if escape_prob < 0.5:  # Escape towards better solution
        for j in range(N):
          if new_perm[j] > population[fitness.index(min(fitness))][j]:
            new_perm[j] -= 1
          elif new_perm[j] < population[fitness.index(min(fitness))][j]:
            new_perm[j] += 1
      else:  # Escape with random swap (ensure unique elements)
        swap_pos1 = random.randint(0, N-1)
        candidates = [x for x in range(N) if x not in new_perm]
        swap_pos2 = random.choice(candidates)
        new_perm[swap_pos1], new_perm[swap_pos2] = new_perm[swap_pos2], new_perm[swap_pos2]

      # Ensure length N and unique elements (avoid f() call for invalid cases)
      if len(set(new_perm)) != N:  # Check for duplicates
        continue

      # Evaluate valid permutation with f()
      new_fitness = f(new_perm)

      # Update archive
      worst_index = fitness.index(max(fitness))
      if new_fitness < fitness[worst_index]:
        population[worst_index] = new_perm
        fitness[worst_index] = new_fitness
