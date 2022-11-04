from typing import NamedTuple

import numpy as np

import Reporter

rng = np.random.default_rng(seed=123)


class Individual(NamedTuple):
    tour: list[int]
    cost: float


class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def initial_population(self, distanceMatrix, n: int) -> list[Individual]:
        problem_size = distanceMatrix.shape[0]
        individuals = []
        for i in range(n):
            tour = np.random.default_rng().permutation(problem_size)
            cost = self.cost(tour, distanceMatrix)
            new_individual = Individual(tour=tour, cost=cost)
            individuals.append(new_individual)
        return individuals

    def cost(self, tour: list[int], distanceMatrix) -> float:
        return np.sum(distanceMatrix[tour, np.roll(tour, -1)])

    def parent_selection(
            self, population: list[Individual], n: int
    ) -> list[Individual]:
        """Random parent selection with size N"""
        return rng.choice(population, n)

    def non_wrapping_ordered_crossover(parent1, parent2):
        cp1 = np.random.randint(len(parent1) - 1)
        cp2 = np.random.randint(cp1 + 1, len(parent1))
        child1 = [n for n in parent1 if n not in parent2[cp1:cp2]]
        child2 = [n for n in parent2 if n not in parent1[cp1:cp2]]
        child1 = child1[:cp1] + parent2[cp1:cp2] + child1[cp1:]
        child2 = child2[:cp1] + parent1[cp1:cp2] + child2[cp1:]
        return [child1, child2]

    def crossover(parents):
        # Morph
        # choose even number parents
        l = len(parents)
        pair_list = list(zip(parents[::2], parents[1::2]))
        children = []
        for i in range(len(pair_list)):
            children.extend(non_wrapping_ordered_crossover(pair_list[i][0], pair_list[i][1]))
        return children

    def calculate_tour_cost(self, distanceMatrix, tour) -> float:
        cost = 0
        for city_idx in range(-1, len(tour)):
            next_city_idx = (city_idx+1) % len(tour)
            cost += distanceMatrix[city_idx, next_city_idx]
        return cost

    def mutate(
            self, distanceMatrix, offspring: list[Individual]
    ) -> list[Individual]:
        """Mutation using edge"""

        mutation_rate = 0.3

        new_offspring = []
        for o in offspring:
            if rng.random() > mutation_rate:
                new_offspring.append(o)
                continue

            a, b = rng.choice(len(o.tour), 2, replace=False)
            tour = np.array(o.tour)
            tour[b], tour[a] = tour[a], tour[b]

            tour_cost = self.cost(distanceMatrix, tour)
            new_offspring.append(Individual(tour, tour_cost))
        return new_offspring

    def selection(
            self,
            population: list[Individual],
            offspring: list[Individual],
            n: int,
            k: int
    ) -> list[Individual]:
        # k-tournament
        pool = population + offspring
        new_population = []
        for i in range(n):
            idx_sampled = rng.choice(len(pool), k, replace=False)  # k-tournament
            pool_subset = sorted(pool[idx_sampled], key=lambda x: x.cost)
            new_population.append(pool_subset[0])
        return new_population

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        population = self.initial_population(distanceMatrix, n=50)
        print('Pop intialized!')

        while True:
            # Your code here.
            # parents is a list of cycles, costs is a list of real numbers
            parents = self.parent_selection(population, n=50)
            offspring = self.crossover(parents, n=50, k=3)
            new_offspring = self.mutate(distanceMatrix, offspring)
            population = self.selection(population, new_offspring, n=50)

            meanObjective = np.mean([ individual.cost for individual in population])
            bestIdx = np.argmin([ individual.cost for individual in population])

            bestObjective = population[bestIdx].cost
            bestSolution = population[bestIdx].tour


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(
                meanObjective, bestObjective, bestSolution)
            print(f'We have {timeLeft} seconds left! :o')
            if timeLeft < 0:
                break

        # Your code here.
        return 0


if __name__ == '__main__':
    solver = r0123456()
    solver.optimize('tour50.csv')