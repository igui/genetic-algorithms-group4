from typing import NamedTuple

import numpy as np

import Reporter


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
        result = 0
        for city_idx in range(len(tour)):
            next_city_idx = (city_idx + 1) % len(tour)
            edge = tour[city_idx], tour[next_city_idx]
            edge_cost = distanceMatrix[edge]
            result += edge_cost
        return result

    def parent_selection(self, population: list[Individual], n: int) -> list[Individual]:
        # Ignacio
        pass

    def crossover(self, parents: list[Individual], n: int) -> list[Individual]:
        # Morph
        pass

    def mutate(self, offspring: list[Individual]) -> list[Individual]:
        # Ignacio
        pass

    def selection(
        self,
        population: list[Individual],
        offspring: list[Individual],
        n: int
        ) -> list[Individual]:
        # Wai Chung
        pass


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
            offspring = self.crossover(parents, n=50)
            new_offspring = self.mutate(offspring)
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
