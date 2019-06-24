from typing import Callable
import numpy as np
import pickle


class GACfg(object):
    def __init__(self, gene_len, gene_size, pop_size, elite_size,
                 tournament_num, tournament_size,
                 mutation_rate, repeatable=True,
                 early_stop_threshold=1e-6, early_stop_step=20):
        self.gene_len = gene_len
        self.gene_size = gene_size
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.tournament_num = tournament_num
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.repeatable = repeatable
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_step = early_stop_step
        self.show_progress = False


class GA(object):
    def __init__(self, fitness_func: Callable, cfg: GACfg):
        self.fitness_func = fitness_func
        self.cfg = cfg
        self.population = np.zeros(5)
        self.last_fitness = [(0, 0)]
        self.early_stop_step = 0
        self.gene = [0, 0]
        self.kwargs = {}

    def create_individual(self):
        return np.random.choice(
            self.cfg.gene_size, self.cfg.gene_len,
            replace=self.cfg.repeatable)

    def prepare_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def _get_kwargs(self):
        return pickle.loads(pickle.dumps(self.kwargs))

    def init_population(self):
        self.population = []
        for _ in range(self.cfg.pop_size):
            self.population.append(self.create_individual())
        self.population = np.array(self.population)
        self.last_fitness = self.cal_fitness()
        self.early_stop_step = 0

    def cal_fitness(self):
        fitness_results = {}
        if self.cfg.show_progress:
            from tqdm import trange
            import sys
            r = trange(self.cfg.pop_size, file=sys.stdout)
        else:
            r = range(self.cfg.pop_size)
        for i in r:
            fitness_results[i] = self.fitness_func(
                self.population[i], **self._get_kwargs())
        return sorted(fitness_results.items(),
                      key=lambda x: x[1], reverse=True)

    def selection(self, fitness):
        selection_results = []

        # Select elites
        for i in range(self.cfg.elite_size):
            selection_results.append(fitness[i][0])

        # Tournament selection
        indexes = np.arange(self.cfg.elite_size, len(fitness))
        for i in range(self.cfg.tournament_num):
            _indexes = np.random.choice(
                indexes, self.cfg.tournament_size, replace=False)
            tournament = []
            for _i in _indexes:
                tournament.append(fitness[_i])
            index = sorted(tournament,
                           key=lambda x: x[1], reverse=True)[0][0]
            selection_results.append(index)
        return selection_results

    def mating_pool(self, selection_results):
        pool = []
        for i in selection_results:
            pool.append(self.population[i])
        return pool

    def breed(self, parent1, parent2):
        start_gene, end_gene = sorted(np.random.randint(
            self.cfg.gene_len, size=2))

        child = [-1] * self.cfg.gene_len
        child[start_gene:end_gene] = parent1[start_gene:end_gene]

        if self.cfg.repeatable:
            child[:start_gene] = parent2[:start_gene]
            child[end_gene:] = parent2[end_gene:]
        else:
            items = [item for item in parent2 if item not in child]
            for i in range(start_gene):
                child[i] = items[i]
            items = [item for item in parent2 if item not in child]
            for idx, i in enumerate(range(end_gene, len(child))):
                child[i] = items[idx]

        return child

    def breed_population(self, pool):
        children = []
        length = self.cfg.pop_size - self.cfg.elite_size

        children.extend(pool)

        indexes = np.random.choice(len(pool), (length, 2))
        for i in range(length):
            child = self.breed(pool[indexes[i, 0]], pool[indexes[i, 1]])
            children.append(child)
        return children

    def mutate(self, individual):
        indexes = np.random.rand(self.cfg.gene_len) < self.cfg.mutation_rate
        if not np.any(indexes):
            return individual
        indexes = np.argwhere(indexes)

        for index in indexes:
            swapped = index[0]
            if self.cfg.repeatable:
                individual[swapped] = np.random.randint(
                    self.cfg.gene_size)
            else:
                swap_with = np.random.randint(self.cfg.gene_len)

                gene_1 = individual[swapped]
                gene_2 = individual[swap_with]

                individual[swapped] = gene_2
                individual[swap_with] = gene_1

        return individual

    def mutate_population(self, population):
        mutated_pop = []

        for ind in range(len(population)):
            mutated_ind = self.mutate(population[ind])
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def evolve(self):
        selection_results = self.selection(self.last_fitness)
        pool = self.mating_pool(selection_results)
        children = self.breed_population(pool)
        self.population = self.mutate_population(children)
        fitness = self.cal_fitness()
        early_stop = False
        if abs(fitness[0][1] - self.last_fitness[0][1]) < \
                self.cfg.early_stop_threshold:
            self.early_stop_step += 1
            if self.early_stop_step >= self.cfg.early_stop_step:
                early_stop = True
        else:
            self.early_stop_step = 0
        self.last_fitness = fitness
        self.gene = self.population[fitness[0][0]]
        return early_stop
