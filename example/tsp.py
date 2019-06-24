import numpy as np
from ga import GACfg, GA
import matplotlib.pyplot as plt


def fitness_func(gene, city_list):
    def route_distance():
        cities = []
        for i in gene:
            cities.append(city_list[i])
        cities = np.array(cities)
        _cities = np.roll(cities, 1, axis=0)
        path_distance = np.sum(np.sqrt(np.sum(
            (cities - _cities) ** 2, 1)))
        return path_distance

    return 1 / float(route_distance())


def init_city():
    city_list = np.random.randint(200, size=(25, 2))
    return city_list


def ga_p(city_list):
    cfg = GACfg(gene_len=25, gene_size=25, pop_size=1000, elite_size=50,
                tournament_num=50, tournament_size=100,
                mutation_rate=0.03, repeatable=False)
    ga = GA(fitness_func, cfg)
    ga.prepare_kwargs(city_list=city_list)
    ga.init_population()
    progress = []
    progress.append(1 / ga.cal_fitness()[0][1])
    for _ in range(100):
        early_stop = ga.evolve()
        progress.append(1 / ga.cal_fitness()[0][1])
        if early_stop:
            break

    print("Final distance: " + str(1 / ga.cal_fitness()[0][1]))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


def main():
    city_list = init_city()
    ga_p(city_list)


if __name__ == '__main__':
    main()
