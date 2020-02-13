import os, sys, random, time
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import logging
import networkx as nx
import numpy as np
from collections import defaultdict

class CitySimple:
    def __init__(self, world_graph, wc, random_average, car_distribution, dist, A, n_intervals):
        self.log = {}
        self.n_intervals = n_intervals
        self.dist = dist
        self.A = A
        self.world = world_graph
        self.random_average = random_average
        self.init_car_distribution = car_distribution
        self.number_of_cars = len(car_distribution)
        self.wc = wc
        self.RANDOM_SEED = 0
        self.target_locations = np.zeros((self.number_of_cars, len(self.world)))
        self.init()

    def run(self, iterations = 10):
        min_revenues = []
        total_revenues = []
        for i in range(iterations):
            self.init()
            for j in range(self.n_intervals):
                self.step()
            min_revenues.append(np.min(self.all_revenues))
            total_revenues.append(np.sum(self.total_revenue))

        self.avg_total = np.mean(total_revenues)
        self.avg_min = np.mean(min_revenues)

    def init(self):
        self.total_revenue = []
        self.all_revenues = np.zeros(self.number_of_cars)
        self.car_distribution = np.copy(self.init_car_distribution)

    def step(self):
        revenue = 0
        considered_cars = set() #debugging

        customers = np.random.poisson(self.random_average)
        customers_per_cell = {}
        cars_per_cell = {}
        new_car_distribution = np.zeros(self.number_of_cars)
        for source_cell, target_distr in enumerate(customers):
            customers_per_cell[source_cell] = []
            cars_per_cell[source_cell] = []
            for target_cell, number_of_customers in enumerate(target_distr):
                customers_per_cell[source_cell] += [target_cell]*number_of_customers
        for car_id, cell_id in enumerate(self.init_car_distribution):
            cars_per_cell[cell_id].append(car_id)

        for i in range(len(self.world)):
            cars, customers = len(cars_per_cell[i]), len(customers_per_cell[i])
            if cars == 0:
                continue

            if cars < customers:
                selected_customers = np.random.choice(customers_per_cell[i], cars, replace=False)
                selected_cars = set(cars_per_cell[i])
                for car_id, location in enumerate(selected_customers):
                    car = cars_per_cell[i][car_id]
                    self.target_locations[car][location] += 1
                    new_car_distribution[car] = location
                    revenue += self.dist[i][location]
                    self.all_revenues[car] += self.dist[i][location]

                    assert(car not in considered_cars)
                    considered_cars.add(car)

            if cars >= customers:
                selected_cars = np.random.choice(cars_per_cell[i], customers, replace=False)
                for customer_id, car in enumerate(selected_cars):
                    target = customers_per_cell[i][customer_id]
                    self.target_locations[car][target] += 1
                    new_car_distribution[car] = target
                    revenue += self.dist[i][target]
                    self.all_revenues[car] += self.dist[i][target]

                    assert(car not in considered_cars)
                    considered_cars.add(car)

            # the rest distribute according to A
            selected_cars = set(selected_cars)
            for car in cars_per_cell[i]:
                if car not in selected_cars:
                    opt = list(self.world.neighbors(i)) + [i]
                    target = np.random.choice(opt, p=self.A[i][opt])
                    self.target_locations[car][target] += 1
                    new_car_distribution[car] = target
                    revenue -= self.wc
                    self.all_revenues[car] -= self.wc

                    assert(car not in considered_cars)
                    considered_cars.add(car)

        self.car_distribution = new_car_distribution
        self.total_revenue.append(revenue)

    @staticmethod
    def build_car_distribution(idle_driver_locations):
        car_distribution = []
        for i in range(len(idle_driver_locations[0])):
            for j in range(int(idle_driver_locations[0,i])):
                car_distribution.append(i)
        return car_distribution

    def get_min_revenue(self):
        return self.avg_min

    def get_total_revenue(self):
        return self.avg_total
