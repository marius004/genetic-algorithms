from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np
import copy

@dataclass
class QuadraticEquation: 
    a: float
    b: float
    c: float
    
    def eval(self, x: float) -> float: 
        return self.a * x**2 + self.b * x + self.c

class Chromosome:
    def __init__(self,
                 str: str,
                 length: int,
                 fitness_function: Callable[[float], float],
                 domain: List[float]
                 ) -> None:
        self.fitness_function = fitness_function
        self.length = length
        self.domain = domain
        self.str = str
        
        self.nr  = self.__number_repr(str, domain, length)
    
    def fitness(self):
        return self.fitness_function(self.nr)
    
    def crossover(self, other: 'Chromosome') -> Tuple['Chromosome', 'Chromosome', int]: 
        breaking_point = np.random.randint(0, self.length)
        child1 = self.str[:breaking_point] + other.str[breaking_point:]
        child2 = other.str[:breaking_point] + self.str[breaking_point:]
        
        return (
            Chromosome(child1, self.length, self.fitness_function, self.domain),
            Chromosome(child2, self.length, self.fitness_function, self.domain),
            breaking_point
        )
        
    def mutate(self) -> 'Chromosome':
        index = np.random.randint(0, self.length)
        mutation = self.str[:index] + ("1" if self.str[index] == "0" else "0") + self.str[index+1:]
        return Chromosome(mutation, self.length, self.fitness_function, self.domain)
    
    def __str__(self) -> str:
        return f"{self.str} x={self.nr} f={self.fitness_function(self.nr)}"
    
    def __repr__(self) -> str:
        return str(self)
    
    @classmethod 
    def from_number(cls: 'Chromosome', 
                    nr: float, 
                    length: int, 
                    fitness_function: Callable[[float], float],
                    domain: List[float]
                    ) -> 'Chromosome':
        return cls(Chromosome.__binary_repr(nr, domain, length), length, fitness_function, domain)
     
    @staticmethod
    def __binary_repr(nr: float, domain: List[float], length: int) -> str:
        d = (domain[1] - domain[0]) / (1 << length)
        
        low, high = 0, 1 << length
        while low <= high:
            mid = (low + high) // 2
            if nr >= domain[0] + mid * d and nr <= domain[0] + (mid + 1) * d:
                return "".join(["1" if mid & (1 << i) else "0" for i in range(length - 1, -1, -1)])
            elif nr < domain[0] + mid * d:
                high = mid - 1
            else:
                low = mid + 1            

        return "1" * length
    
    @staticmethod
    def __number_repr(str: str, domain: List[float], length: int) -> float: 
        d = (domain[1] - domain[0]) / (1 << length)
        return domain[0] + d * sum((1 << i) for i, bit in enumerate(str[::-1]) if bit == '1')
   
class Population: 
    def __init__(self, 
                 size: int,
                 domain: List[float],
                 fitness_function: Callable[[float], float], 
                 crossover_probability: float,
                 mutation_probability: float,
                 population: List[Chromosome],
                 ) -> None:
        self.size = size
        self.domain = domain
        self.population = population
        self.fitness_function = fitness_function
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        
    @classmethod 
    def from_old_population(cls: 'Population', old_population: 'Population', new_population):
        return cls(size=old_population.size, 
                   domain=old_population.domain,
                   fitness_function=old_population.fitness_function,
                   crossover_probability=old_population.crossover_probability,
                   mutation_probability=old_population.mutation_probability,
                   population=new_population
                )
    
    def select_chromosome(self, 
                        cumulative_probabilities: List[float],
                        probability: float) -> Tuple[Chromosome, int]:
        low, high = 0, len(cumulative_probabilities) - 1
        while low <= high:
            mid = (low + high) // 2
            if probability >= cumulative_probabilities[mid] and \
                probability <= cumulative_probabilities[mid + 1]:
                return self.population[mid], mid
            if probability < cumulative_probabilities[mid]: 
                high = mid - 1
            else: 
                low = mid + 1
        
        raise ValueError("Binary search failed: Chromosome not found")
    
    def selection(self, is_first_generation: bool, fd=None) -> List[Chromosome]:
        population_fitness = [chromosome.fitness() for chromosome in self.population]
        total_fitness = sum(population_fitness) 
        
        probabilities = [fitness / total_fitness for fitness in population_fitness]
        cumulative_probabilities = np.cumsum([0] + probabilities)
        
        choices, selected = np.random.uniform(0, 1, self.size), [(None, None)] * self.size 
        
        for i, choice in enumerate(choices):
            selected[i] = copy.deepcopy(self.select_chromosome(cumulative_probabilities, choice))
        
        # this can be further restructured
        if is_first_generation:
            fd.write("Probabilitati selectie:\n")
            for i, probability in enumerate(probabilities):
                fd.write(f"\tcromozom {i+1} probabilitate {probability}\n")
            
            fd.write("\nIntervale probabilitati selectie:\n")
            fd.write(" ".join(map(str, cumulative_probabilities.tolist())))
            fd.write("\n\n")
            
            for choice, chromosome in zip(choices, selected):
                fd.write(f"u={choice} selectam cromozomul {chromosome[-1] + 1} \n")    

            fd.write("\nDupa selectie:\n")
            fd.write(str(Population.from_old_population(self, list(map(lambda key: key[0], selected)))))
            fd.write("\n\n")
        
        return list(map(lambda key: key[0], selected)) 
    
    def crossover(self, population: List[Chromosome], is_first_generation=False, fd=None) -> List[Chromosome]:
        crossover_probabilities = np.random.uniform(0, 1, len(population))
        
        crossover_participants = []
        for i, probability in enumerate(crossover_probabilities): 
            if probability < self.crossover_probability:
                crossover_participants.append(i)
            
        np.random.shuffle(crossover_participants)
        if len(crossover_participants) % 2 == 1: 
            crossover_probabilities[crossover_participants[-1]] = self.crossover_probability + np.finfo(float).eps 
            crossover_participants.pop()
        
        to_delete, to_append = set(), []
        chromosomes_used = []
        
        for i in range(0, len(crossover_participants), 2): 
            idx1, idx2 = crossover_participants[i], crossover_participants[i + 1]
            child1, child2, point = population[idx1].crossover(population[idx2])
            
            to_append.append(child1)
            to_append.append(child2)
            to_delete.add(idx1)
            to_delete.add(idx2)
            
            chromosomes_used.append((idx1, idx2, point))
        
        new_population = [chromosome for i, chromosome in enumerate(population) if i not in to_delete] + to_append     
        
        # this can be further restructured
        if is_first_generation:
            fd.write(f"Probabilitatea de incrucisare {self.crossover_probability}\n")
            for i, (chromosome, probability) in enumerate(zip(population, crossover_probabilities)): 
                fd.write(f"{i+1}:\t{chromosome.str} u={probability}")
                if probability < self.crossover_probability:
                    fd.write(f"<{self.crossover_probability} participa")
                fd.write("\n")
            fd.write("\n\n") 
            
            for indexes in chromosomes_used:
                fd.write(f"Recomobinare dintre cromozomul {indexes[0]+1} cu cromozomul {indexes[1]+1}:\n")
                fd.write(f"{population[indexes[0]].str} {population[indexes[1]].str} punct {indexes[2]}\n")

            fd.write("Dupa recombinare:\n")
            fd.write(str(Population.from_old_population(self, new_population)))
            fd.write("\n\n")
        
        return new_population
    
    def mutation(self, population: List[Chromosome], is_first_gen:bool=False, fd=None) -> List[Chromosome]: 
        new_population = copy.deepcopy(population)
        
        to_mutate = []
        for i, probability in enumerate(np.random.uniform(0, 1, len(new_population))):
            if probability < self.mutation_probability: 
                new_population[i] = new_population[i].mutate()
                to_mutate.append(i)
        
        # this can be further restructured
        if is_first_gen:
            fd.write(f"Probabilitate de mutatie pentru fiecare gena {self.mutation_probability}\n")
            fd.write(f"Au fost modificati cromozomii: {to_mutate}\n")
            
            fd.write(f"Dupa mutatie:\n")
            fd.write(str(Population.from_old_population(self, new_population)))
            fd.write("\n\n")
        
        return new_population
    
    # this can be further restructured
    def keep_best_chromosomes(self, next_population: List[Chromosome]) -> List[Chromosome]:
        keep_count = max(1, int(0.05 * len(self.population))) 
        sorted_current_population = sorted(self.population, key=lambda x: x.fitness(), reverse=True)
        best_chromosomes = sorted_current_population[:keep_count]
       
        population_fitness = [chromosome.fitness() for chromosome in next_population]
        total_fitness = sum(population_fitness) 
        
        probabilities = [fitness / total_fitness for fitness in population_fitness]
        cumulative_probabilities = np.cumsum([0] + probabilities)
       
        choices = np.random.uniform(0, 1, keep_count) 
        new_population = Population.from_old_population(self, copy.deepcopy(next_population))
        
        for i, probability in enumerate(choices):
            _, chromosome_index = new_population.select_chromosome(cumulative_probabilities, probability)
            next_population[chromosome_index] = best_chromosomes[i]    
        
        return next_population    
    
    def next_generation(self, is_first_gen:bool=False, fd=None) -> 'Population':
        population = self.selection(is_first_gen, fd)
        population = self.crossover(population, is_first_gen, fd)
        population = self.mutation(population, is_first_gen, fd)
        
        return Population.from_old_population(self, self.keep_best_chromosomes(population)) 
    
    def evolution(self) -> float:
        return max(chromosome.fitness() for chromosome in self.population) 
    
    def __str__(self):
        return "\n".join([f"\t{idx + 1}: " + str(chromosome) for idx, chromosome in enumerate(self.population)])
    
    def __repr__(self):
        return str(self)
 
    @classmethod
    def initial_population(cls: 'Chromosome',
                           size: int,
                           domain: List[float],
                           length: int,
                           fitness_function: Callable[[int], int], 
                           crossover_probability: float,
                           mutation_probability: float
                        ) -> 'Population':
        return Population(size=size, 
                          domain=domain,
                          fitness_function=fitness_function,
                          crossover_probability=crossover_probability,
                          mutation_probability=mutation_probability,
                          population=Population.__default_population(domain, fitness_function, length, size))
    
    @staticmethod
    def __default_population(domain: List[float],
                            fitness_function: Callable[[int], int], 
                            length: int,
                            size: int) -> List[Chromosome]:
        chromosomes = []
        for _ in range(size): 
            chromosomes.append(Chromosome.from_number(
                nr=np.random.uniform(domain[0], domain[1]),
                fitness_function=fitness_function,
                length=length,
                domain=domain
            ))
        
        return chromosomes

def chromosome_length(domain: List[float], precision: int) -> int:
    nr = int((domain[1] - domain[0]) * 10 ** precision)
    return int(np.floor(np.log2(nr)) + 1 if nr & (nr - 1) else np.floor(np.log2(nr)))

def main(): 
    with open("input.txt", "r") as in_file:
        population_size = int(in_file.readline().strip())
        domain = list(map(float, in_file.readline().strip().split()))
        a, b, c = list(map(float, in_file.readline().strip().split()))
        precision = float(in_file.readline().strip())
        crossover_probability = float(in_file.readline().strip())
        mutation_probability = float(in_file.readline().strip())
        iterations = int(in_file.readline().strip())
        
    equation = QuadraticEquation(a, b, c)
    population = Population.initial_population(
                    size=population_size,
                    domain=domain,
                    length=chromosome_length(domain, precision),
                    fitness_function=equation.eval,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability                    
                )
    
    with open("output.txt", "w") as out_file:
        out_file.write("Populatia initiala\n")
        out_file.write(str(population) + "\n\n")
        
        for iteration in range(2, iterations + 1):
            population = population.next_generation(True if iteration == 2 else False, out_file)      
            out_file.write(f"{population.evolution()}\n")
            
if __name__ == "__main__":
    main()