import random
import pickle
import numpy as np
import pandas as pd
import pygad
from pyeasyga import pyeasyga
from pyeasyga.pyeasyga import GeneticAlgorithm
import matplotlib.pyplot as plt

# TODO
# random_user selection -> tick
# save the movie indexes -> tick
# calculate the pearson of each user -> tick
# calculate the neighbors of the randomly selected user -> tick
# repair procedure -> tick
# random chromosome -> tick
# crossover -> tick
# mutation -> tick
# fitness -> tick
# initialize population -> tick
np.seterr(divide='ignore', invalid='ignore') # corcoef strange error

df = pd.read_csv('u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['userid', 'movie', 'rating'] # rename columns
movies = df['movie'].unique() # find unique movies
users = df['userid'].unique() # find unique users

sorted_movies = np.copy(movies) # create sorted movies array
sorted_movies = np.sort(sorted_movies) # sort the movies by id
position = {}
for index, movie in enumerate(sorted_movies):
    position.update({movie: index}) # get the index of the odered movies and save it as a dictionary {movie: index}

mean = df.groupby(['userid'])
means = {}
# calculate the mean of each user and store it on means[]
for usr in users:
    temp_mean = mean.get_group(usr)['rating'].mean()
    means[usr] = temp_mean  # means is indexed by user id


def select_random_user(data):  # data is a dataframe
    random_user = data.sample(1).iloc[0]['userid']  # get one random user
    random_user_ratings = data.loc[df['userid'] == random_user][['movie', 'rating']].reset_index(drop=True)
    # returns all the ratings of th random user
    return random_user, random_user_ratings

fitness_list = list()
# check corcoef value range
def fitness_function(chromosome, solution_index):  # get the chromosome
    top10 = most_similar  # retrieve to top users from the other function
    top_users = list()
    for i in range(10):  # get only the top 10
        top_users.append(top10[i][0])
    r = list()

    most_fit = None
    for i in range(10):  # calculate the pearson helper for each other user
        neighbor = prods[i]
        prod = (np.corrcoef(chromosome, neighbor)[0, 1])  # and now calculate the corrcoef
        r.append(prod)
    fit = np.mean(r)
    if most_fit is None:  # first iteration
        most_fit = fit
    elif fit >= most_fit:
        most_fit = fit
    elif fit < most_fit:
        most_fit = most_fit
    fitness_list.append(most_fit)
    return most_fit  # return the fitness value

# add MEAN values to the missing ratings
def calculate_pearson_helper(rand_user_id):
    usr_mean = means[rand_user_id] # get the mean of the user
    random_user_ratings = df[df['userid'] == rand_user_id] # get the random user ratings
    pearson_cor = np.full(len(movies), usr_mean) # create an array full of the user means
    for _col, row in random_user_ratings.iterrows():  # for each real rating of the user place it
        movie = row.movie                             # in the array to correct position
        rating = row.rating
        pearson_cor[position[movie]] = rating
    return pearson_cor


# add RANDOM values to the missing ratings
def calculate_pearson_helper2(data, rand_user_id): # same logic as before but uses random values
    chromosome = random_chromosome(movies)
    user_ratings = data[data['userid'] == rand_user_id]
    for _col, row in user_ratings.iterrows():
        movie = row.movie
        rating = row.rating
        chromosome[position[movie]] = rating
    return chromosome


# find the neighborhood
def calculate_correlation(rand_userid, users):  # function to calculate the corrcoef and finally the most similar users
    global most_similar, prods
    saved_correlations = list()
    matches = list()
    pearson_cor = calculate_pearson_helper(rand_userid)
    # remove the randomly selected user
    neighbour_users = np.delete(users, np.where(users == rand_userid))
    # calculate each user pearson
    prods = list()
    for usr in neighbour_users:
        pearson_prod = calculate_pearson_helper(usr)
        # save pearson to list
        saved_correlations.append((usr, pearson_prod))
        prods.append(pearson_prod)
    # for each user's pearson calculate the correlation
    for item in saved_correlations:
        matched_user = item[0]
        other_pearson_cor = item[1]
        coefficient_value = np.corrcoef(pearson_cor, other_pearson_cor)[0, 1]
        matches.append((matched_user, coefficient_value))
        # order the matches list
        matches.sort(key=lambda matches: matches[1], reverse=True)
        # get the top 10 matches
        most_similar = matches[:10]
    return most_similar


def generate_mutation(offspring_crossover): # mut_prob is the probability of mutation
    mut_prob = 0
    roll = random.randrange(1, 100)
    if roll <= mut_prob:
        index = random.randrange(len(offspring_crossover))
        mutated_rating = random.randrange(1, 5)
        offspring_crossover[index] = offspring_crossover[index] - mutated_rating
        if offspring_crossover[index] < 0:
            offspring_crossover[index] = offspring_crossover[index] * (-1)
        return repair_procedure(user_rating, offspring_crossover)
    else:
        return offspring_crossover


def perform_crossover(m, f):  # m is mother and f is father
    cross_prob = 60 # probability of crossover
    roll = random.randrange(1, 100)
    if roll <= cross_prob:
        index = random.randrange(1, len(m))
        ch1 = np.concatenate([m[:index], f[index:]])  # perform crossover for child 1
        ch2 = np.concatenate([f[:index], m[index:]])  # perform crossover for child 2
        return ch1, ch2  # ch1, ch2 are the children
    else:
        return m, f


def random_chromosome(movies):  # generate random chromosome then repair them, with the correct ratings
    x = np.random.randint(low=1, high=5, size=len(movies))
    y = repair_procedure(user_rating, x)
    return y


def initialize_population(size): # calls n-times the random chromosome function
    population = list()
    for j in range(size):
        population.append(random_chromosome(movies))
    np.array(population)
    return population


# add the pre rated movies to the random chromosomes
def repair_procedure(user_rating, chromosome): # fixes the chromosomes
    for col, row in user_rating.iterrows():
        movie = row.movie
        rating = row.rating
        chromosome[position[movie]] = rating
    return chromosome


last_fitness = 0


def callback_generation(ga_instance): # helper functions for the second library to print out the values of each generation
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1] * 100))
    print("Change     = {change}".format(change=(ga_instance.best_solution()[1] - last_fitness) * 100))
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # if (ga_instance.best_solution()[1] - last_fitness) * 100 < 0.1:
    #     print("Terminated because it doesn't progress.")
    #     ga.plot_result()
    #     quit()
    if ga_instance.generations_completed > 201:
        print("Too many generations.")
        ga.plot_result()
        quit()
    last_fitness = solution_fitness


print("Creating Population")
global user_rating
random_user, random_user_ratings = select_random_user(df) # get the random user
user_rating = random_user_ratings # save to global var the ratings

population = initialize_population(2) # initialize population of n-chromosomes
fixed_population = list()
for chromosome in population:
    fixed_population.append(repair_procedure(random_user_ratings, chromosome)) # fix the chromosomes
print("Calculating Pearson")
calculate_pearson_helper2(df, random_user)  # user first or second method to fill the values
print("Calculating Correlation")
x = calculate_correlation(random_user, users)  # corrcoef calculations
print(most_similar)
for i in range(10):
    prods[i] = repair_procedure(user_rating, prods[i])
print("Training")

# BEGIN TRAINING
# --------- First Library --------- #
# uncomment to run one of each libraries
initial = random_chromosome(movies)
generations = 40
ga = pyeasyga.GeneticAlgorithm(initial,
                               population_size=20,
                               generations=generations,
                               maximise_fitness=True,
                               elitism=True)

ga.create_individual = random_chromosome
ga.fitness_function = fitness_function
ga.crossover_function = perform_crossover
ga.mutate_function = generate_mutation
ga.run()

print(ga.best_individual())
counter = 0
fit_list = list()
print(len(fitness_list))
for i in range(0, len(fitness_list), generations):
    fit_list.append(sum(fitness_list[i: i+10])/10)
    counter = counter + 1
plt.plot(fit_list)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.show()
best_match = ga.best_individual()[1]
best_match = repair_procedure(random_user_ratings, best_match)
print(fitness_function(best_match, 1))


# --------- Second Library --------- #
# uncomment to run one of each libraries
#
# ga = pygad.GA(num_generations=1000,
#               num_parents_mating=2,
#               initial_population=fixed_population,
#               init_range_low=1,
#               init_range_high=5,
#               fitness_func=fitness_function,
#               parent_selection_type="sss",
#               crossover_type="single_point",
#               mutation_type="swap",
#               mutation_percent_genes=1,
#               callback_generation=callback_generation)
# ga.run()
# ga.plot_result()

