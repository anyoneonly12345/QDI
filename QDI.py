import random
import time
import numpy as np

# 初始化参数
population_size = 100  # 种群大小
num_generations = 500  # 最大迭代次数
mutation_prob = 0.1    # 变异概率
crossover_prob = 0.9   # 交叉概率
T_Int = 200            # 知识共享机制的触发代数
elite_ratio = 0.25     # 精英选择比例
num_uavs = 5           # 无人机数量
num_tasks = 20         # 任务数量

# 随机生成三维任务坐标，范围为(0, 100)
tasks = np.random.rand(num_tasks, 3) * 100
# tasks = np.array([
#     [10, 20, 30],
#     [40, 50, 60],
#     [70, 80, 90],
#     [15, 25, 35],
#     [45, 55, 65],
#     [75, 85, 95],
#     [12, 22, 32],
#     [42, 52, 62],
#     [72, 82, 92],
#     [17, 27, 37],
#     [47, 57, 67],
#     [77, 87, 97],
#     [14, 24, 34],
#     [44, 54, 64],
#     [74, 84, 94],
#     [19, 29, 39],
#     [49, 59, 69],
#     [79, 89, 99],
#     [16, 26, 36],
#     [46, 56, 66]
# ])
# 初始化种群，种群按访问点总数范围分为多个子群体
def initialize_population():
    population = []
    for i in range(population_size):
        individual = generate_individual()  # 随机生成个体
        population.append(individual)
    return population

# 双部分编码生成个体，任务执行顺序和任务分配
def generate_individual():
    task_order = random.sample(range(num_tasks), num_tasks)  # 随机生成任务顺序
    task_distribution = random_distribution(num_uavs, num_tasks)  # 随机任务分配
    return (task_order, task_distribution)

# 生成随机的任务分配，分配任务给多个无人机
def random_distribution(num_uavs, num_tasks):
    distribution = [[] for _ in range(num_uavs)]
    tasks_list = list(range(num_tasks))
    random.shuffle(tasks_list)
    for i, task in enumerate(tasks_list):
        distribution[i % num_uavs].append(task)
    return distribution

# 适应度函数：计算个体的总路径长度（每个无人机的路径和）
def evaluate_fitness(individual):
    task_order, task_distribution = individual
    total_distance = 0
    for uav_tasks in task_distribution:
        total_distance += calculate_route_distance(task_order, uav_tasks)
    return total_distance

# 计算某个无人机的路径长度，基于三维欧氏距离
def calculate_route_distance(task_order, uav_tasks):
    if len(uav_tasks) < 2:
        return 0
    route_distance = 0
    for i in range(len(uav_tasks) - 1):
        start_idx = task_order[uav_tasks[i]]
        end_idx = task_order[uav_tasks[i + 1]]
        start = tasks[start_idx]
        end = tasks[end_idx]
        route_distance += np.linalg.norm(start - end)  # 计算三维欧氏距离
    return route_distance

# 任务组合交叉算子（TCX）：将父代访问点组合生成新个体
def task_combination_crossover(parent1, parent2):
    task_order1, task_distribution1 = parent1
    task_order2, task_distribution2 = parent2
    
    # 随机选择一部分任务的顺序来自父母1，剩下的顺序来自父母2
    midpoint = random.randint(0, num_tasks - 1)
    child_task_order = task_order1[:midpoint] + [task for task in task_order2 if task not in task_order1[:midpoint]]
    
    # 复制任务分配
    child_task_distribution = task_distribution1[:]
    return (child_task_order, child_task_distribution)

# 改进的随机段反转变异操作
def improved_mutation(individual):
    task_order, task_distribution = individual
    for uav_tasks in task_distribution:
        if len(uav_tasks) > 1:
            if random.random() < mutation_prob:
                # 执行概率性反转操作
                i, j = sorted(random.sample(range(len(uav_tasks)), 2))
                uav_tasks[i:j] = reversed(uav_tasks[i:j])
    return (task_order, task_distribution)

# 多种群知识共享交互机制
def multi_population_knowledge_sharing(populations, generation):
    if generation >= T_Int:
        for i, pop_i in enumerate(populations):
            top_individuals = select_top_individuals(pop_i, elite_ratio)
            for j in range(len(populations)):
                if i != j:
                    if j < i:
                        # 截取操作生成新个体
                        new_individual = intercept_operation(top_individuals, populations[j])
                    else:
                        # 插入操作生成新个体
                        new_individual = insert_operation(top_individuals, populations[j])
                    populations[j].append(new_individual)

# 选择操作，精英选择策略
def select_top_individuals(population, elite_ratio):
    sorted_population = sorted(population, key=lambda x: evaluate_fitness(x))
    elite_count = int(elite_ratio * len(population))
    return sorted_population[:elite_count]

# 截取操作
def intercept_operation(top_individuals, population):
    parent = random.choice(top_individuals)
    task_order, task_distribution = parent
    new_task_order = task_order[:]
    random.shuffle(new_task_order)
    new_distribution = random_distribution(num_uavs, num_tasks)
    return (new_task_order, new_distribution)

# 插入操作
def insert_operation(top_individuals, population):
    parent = random.choice(top_individuals)
    task_order, task_distribution = parent
    new_task_order = task_order[:]
    new_distribution = random_distribution(num_uavs, num_tasks)
    return (new_task_order, new_distribution)

# 主循环，执行遗传算法
def genetic_algorithm():
    populations = [initialize_population() for _ in range(5)]  # 初始化5个子种群
    for generation in range(num_generations):
        for i in range(len(populations)):
            new_population = []
            for _ in range(population_size // 2):
                # 选择父代，随机从精英个体中选择两个
                top_individuals = select_top_individuals(populations[i], elite_ratio)
                parent1 = random.choice(top_individuals)
                parent2 = random.choice(top_individuals)
                
                # 执行交叉
                if random.random() < crossover_prob:
                    child = task_combination_crossover(parent1, parent2)
                else:
                    child = parent1  # 保留父代
                
                # 执行变异
                child = improved_mutation(child)
                
                # 添加到新种群
                new_population.append(child)
            
            # 更新种群
            populations[i] = new_population
        
        # 执行多种群知识共享交互机制
        multi_population_knowledge_sharing(populations, generation)
    
    # 返回最终解
    return find_pareto_solutions(populations)

# 返回帕累托最优解集
def find_pareto_solutions(populations):
    all_individuals = [ind for pop in populations for ind in pop]
    all_individuals.sort(key=lambda x: evaluate_fitness(x))
    return all_individuals[:10]  # 返回适应度最好的10个解
def print_uav_tasks(solution):
    task_order, task_distribution = solution
    for uav_index, uav_tasks in enumerate(task_distribution):
        print(f"Drone {uav_index + 1} tasks:")
        for task_index in uav_tasks:
            print(f"  Task {task_index}: {tasks[task_order.index(task_index)]}")
        print()
# 执行遗传算法
start_time = time.time()
pareto_solutions = genetic_algorithm()
end_time = time.time()
# 输出最优解
for idx, solution in enumerate(pareto_solutions):
    print(f"Solution {idx + 1}:")
    print_uav_tasks(solution)
    print(f"Fitness: {evaluate_fitness(solution)}")
print(f"Total computation time: {end_time - start_time} seconds")