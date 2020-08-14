def find_pareto(utilities):
    num_dim = len(utilities)
    num_solutions = len(utilities[0])
    solutions = []
    
    for i in range(num_solutions):
        is_dominated = False
        for j in range(num_solutions):
            dominated = [utilities[d][i] < utilities[d][j] for d in range(num_dim)]
            is_dominated = is_dominated or all(dominated)
        
        if not is_dominated:
            solutions.append(i)
            
    return solutions