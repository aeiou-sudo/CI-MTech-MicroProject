import numpy as np
import matplotlib.pyplot as plt
import re

# Logistic map for chaotic noise
def logistic_map(x, r=4):
    return r * x * (1 - x)

# Fitness function: Sum of squared errors for nonlinear equations
def fitness_function(y, equations):
    total_error = 0
    for eq in equations:
        total_error += eq(y)**2  # Square of each equation's error
    return total_error

# Validate and parse new equations entered by the user
def parse_equations(equation_str):
    try:
        # Split equations by ','
        eq_list = equation_str.split(',')
        equations = []

        # Parse each equation
        for eq in eq_list:
            # Example regex for simple equation parsing: 'cos(2*y[0]) - cos(2*y[1]) - 0.4'
            eq = eq.strip()
            if not re.match(r"^[\w\s\(\)\+\-\*\^\/\.\,]*$", eq):
                raise ValueError("Invalid characters in equation.")
            equations.append(lambda y, eq=eq: eval(eq))  # Convert to function
        return equations
    except Exception as e:
        print(f"Error parsing equations: {e}")
        return None

# Initialize particles for PSO
def initialize_particles(num_particles, dimensions, bounds):
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
    velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
    return positions, velocities

# Apply chaotic noise when needed
def apply_chaotic_noise(positions, logistic_seed):
    chaotic_values = logistic_map(logistic_seed)
    print("Noise is:", chaotic_values)
    new_positions = positions * chaotic_values
    return new_positions, chaotic_values

# CN-BPSOA Algorithm
def CN_BPSOA(num_particles, dimensions, bounds, max_iter, equations, stagnation_limit=5):
    # Initialize positions and velocities
    positions, velocities = initialize_particles(num_particles, dimensions, bounds)
    
    # Personal and global bests
    pbest_positions = positions.copy()
    pbest_scores = np.array([fitness_function(p, equations) for p in positions])
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index]
    gbest_score = pbest_scores[gbest_index]
    
    inertia_weight = 0.7
    cognitive_param, social_param = 1.5, 1.5
    logistic_seed = 0.5  # Initial seed for chaotic noise
    stagnation_counter = 0

    # Store best scores for convergence visualization
    best_scores = []

    # Start iterations
    for t in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (inertia_weight * velocities[i] +
                            cognitive_param * r1 * (pbest_positions[i] - positions[i]) +
                            social_param * r2 * (gbest_position - positions[i]))
            
            # Update position
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])  # Enforce bounds

            # Evaluate fitness
            score = fitness_function(positions[i], equations)
            print("Position and Score: ")
            print(score)
            # Update personal and global bests
            if score < pbest_scores[i]:
                pbest_positions[i] = positions[i]
                pbest_scores[i] = score
            if score < gbest_score:
                gbest_position = positions[i]
                gbest_score = score
                stagnation_counter = 0  # Reset stagnation counter

        # Chaotic noise phase if stagnation occurs
        stagnation_counter += 1
        if stagnation_counter >= stagnation_limit:
            print("Limit reached\n",score)

            positions, logistic_seed = apply_chaotic_noise(positions, logistic_seed)
            stagnation_counter = 0  # Reset stagnation counter

        # Store current best score
        best_scores.append(gbest_score)
        print(f"Iteration {t+1}, Best Fitness: {gbest_score:.6f}")

        # Termination condition
        if gbest_score < 1e-5:  # Fitness threshold
            break
    
    # Plot convergence curve
    plt.plot(best_scores)
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve of CN-BPSOA')
    plt.show()

    return gbest_position, gbest_score

# Menu Interface for user choices
def menu():
    print("\n--- CN-BPSOA Micro Project ---")
    print("1. Run with default parameters")
    print("2. Change parameters")
    print("3. Input new equations")
    print("4. Exit")

    choice = input("Enter your choice: ")
    return choice

def main():
    num_particles = 10  # Default number of particles
    dimensions = 2  # Default number of variables (y1, y2)
    bounds = [-10, 10]  # Default search space bounds
    max_iter = 25  # Default maximum iterations

    # Default equations
    default_equations = [
        lambda y: np.cos(2*y[0]) - np.cos(2*y[1]) - 0.4,
        lambda y: 2 * (y[1] - y[0]) + np.sin(2*y[1]) - np.sin(2*y[0]) - 1.2
    ]

    while True:
        choice = menu()
        if choice == '1':
            best_solution, best_fitness = CN_BPSOA(num_particles, dimensions, bounds, max_iter, default_equations)
            print("Best Solution Found:", best_solution)
            print("Best Fitness Achieved:", best_fitness)
        elif choice == '2':
            # Change parameters
            num_particles = int(input("Enter number of particles: "))
            dimensions = int(input("Enter number of dimensions (variables): "))
            bounds = list(map(float, input("Enter bounds (min max): ").split()))
            max_iter = int(input("Enter maximum iterations: "))
        elif choice == '3':
            # Input new equations
            equation_str = input("Enter new equations separated by commas: ")
            new_equations = parse_equations(equation_str)
            if new_equations:
                default_equations = new_equations
                print("Equations updated successfully.")
            else:
                print("Invalid equations entered.")
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
