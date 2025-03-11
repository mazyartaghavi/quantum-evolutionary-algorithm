pip install pennylane tensorflow matplotlib scikit-learn
# Import necessary libraries
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Select a Machine Learning Model
def create_model(learning_rate=0.01, num_units=128):
    """Create a simple deep learning model for MNIST classification."""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(num_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 2: Load and Preprocess the Dataset
def load_data():
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
    return x_train, y_train, x_test, y_test

# Step 3: Quantum Evolutionary Algorithm Design
def quantum_crossover(parent1, parent2):
    """Quantum-inspired crossover operator."""
    # Use quantum superposition to combine parent genes
    child = (parent1 + parent2) / np.sqrt(2)
    return child

def quantum_mutation(individual, mutation_rate=0.1):
    """Quantum-inspired mutation operator."""
    # Apply random mutations based on mutation_rate
    mask = np.random.rand(*individual.shape) < mutation_rate
    individual[mask] = np.random.rand(*individual.shape)[mask]
    return individual

def evaluate_fitness(individual, x_train, y_train, x_val, y_val):
    """Evaluate the fitness of an individual (hyperparameters)."""
    learning_rate, num_units = individual
    model = create_model(learning_rate=learning_rate, num_units=num_units)
    model.fit(x_train, y_train, epochs=1, verbose=0)
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy

def quantum_evolutionary_algorithm(population_size=10, generations=5, mutation_rate=0.1):
    """Quantum Evolutionary Algorithm for hyperparameter tuning."""
    # Load dataset
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # Initialize population
    population = np.random.rand(population_size, 2)  # [learning_rate, num_units]
    population[:, 0] = population[:, 0] * 0.1  # Scale learning_rate to [0, 0.1]
    population[:, 1] = (population[:, 1] * 128).astype(int)  # Scale num_units to [1, 128]

    # Track best fitness and convergence
    best_fitness = []
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        fitness_scores = [evaluate_fitness(individual, x_train, y_train, x_val, y_val) for individual in population]
        best_fitness.append(max(fitness_scores))

        # Select parents (top 50%)
        parents = population[np.argsort(fitness_scores)[-population_size // 2:]]

        # Generate offspring using quantum crossover and mutation
        offspring = []
        for i in range(population_size):
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            child = quantum_crossover(parent1, parent2)
            child = quantum_mutation(child, mutation_rate)
            offspring.append(child)
        population = np.array(offspring)

    return best_fitness

# Step 4: Performance Metrics and Comparison
def run_experiment():
    """Run the QEA and compare with a classical EA."""
    # Run Quantum Evolutionary Algorithm
    qea_fitness = quantum_evolutionary_algorithm(population_size=10, generations=5, mutation_rate=0.1)

    # Plot results
    plt.plot(qea_fitness, label="Quantum EA")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Accuracy)")
    plt.title("Hyperparameter Tuning Performance")
    plt.legend()
    plt.show()

# Run the experiment
run_experiment()
