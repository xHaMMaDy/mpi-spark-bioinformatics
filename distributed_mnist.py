from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only master node loads the data
if rank == 0:
    print(f"Loading MNIST digits dataset on master node...")
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split data for distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Calculate chunk sizes for each process
    chunk_size = len(X_train) // size
    remainders = len(X_train) % size
else:
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    chunk_size = None
    remainders = None

# Broadcast test data to all processes
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Broadcast chunk size to all processes
chunk_size = comm.bcast(chunk_size, root=0)
remainders = comm.bcast(remainders, root=0)

# Calculate actual chunk size for this rank
my_chunk_size = chunk_size + (1 if rank < remainders else 0)

# Scatter training data
if rank == 0:
    # Prepare data chunks
    X_chunks = []
    y_chunks = []
    start_idx = 0
    
    for i in range(size):
        r_chunk_size = chunk_size + (1 if i < remainders else 0)
        X_chunks.append(X_train[start_idx:start_idx + r_chunk_size])
        y_chunks.append(y_train[start_idx:start_idx + r_chunk_size])
        start_idx += r_chunk_size
else:
    X_chunks = None
    y_chunks = None

# Scatter the data
my_X = comm.scatter(X_chunks, root=0)
my_y = comm.scatter(y_chunks, root=0)

print(f"Process {rank}: Training with {len(my_X)} samples")

# Train a model on local data
start_time = time.time()
clf = RandomForestClassifier(n_estimators=10, random_state=rank)
clf.fit(my_X, my_y)
training_time = time.time() - start_time

# Make predictions on test data
my_predictions = clf.predict(X_test)
my_accuracy = accuracy_score(y_test, my_predictions)

# Gather results
all_predictions = comm.gather(my_predictions, root=0)
all_accuracies = comm.gather(my_accuracy, root=0)
all_times = comm.gather(training_time, root=0)

# Master node combines results
if rank == 0:
    # Ensemble the predictions (majority vote)
    ensemble_predictions = np.zeros((len(y_test), 10), dtype=int)
    for pred in all_predictions:
        for i, p in enumerate(pred):
            ensemble_predictions[i, p] += 1
    
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    ensemble_accuracy = accuracy_score(y_test, final_predictions)
    
    print("\n===== Results =====")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Number of processes: {size}")
    
    print("\nIndividual Model Performance:")
    for i, (acc, t) in enumerate(zip(all_accuracies, all_times)):
        print(f"Process {i}: Accuracy = {acc:.4f}, Training Time = {t:.4f}s")
    
    print("\nEnsemble Model Performance:")
    print(f"Accuracy = {ensemble_accuracy:.4f}")
    print(f"Average Training Time = {sum(all_times)/len(all_times):.4f}s")
    print(f"Total Execution Time = {time.time() - start_time:.4f}s")
