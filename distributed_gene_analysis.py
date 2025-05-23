from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import os

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paths to the dataset files
train_path = os.path.join(os.path.expanduser("~"), "bioinfo_data", "data_set_ALL_AML_train.csv")
test_path = os.path.join(os.path.expanduser("~"), "bioinfo_data", "data_set_ALL_AML_independent.csv")
labels_path = os.path.join(os.path.expanduser("~"), "bioinfo_data", "actual.csv")

# Only master node loads the data
if rank == 0:
    print(f"Loading Golub leukemia dataset on master node...")
    
    # Load labels first
    labels_df = pd.read_csv(labels_path)
    labels_dict = dict(zip(labels_df['patient'], labels_df['cancer']))
    
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Extract sample IDs from column names (every other column starting from index 2)
    train_sample_ids = [int(col) for col in train_df.columns[2::2]]
    
    # Extract gene expression values (every other column starting from index 2)
    X_train_raw = train_df.iloc[:, 2::2].values
    
    # Transpose to get samples as rows and genes as columns
    X_train = X_train_raw.T
    
    # Create labels for training data
    y_train = np.array([1 if labels_dict.get(sample_id) == 'AML' else 0 for sample_id in train_sample_ids])
    
    # Load test data
    test_df = pd.read_csv(test_path)
    
    # Extract sample IDs from column names (every other column starting from index 2)
    test_sample_ids = [int(col) for col in test_df.columns[2::2]]
    
    # Extract gene expression values (every other column starting from index 2)
    X_test_raw = test_df.iloc[:, 2::2].values
    
    # Transpose to get samples as rows and genes as columns
    X_test = X_test_raw.T
    
    # Create labels for test data
    y_test = np.array([1 if labels_dict.get(sample_id) == 'AML' else 0 for sample_id in test_sample_ids])
    
    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
    print(f"Number of genes: {X_train.shape[1]}")
    print(f"Training labels distribution: ALL: {np.sum(y_train == 0)}, AML: {np.sum(y_train == 1)}")
    print(f"Test labels distribution: ALL: {np.sum(y_test == 0)}, AML: {np.sum(y_test == 1)}")
    
    # Feature selection on master node
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(500, X_train.shape[1]))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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
clf = RandomForestClassifier(n_estimators=50, random_state=rank)
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
    ensemble_predictions = np.zeros((len(y_test), 2), dtype=int)
    for pred in all_predictions:
        for i, p in enumerate(pred):
            ensemble_predictions[i, p] += 1
    
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    ensemble_accuracy = accuracy_score(y_test, final_predictions)
    
    print("\n===== Golub Leukemia Analysis Results =====")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Number of processes: {size}")
    print(f"Features after selection: {X_train.shape[1]}")
    
    print("\nIndividual Model Performance:")
    for i, (acc, t) in enumerate(zip(all_accuracies, all_times)):
        print(f"Process {i}: Accuracy = {acc:.4f}, Training Time = {t:.4f}s")
    
    print("\nEnsemble Model Performance:")
    print(f"Accuracy = {ensemble_accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, final_predictions))
    print(f"Average Training Time = {sum(all_times)/len(all_times):.4f}s")
    print(f"Total Execution Time = {time.time() - start_time:.4f}s")

