# Mini-HPC and Hybrid HPC-Big Data Clusters

A comprehensive distributed computing project demonstrating both traditional HPC and modern Big Data processing approaches through machine learning on bioinformatics datasets.

## 🎯 Project Overview

This project implements two distinct distributed computing paradigms:

- **Task 1**: Traditional Mini-HPC cluster using MPI for distributed machine learning
- **Task 2**: Hybrid HPC-Big Data cluster using Docker Swarm and Apache Spark

Both approaches are evaluated on real-world bioinformatics datasets, specifically gene expression analysis for leukemia classification using the Golub dataset.

## 🏗️ Architecture

### Task 1: Mini-HPC Cluster

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Master    │    │  Worker 1   │    │  Worker 2   │
│    Node     │────│    Node     │────│    Node     │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                    MPI Communication
```


### Task 2: Docker Swarm + Spark Cluster

```
┌─────────────────────────────────────────────────────┐
│                Docker Swarm                         │
├─────────────┬─────────────────┬─────────────────────┤
│ Spark       │ Spark Worker 1  │ Spark Worker 2      │
│ Master      │                 │                     │
│ + Jupyter   │                 │                     │
└─────────────┴─────────────────┴─────────────────────┘
```


## 📊 Datasets

**Golub Leukemia Dataset**: Gene expression data for Acute Lymphoblastic Leukemia (ALL) vs Acute Myeloid Leukemia (AML) classification

- **Training samples**: 38 samples across 7,129 genes
- **Test samples**: 34 samples
- **Challenge**: High-dimensional feature space with small sample size

**MNIST Digits Dataset**: Used for initial MPI testing and validation

- **Samples**: 1,797 digit images (8x8 pixels)
- **Classes**: 10 digits (0-9)
- **Purpose**: Validate MPI implementation before bioinformatics analysis


## 🚀 Quick Start

### Prerequisites

- **Hardware**: 3 Virtual Machines (1 master + 2 workers)
- **OS**: Ubuntu 20.04+ or CentOS 7+
- **Memory**: Minimum 4GB per VM
- **Network**: Configured for inter-VM communication


### Task 1: MPI Cluster Setup

**Install Dependencies**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install OpenMPI
sudo apt install openmpi-bin openmpi-common libopenmpi-dev -y

# Install Python dependencies
pip install mpi4py numpy pandas scikit-learn
```

**Configure Passwordless SSH**

```bash
# Generate SSH key on master
ssh-keygen -t rsa -b 4096

# Copy to all nodes
ssh-copy-id user@worker1
ssh-copy-id user@worker2
```

**Create Hostfile**

```bash
# /etc/openmpi/hostfile
master-node slots=1
worker1-node slots=2  
worker2-node slots=2
```


### Task 2: Docker Swarm + Spark Setup

**Initialize Docker Swarm**

```bash
# On master node
docker swarm init --advertise-addr <master-ip>

# Join workers (run on worker nodes)
docker swarm join --token <token> <master-ip>:2377
```

**Deploy Spark Cluster**

```bash
# Deploy the stack
docker stack deploy -c spark-stack.yml spark-cluster

# Verify deployment
docker service ls
```


## 🎮 Usage

### Running MPI Distributed Analysis

**MNIST Digit Classification**

```bash
mpirun -np 4 --hostfile hostfile python distributed_mnist.py
```

**Gene Expression Analysis**

```bash
mpirun -np 4 --hostfile hostfile python distributed_gene_analysis.py
```


### Running Spark Distributed Analysis

**Access Jupyter Notebook**

```bash
# Navigate to http://master-ip:8888
# Open distributed_gene_expression_analysis.py
```

**Monitor Spark Jobs**

```bash
# Spark UI: http://master-ip:8080
# Job monitoring and resource utilization
```


## 📈 Performance Results

### MPI Implementation Results

- **Training Time**: 0.0398 seconds (average across processes)
- **Test Accuracy**: 58.82%
- **Ensemble Method**: Majority voting across distributed models
- **Scalability**: Linear speedup with additional processes


### Spark Implementation Results

- **Training Time**: ~15-20 seconds (including overhead)
- **Test Accuracy**: ~65-70% (improved with advanced feature selection)
- **Features**: Automated feature selection pipeline (7,129 → 500 features)
- **Advantage**: Better handling of large-scale data processing


### Performance Comparison

| Metric | MPI | Spark |
| :-- | :-- | :-- |
| **Setup Complexity** | Low | Medium |
| **Training Time** | 0.04s | 15-20s |
| **Accuracy** | 58.82% | 65-70% |
| **Scalability** | Excellent | Good |
| **Fault Tolerance** | Limited | High |
| **Big Data Support** | Limited | Excellent |

## 🔬 Technical Implementation

### MPI Approach

- **Data Distribution**: Manual data partitioning across processes
- **Model Training**: Independent RandomForest models per process (50 estimators)
- **Feature Selection**: SelectKBest with f_classif (top 500 features)
- **Aggregation**: Majority voting ensemble
- **Communication**: Point-to-point and collective MPI operations


### Spark Approach

- **Data Processing**: Distributed DataFrames with automatic partitioning
- **Feature Engineering**: MLlib pipeline with ChiSqSelector (1000 → 500 features)
- **Model Training**: Distributed RandomForest (20 trees, maxDepth=10)
- **Fault Tolerance**: Automatic recovery and lineage tracking
- **Optimization**: Checkpointing and caching for performance


## 📁 Project Structure

```
project_hpc_hybrid_cluster/
├── README.md                           # This file
├── hostfile                           # MPI hostfile configuration
├── distributed_mnist.py               # MPI MNIST classification
├── distributed_gene_analysis.py       # MPI gene expression analysis
├── spark-stack.yml                    # Docker Swarm Spark deployment
├── distributed_gene_expression_analysis.py  # Spark gene analysis
├── bioinfo_data/                      # Dataset directory
│   ├── data_set_ALL_AML_train.csv    # Training data
│   ├── data_set_ALL_AML_independent.csv # Test data
│   └── actual.csv                     # Labels
├── screenshots/                       # Setup and results screenshots
└── Final_Report.pdf                   # Comprehensive analysis report
```

## 🎯 Key Learning Outcomes

- **HPC Fundamentals**: Understanding of distributed computing paradigms
- **MPI Programming**: Hands-on experience with message passing interface
- **Big Data Processing**: Apache Spark ecosystem and MLlib
- **Container Orchestration**: Docker Swarm cluster management
- **Bioinformatics ML**: Real-world gene expression analysis
- **Performance Analysis**: Comparative evaluation of different approaches

## 📚 References

- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [Apache Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Golub et al. (1999) - Molecular Classification of Cancer](https://www.science.org/doi/10.1126/science.286.5439.531)
- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for distributed computing education**
