# Initialize Spark Session with timeout configurations
from pyspark.sql import SparkSession

# Create a Spark session with timeout configurations
spark = SparkSession.builder \
    .appName("Distributed Gene Expression Analysis") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "30s") \
    .config("spark.sql.broadcastTimeout", "1200s") \
    .getOrCreate()

# Print Spark version
print(f"Spark version: {spark.version}")

#----End Of Cell----

# Import necessary libraries
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType, StringType, IntegerType, StructType, StructField
import os
import time

# Load data with optimized options
# Load training data
train_df = spark.read.option("mode", "DROPMALFORMED") \
    .option("header", "true") \
    .csv('bioinfo_data/data_set_ALL_AML_train.csv')

# Load test data
test_df = spark.read.option("mode", "DROPMALFORMED") \
    .option("header", "true") \
    .csv('bioinfo_data/data_set_ALL_AML_independent.csv')

# Load labels
labels_df = spark.read.option("header", "true") \
    .csv('bioinfo_data/actual.csv')

# Cache dataframes that will be reused
train_df.cache()
test_df.cache()
labels_df.cache()

# Display schema and sample data
print("Training data schema:")
train_df.printSchema()

print("\nSample training data:")
train_df.show(5)

# Create a mapping of sample IDs to cancer types
labels_pd = labels_df.toPandas()
labels_dict = dict(zip(labels_pd['patient'], labels_pd['cancer']))

# Extract sample IDs from column names (every other column starting from index 2)
train_sample_ids = []
for col_name in train_df.columns[2::2]:
    try:
        train_sample_ids.append(int(col_name))
    except ValueError:
        print(f"Warning: Could not convert column name '{col_name}' to integer")

# Print distribution of cancer types
print("\nCancer type distribution in training data:")
all_count = sum(1 for sid in train_sample_ids if labels_dict.get(sid) == 'ALL')
aml_count = sum(1 for sid in train_sample_ids if labels_dict.get(sid) == 'AML')
print(f"ALL: {all_count}, AML: {aml_count}")

#----End Of Cell----

# Function to transform gene expression data
def transform_gene_data(df, sample_ids, labels_dict):
    # Process in smaller batches to avoid memory issues
    batch_size = 1000  # Process 1000 genes at a time
    
    # Get total number of genes
    total_genes = df.count()
    
    # Initialize an empty array for gene values
    gene_values = []
    
    # Process in batches
    for i in range(0, total_genes, batch_size):
        # Get batch of rows
        batch = df.limit(batch_size).offset(i)
        
        # Process each row (gene) in the batch
        for row in batch.collect():
            # Extract expression values for this gene across all samples
            try:
                values = [float(row[j]) for j in range(2, len(row), 2)]
                gene_values.append(values)
            except (ValueError, TypeError) as e:
                # Skip rows with non-numeric values
                print(f"Warning: Skipping row due to error: {e}")
        
        # Print progress
        print(f"Processed {min(i + batch_size, total_genes)}/{total_genes} genes")
    
    # Transpose to get samples as rows and genes as columns
    gene_values = np.array(gene_values).T
    
    # Create a pandas DataFrame with samples as rows
    samples_df = pd.DataFrame(gene_values)
    
    # Add label column (0 for ALL, 1 for AML)
    samples_df['label'] = [1 if labels_dict.get(sid) == 'AML' else 0 for sid in sample_ids]
    
    # Convert to Spark DataFrame
    return spark.createDataFrame(samples_df)

# Set checkpoint directory
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

# Transform training data
print("Transforming training data...")
train_transformed = transform_gene_data(train_df, train_sample_ids, labels_dict)

# Checkpoint to avoid long lineage
train_transformed = train_transformed.checkpoint()
train_transformed.cache()

# Extract test sample IDs
test_sample_ids = []
for col_name in test_df.columns[2::2]:
    try:
        test_sample_ids.append(int(col_name))
    except ValueError:
        print(f"Warning: Could not convert column name '{col_name}' to integer")

# Transform test data
print("Transforming test data...")
test_transformed = transform_gene_data(test_df, test_sample_ids, labels_dict)
test_transformed.cache()

print("Transformed training data:")
train_transformed.show(5)

# Count number of features
num_features = len(train_transformed.columns) - 1  # Subtract 1 for label column
print(f"Number of features: {num_features}")

#----End Of Cell----

# Import ML libraries
from pyspark.ml.feature import VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Prepare feature columns
feature_cols = [str(i) for i in range(num_features)]

# Create ML pipeline with simpler configuration
# 1. Assemble features into a vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 2. Feature selection - first select top 1000 features
selector1 = ChiSqSelector(numTopFeatures=1000, featuresCol="features", 
                         outputCol="intermediateFeatures", labelCol="label")

# 3. Then select top 500 features
selector2 = ChiSqSelector(numTopFeatures=500, featuresCol="intermediateFeatures", 
                         outputCol="selectedFeatures", labelCol="label")

# 4. Standardize features
scaler = StandardScaler(inputCol="selectedFeatures", outputCol="scaledFeatures")

# 5. Train a Random Forest classifier with fewer trees
rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures", 
                          numTrees=20, maxDepth=10, seed=42)

# Create the pipeline
pipeline = Pipeline(stages=[assembler, selector1, selector2, scaler, rf])

# Start timing
start_time = time.time()

# Fit the model
print("Training model...")
model = pipeline.fit(train_transformed)

# Calculate training time
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

#----End Of Cell----

# Make predictions
print("Making predictions...")
predictions = model.transform(test_transformed)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test accuracy: {accuracy:.4f}")

# Show confusion matrix
print("Confusion matrix:")
predictions.groupBy("label", "prediction").count().show()

# Extract feature importances
rf_model = model.stages[-1]
feature_importances = rf_model.featureImportances

# Get the indices of the top 10 features
top_indices = feature_importances.toArray().argsort()[-10:][::-1]

print("Top 10 important features (indices):")
for idx in top_indices:
    print(f"Feature {idx}: {feature_importances[idx]:.6f}")

#----End Of Cell----

# Compare with MPI implementation
print("\nComparison with MPI Implementation:")
print(f"Spark training time: {training_time:.4f} seconds")
print(f"Spark test accuracy: {accuracy:.4f}")

# Insert your MPI results from your previous run
mpi_training_time = 0.0398  # Replace with your actual MPI training time
mpi_accuracy = 0.5882  # Replace with your actual MPI accuracy

print(f"MPI training time: {mpi_training_time:.4f} seconds")
print(f"MPI test accuracy: {mpi_accuracy:.4f}")

# Calculate speedup
if mpi_training_time > 0:
    speedup = mpi_training_time / training_time
    print(f"Speedup (MPI/Spark): {speedup:.2f}x")
    
    if speedup < 1:
        print("MPI was faster for this dataset, likely due to its smaller size and the overhead of Spark's distributed processing.")
    else:
        print("Spark was faster, demonstrating the benefits of its distributed processing model.")

# Compare accuracy
if accuracy > mpi_accuracy:
    print(f"Spark achieved higher accuracy by {(accuracy - mpi_accuracy)*100:.2f}%")
elif mpi_accuracy > accuracy:
    print(f"MPI achieved higher accuracy by {(mpi_accuracy - accuracy)*100:.2f}%")
else:
    print("Both approaches achieved the same accuracy.")

#----End Of Cell----

# Clean up
spark.stop()
print("Spark session stopped.")

#----End Of Cell----
