# COCO: A Cohesiveness-aware Learning Framework for Community Search over Temporal Graphs

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **COCO: A Cohesiveness-aware Learning Framework for Community Search over Temporal Graphs**.

## Overview

COCO is a cohesiveness-aware learning framework designed for community search over temporal graphs. It seamlessly integrates the classical 𝑘-core decomposition with Graph Neural Networks (GNNs). The framework addresses the temporal interval community search problem through a three-stage process: pre-training, query-driven fine-tuning, and community search.


## Workflow and Usage

The end-to-end workflow consists of data preprocessing, model pre-training, and query execution.

### Step 1: Data Preparation

1.  **Download Datasets**: Obtain raw temporal graph data from public repositories such as [SNAP](http://snap.stanford.edu/) and [KONECT](http://konect.uni-koblenz.de/networks) and place them in the `datasets/` directory.

2.  **Process Graph Data**: Execute the script to convert the raw graph into the required format.

    ```bash
    python process_graph.py
    ```

      * **Output**: `datasets/<dataset_name>.txt`

3.  **Perform Core Decomposition**: Run the decomposition script to generate the core number file.

    ```bash
    python decomposition.py 
    ```
      * **Output**: `datasets/<dataset_name>-core_number.txt`

### Step 2: Model Pre-training

1.  **Train HM-Index (MLP Models)**: Train the MLP models that constitute the Cohesiveness Prediction Index.

    ```bash
    python MLP.py 
    ```
      * **Output**: Trained models saved in `models/<dataset_name>/`

2.  **Pre-train the Main GNN Model**: Run the main training script to pre-train the primary graph model.

    ```bash
    python main.py 
    ```
      * **Output**: Pre-trained model weights saved in the `./` directory.

### Step 3: Model Fine-tuning and Community Search

Perform query-driven fine-tuning on the pre-trained model, and then use the fine-tuned model for search.

```bash
python single_query.py 
```

## Repository Structure

#### Core Logic Files

  * `data_loader.py`: Reads datasets and constructs Graph objects.
  * `extract_subgraph.py`: Handles subgraph sampling and training tuple generation.
  * `loss.py`: Defines loss functions for model training.
  * `model.py`: Defines the AT-GNN and Adapter models.
  * `MLP_models.py`: Defines the series of MLP models for the HM-Index.
  * `train.py`: Implements model training and validation routines.
  * `utils.py`: Contains various utility functions.

#### Executable Scripts

  * `process_graph.py`: Preprocesses raw graph data.
  * `decomposition.py`: Performs k-core decomposition on a graph.
  * `MLP.py`: Trains the HM-Index models.
  * `main.py`: Pre-trains the main GNN model.
  * `single_query.py`: Executes a fine-tuning and community search task for a given query.
  * `MLP_search.py`: Utility script for analyzing HM-Index prediction targets.

