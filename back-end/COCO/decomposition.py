# -*- coding: utf-8 -*-
"""Performs k-core decomposition on a temporal graph over various time intervals.

This script loads a temporal graph from a text file, generates a set of
hierarchically sampled time ranges, and then calculates the k-core numbers
for the subgraph corresponding to each time range. The computation is
parallelized using Python's multiprocessing module to improve performance.
The final results, mapping time ranges to the core numbers of their constituent
vertices, are saved to a text file.
"""

import os
import time
import random
import multiprocessing
from multiprocessing import Value
import graph_tool.all as gt

# --- Global Configuration ---

# The dataset name is retrieved from an environment variable, with 'collegemsg'
# as a default fallback.
DATASET_NAME = os.getenv("DATASET_NAME", "collegemsg")
# The partition factor controls the granularity of time range sampling.
PARTITION = 4
# The maximum size for the initial, smallest time ranges.
MAX_RANGE = 45
# The number of sub-ranges to sample from larger, non-leaf time ranges.
LOCAL_SAMPLE_NUM = 20
# The number of sub-ranges to sample from the smallest, leaf-level time ranges.
LOCAL_SAMPLE_NUM_LEAF = 10

# --- Global State Variables ---
# These variables are populated by `read_temporal_graph` and are treated as
# read-only by the parallel worker processes.

# Total number of unique vertices in the graph.
num_vertex = 0
# Total number of edges in the dataset file.
num_edge = 0
# The maximum timestamp value, defining the overall time span.
num_timestamp = 0
# A dictionary mapping each timestamp to a list of edges active at that time.
# Format: {timestamp: [(v1, v2), ...]}
time_edge = {}

# A shared counter for tracking the progress of the parallel processing tasks.
# It is safely incremented by each worker process upon task completion.
progress_counter = Value("i", 0)
# A global variable to hold the total number of ranges to be processed.
total_ranges = 0


def read_temporal_graph(filename):
    """Loads temporal graph data from a file into global variables.

    This function reads a file where the first line specifies the graph's
    dimensions (vertices, edges, timestamps), and subsequent lines represent
    temporal edges (v1, v2, t). It populates the global variables `num_vertex`,
    `num_edge`, `num_timestamp`, and the `time_edge` dictionary.

    Args:
        filename (str): The path to the dataset file.
    """
    global num_vertex, num_edge, num_timestamp, time_edge
    print(f"Reading temporal graph from: {filename}")
    with open(filename, "r") as f:
        is_first_line = True
        for line in f:
            if is_first_line:
                num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
                is_first_line = False
                continue
            v1, v2, t = map(int, line.strip().split())
            if t not in time_edge:
                time_edge[t] = []
            time_edge[t].append((v1, v2))

    # Ensure all timestamps up to num_timestamp have an entry in the dictionary,
    # even if they have no edges. This prevents key errors later.
    for t in range(num_timestamp):
        if t not in time_edge:
            time_edge[t] = []
    print("Graph data loaded successfully.")


def generate_ranges():
    """Generates a list of time ranges using a hierarchical sampling strategy.

    The strategy starts with single-timestamp ranges (leaves of the hierarchy).
    It then creates progressively larger ranges by a factor of `PARTITION`.
    From each of these larger ranges, it samples a fixed number of smaller
    sub-ranges to ensure coverage across different time scales.

    Returns:
        list[tuple[int, int]]: A list where each tuple represents a sampled
                               time range (start_time, end_time).
    """
    print("Generating hierarchical time ranges for sampling...")
    time_ranges = []

    # Level 0: Add all single-timestamp intervals.
    for t in range(num_timestamp):
        time_ranges.append((t, t))

    # Determine the starting size for hierarchical aggregation.
    temp_range_size = num_timestamp
    while temp_range_size > MAX_RANGE:
        temp_range_size //= PARTITION

    layer_id = 0
    while temp_range_size < num_timestamp:
        range_start = 0
        while range_start < num_timestamp:
            # Determine the number of samples for the current layer.
            # Leaf layers (the smallest aggregated ranges) get a different count.
            num_samples = LOCAL_SAMPLE_NUM_LEAF if layer_id == 0 else LOCAL_SAMPLE_NUM

            range_end = min(range_start + temp_range_size - 1, num_timestamp - 1)
            # Ensure the last range extends to the end of the timeline.
            if range_end + temp_range_size > num_timestamp - 1:
                range_end = num_timestamp - 1

            sampled_set = set()
            # The full range is always included.
            sampled_set.add((range_start, range_end))

            # Sample additional sub-ranges within the current large range.
            # This ensures that we analyze not just the full range, but also
            # diverse intervals within it.
            while (
                len(sampled_set) < num_samples
                and (range_end - range_start) > temp_range_size // PARTITION
            ):
                v1 = random.randint(
                    range_start, range_end - (temp_range_size // PARTITION)
                )
                v2 = random.randint(v1 + (temp_range_size // PARTITION), range_end)
                # Ensure the sampled sub-range is valid and non-trivial.
                if v1 < v2:
                    sampled_set.add((v1, v2))

            time_ranges.extend(list(sampled_set))

            # Move to the start of the next range.
            if range_end >= num_timestamp - 1:
                break
            range_start = range_end + 1

        temp_range_size *= PARTITION
        layer_id += 1

    print(f"Generated {len(time_ranges)} time ranges.")
    return time_ranges


def k_core_decomposition_for_range(time_range):
    """Performs k-core decomposition for a single time range.

    This function is executed by a worker process. It aggregates all unique
    edges within the given time range, constructs a `graph-tool` graph object,
    and then runs the k-core decomposition algorithm on it.

    Args:
        time_range (tuple[int, int]): A tuple containing the start and end
                                      timestamps of the range to analyze.

    Returns:
        tuple: A tuple containing the original time range and a dictionary
               mapping vertex IDs to their core numbers.
               Format: (start_time, end_time, {vertex_id: core_number, ...})
    """
    t_s, t_e = time_range

    # Collect all unique edges that appear within the specified time range.
    edge_set = set()
    for t in range(t_s, t_e + 1):
        # The use of `time_edge.get(t, [])` is a safe way to access edges,
        # although `read_temporal_graph` should prevent KeyErrors.
        for edge in time_edge.get(t, []):
            edge_set.add(edge)

    if not edge_set:
        # If no edges exist in this range, return an empty result.
        return (t_s, t_e, {})

    # Create a graph-tool Graph object.
    g = gt.Graph(directed=False)
    # Use a dictionary to map original vertex IDs to graph-tool's internal
    # vertex descriptors, which are added on-the-fly.
    vertex_map = {}

    # Add vertices and edges to the graph-tool graph.
    for u, v in edge_set:
        if u not in vertex_map:
            vertex_map[u] = g.add_vertex()
        if v not in vertex_map:
            vertex_map[v] = g.add_vertex()
        g.add_edge(vertex_map[u], vertex_map[v])

    # Perform the k-core decomposition. This returns a vertex property map.
    core_numbers_prop = gt.kcore_decomposition(g)

    # Convert the resulting property map back to a standard dictionary with
    # original vertex IDs.
    core_numbers_dict = {
        node: core_numbers_prop[vertex_map[node]] for node in vertex_map
    }

    # Safely increment the shared progress counter and print the status.
    with progress_counter.get_lock():
        progress_counter.value += 1
        print(
            f"Progress: {progress_counter.value}/{total_ranges} ranges processed. "
            f"Completed [{t_s}, {t_e}]."
        )

    return (t_s, t_e, core_numbers_dict)


def main():
    """Main execution function.

    Orchestrates the loading of data, generation of time ranges, parallel
    computation of k-core numbers, and saving of the results.
    """
    global total_ranges

    # Define file paths based on the dataset name.
    input_filename = f"../datasets/{DATASET_NAME}.txt"
    output_filename = f"../datasets/{DATASET_NAME}-core_number.txt"

    # Load the graph data into global memory.
    read_temporal_graph(input_filename)

    # Generate the list of time ranges to be analyzed.
    time_ranges = generate_ranges()
    total_ranges = len(time_ranges)

    dec_start_time = time.time()

    # Use all available CPU cores for parallel processing.
    num_threads = multiprocessing.cpu_count()
    print(f"Starting k-core decomposition with {num_threads} parallel processes...")

    # Create a processing pool and map the decomposition function to the ranges.
    # `chunksize=1` is often a good choice for tasks with variable completion
    # times, as it allows for more dynamic load balancing.
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.map(k_core_decomposition_for_range, time_ranges, chunksize=1)

    # Convert the list of result tuples into a single dictionary for easy lookup.
    time_range_core_numbers = {(r[0], r[1]): r[2] for r in results}
    dec_end_time = time.time()

    print(
        f"All computations finished. Total time: {dec_end_time - dec_start_time:.2f} seconds."
    )
    print(f"Saving core number results to: {output_filename}")

    # Write the results to the output file in the specified format.
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(f"{len(time_range_core_numbers)}\n")
        for key, value in time_range_core_numbers.items():
            # Format: "[start,end] v1:core1 v2:core2 ..."
            file.write(f"[{key[0]},{key[1]}] ")
            for v_id, core_num in value.items():
                file.write(f"{v_id}:{core_num} ")
            file.write("\n")

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
