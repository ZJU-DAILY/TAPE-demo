import math

t_min = 99999999999
t_max = 0
num_vertex = 0

time_period = 86400


# load the temporal graph data
def read_temporal_graph(filename):
    print("loading temporal graph...")
    global t_min, t_max, max_vertex
    edges = []
    timestamps = []
    with open(filename, "r") as f:

        next(f)
        for line in f:
            v1, v2, t_w, t_str = map(int, line.strip().split())
            if v1 == v2:
                continue
            edges.append((v1, v2))
            t = int(t_str)
            timestamps.append(t)
            if t > t_max:
                t_max = t
            if t < t_min:
                t_min = t
    return edges, timestamps


def partition_time(edges, timestamps):
    global num_vertex
    start_time = 0
    while start_time + time_period <= t_min:
        start_time = start_time + time_period
    new_edge_set = set()
    vertex_set = set()
    vertex_map = {}
    for i in range(len(edges)):
        v1 = edges[i][0]
        v2 = edges[i][1]
        if v1 == v2:
            continue
        elif v1 > v2:
            v1, v2 = v2, v1
        # print(timestamps[i] - start_time)
        t = (timestamps[i] - start_time) // time_period
        # print(t)
        if v1 not in vertex_set:
            vertex_set.add(v1)
            vertex_map[v1] = len(vertex_set) - 1
        if v2 not in vertex_set:
            vertex_set.add(v2)
            vertex_map[v2] = len(vertex_set) - 1

        new_v1 = vertex_map[v1]
        new_v2 = vertex_map[v2]

        if (new_v1, new_v2, t) not in new_edge_set:
            new_edge_set.add((new_v1, new_v2, t))
    num_vertex = len(vertex_set)
    return new_edge_set


print("preprocessing graph...")
# filename = 'datasets/sx-mathoverflow-old.txt'
filename = "/home/lcy/liucy/datasets/edit-dewiki/out.edit-dewiki"
edges, timestamps = read_temporal_graph(filename)
edges_set = partition_time(edges, timestamps)
num_edges = len(edges_set)
num_timestamps = t_max - t_min + 1
num_days = math.ceil(num_timestamps / time_period)
edges_set = sorted(edges_set, key=lambda x: (x[2], x[0], x[1]))
# new_filename = 'datasets/sx-mathoverflow.txt'
new_filename = "../datasets/dewiki.txt"
with open(new_filename, "w") as f:
    f.write(f"{num_vertex} {num_edges} {num_days}\n")
    for edge in edges_set:
        f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
