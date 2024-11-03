import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def generate_random_points(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    points = np.random.rand(N, 2)
    return points


def compute_distance_matrix(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix


def edges_cross(p1, p2, q1, q2):
    """Check if the line segments (p1,p2) and (q1,q2) cross."""
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def eliminate_blocking_edges_visibility(points, all_edges, dist_matrix, grid_size=10):
    """
    Eliminate edges that block visibility between vertex pairs using spatial partitioning.
    
    Parameters:
    - points: numpy array of point coordinates.
    - all_edges: set of all possible edges (tuples).
    - dist_matrix: precomputed distance matrix.
    - grid_size: number of cells per axis for the grid.
    
    Returns:
    - kept_edges: set of edges that do not block visibility based on criteria.
    """
    N = len(points)
    kept_edges = set()
    
    # Sort edges by ascending order of length (shorter edges first)
    edges_sorted = sorted(all_edges, key=lambda edge: dist_matrix[edge[0], edge[1]])

    # Initialize visibility counts for each vertex
    visibility_counts = {v: N - 1 for v in range(N)}
    # Initialize visible_pairs as all possible pairs
    visible_pairs = set(combinations(range(N), 2))

    print("Starting edge elimination based on visibility criteria with spatial partitioning...")

    # Define grid boundaries
    min_x, min_y = 0.0, 0.0
    max_x, max_y = 1.0, 1.0
    cell_width = (max_x - min_x) / grid_size
    cell_height = (max_y - min_y) / grid_size

    # Initialize grid cells
    grid = [[set() for _ in range(grid_size)] for _ in range(grid_size)]

    def get_grid_cells(p_start, p_end):
        """Determine which grid cells the edge from p_start to p_end intersects."""
        x1, y1 = p_start
        x2, y2 = p_end

        # Compute bounding box of the edge
        min_cell_x = int(min(x1, x2) / cell_width)
        max_cell_x = int(max(x1, x2) / cell_width)
        min_cell_y = int(min(y1, y2) / cell_height)
        max_cell_y = int(max(y1, y2) / cell_height)

        # Clamp to grid boundaries
        min_cell_x = max(min_cell_x, 0)
        max_cell_x = min(max_cell_x, grid_size - 1)
        min_cell_y = max(min_cell_y, 0)
        max_cell_y = min(max_cell_y, grid_size - 1)

        cells = set()
        for i in range(min_cell_x, max_cell_x + 1):
            for j in range(min_cell_y, max_cell_y + 1):
                cells.add((i, j))
        return cells

    # Iterate over sorted edges
    for idx, edge in enumerate(edges_sorted):
        A, B = edge
        p1, p2 = points[A], points[B]

        # Find grid cells the edge intersects
        cells = get_grid_cells(p1, p2)

        # Collect potential blocking edges from these cells
        potential_blocking_edges = set()
        for (i, j) in cells:
            potential_blocking_edges.update(grid[i][j])

        # Flag to determine if the current edge should be eliminated
        eliminate_edge = False

        # Find all pairs that this edge would block (i.e., edges that cross with this edge)
        blocking_pairs = set()
        for pair in visible_pairs:
            C, D = pair
            # Skip if the pair shares a vertex with the current edge
            if C in edge or D in edge:
                continue
            q1, q2 = points[C], points[D]
            # Check if (A,B) crosses (C,D)
            if edges_cross(p1, p2, q1, q2):
                blocking_pairs.add(pair)

        # Check each blocking pair to see if adding this edge would violate the criteria
        for pair in blocking_pairs:
            C, D = pair
            current_vis_C = visibility_counts[C]
            current_vis_D = visibility_counts[D]

            # After blocking, C and D lose visibility to each other
            new_vis_C = current_vis_C - 1
            new_vis_D = current_vis_D - 1

            if new_vis_C <= 1 or new_vis_D <= 1:
                # Adding this edge would cause C or D to have visibility to <=1 vertices
                eliminate_edge = True
                print(f"Eliminating edge {edge} as it blocks visibility between pair {pair}, "
                      f"causing visibility counts to drop to C:{new_vis_C}, D:{new_vis_D}.")
                break  # No need to check further pairs

        if not eliminate_edge:
            # Keep the edge
            kept_edges.add(edge)
            # Assign the edge to grid cells
            for (i, j) in cells:
                grid[i][j].add(edge)
            # Update visibility by removing blocked pairs
            for pair in blocking_pairs:
                if pair in visible_pairs:
                    visible_pairs.remove(pair)
                    C, D = pair
                    visibility_counts[C] -= 1
                    visibility_counts[D] -= 1
            print(f"Keeping edge {edge}. Total kept edges: {len(kept_edges)}")
        else:
            # Edge is eliminated; no need to assign it to grid cells
            pass

    print(f"\nTotal non-blocking edges after elimination: {len(kept_edges)}")
    return kept_edges


def solve_tsp_ortools(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000000)  # Scale to integer

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set parameters for a better solution
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 30  # Increased time limit for better solutions

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        tsp_route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tsp_route.append(node)
            index = solution.Value(routing.NextVar(index))
        tsp_route.append(manager.IndexToNode(index))
        return tsp_route
    else:
        print("No solution found for TSP (OR-Tools).")
        return []


def plot_results(points, non_blocking_edges, tsp_route_ortools, cycle, iteration):
    plt.figure(figsize=(11, 11))
    plt.scatter(points[:, 0], points[:, 1], color='blue', zorder=5, label='Points')

    # Plot non-blocking edges
    for (p, r) in non_blocking_edges:
        plt.plot([points[p, 0], points[r, 0]], [points[p, 1], points[r, 1]],
                 color='red', linewidth=1, alpha=0.5)

    # Plot TSP route from OR-Tools (if available)
    if tsp_route_ortools:
        tsp_x = [points[node, 0] for node in tsp_route_ortools]
        tsp_y = [points[node, 1] for node in tsp_route_ortools]
        plt.plot(tsp_x, tsp_y, color='cyan', linewidth=3, alpha=0.9, linestyle='--', label='TSP Tour (OR-Tools)')
        plt.scatter([points[node, 0] for node in tsp_route_ortools[:-1]],
                    [points[node, 1] for node in tsp_route_ortools[:-1]],
                    color='cyan', s=50, zorder=7)

    # Plot the found cycle
    if cycle:
        cycle_x = [points[node, 0] for node in cycle]
        cycle_y = [points[node, 1] for node in cycle]
        plt.plot(cycle_x, cycle_y, color='magenta', linewidth=2.5, alpha=0.9, linestyle='-', label='Found Cycle (Custom)')
        plt.scatter([points[node, 0] for node in cycle[:-1]],
                    [points[node, 1] for node in cycle[:-1]],
                    color='magenta', s=50, zorder=8)

    # Highlighting the start/end point
    if cycle:
        plt.scatter(points[cycle[0], 0], points[cycle[0], 1],
                    color='green', edgecolors='black', s=200, zorder=10, label='Start/End Point')

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Points',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], color='red', lw=1, label='Non-Blocking Edges'),
        Line2D([0], [0], color='cyan', lw=3, linestyle='--', label='TSP Tour (OR-Tools)'),
        Line2D([0], [0], color='magenta', lw=2.5, linestyle='-', label='Found Cycle (Custom)'),
        Line2D([0], [0], marker='o', color='w', label='Start/End Point',
               markerfacecolor='green', markeredgecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='best')

    title = f'Iteration {iteration}'
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def compare_cycles(cycle1, cycle2):
    if not cycle1 or not cycle2:
        return False

    # Remove the last element if it's the same as the first (cycle completion)
    if cycle1[0] == cycle1[-1]:
        cycle1 = cycle1[:-1]
    if cycle2[0] == cycle2[-1]:
        cycle2 = cycle2[:-1]

    if len(cycle1) != len(cycle2):
        return False

    N = len(cycle1)
    cycle1_doubled = cycle1 * 2
    cycle2_reversed = cycle2[::-1]

    for i in range(N):
        # Check for rotations
        if cycle1_doubled[i:i+N] == cycle2:
            return True
        if cycle1_doubled[i:i+N] == cycle2_reversed:
            return True

    return False


def find_shortest_hamiltonian_cycle_dp(N, collected_edges, dist_matrix):
    from collections import defaultdict

    edge_set = set(collected_edges)

    total_states = 0  # To count total number of states in DP
    total_transitions = 0  # To count total number of transitions considered

    # Initialize DP table
    C = {}
    C[(1 << 0, 0)] = (0, -1)

    # Total possible subsets including node 0
    total_subsets = 1 << (N - 1)

    print(f"Total number of subsets to consider: {total_subsets}")

    for subset_size in range(2, N + 1):
        print(f"Processing subsets of size {subset_size}")
        subsets = [set(s) | {0} for s in combinations(range(1, N), subset_size - 1)]
        for subset in subsets:
            subset_mask = sum([1 << i for i in subset])
            for j in subset:
                if j == 0:
                    continue
                prev_subset = subset - {j}
                prev_subset_mask = subset_mask ^ (1 << j)
                min_cost = float('inf')
                min_prev = -1
                for k in prev_subset:
                    if (k, j) in edge_set or (j, k) in edge_set:
                        prev_entry = C.get((prev_subset_mask, k))
                        if prev_entry is not None:
                            prev_cost = prev_entry[0]
                            cost = prev_cost + dist_matrix[k][j]
                            total_transitions += 1
                            if cost < min_cost:
                                min_cost = cost
                                min_prev = k
                if min_prev != -1:
                    C[(subset_mask, j)] = (min_cost, min_prev)
                    total_states += 1

    subset_mask = (1 << N) - 1
    min_cost = float('inf')
    min_prev = -1
    for j in range(1, N):
        if (j, 0) in edge_set or (0, j) in edge_set:
            entry = C.get((subset_mask, j))
            if entry is not None:
                cost = entry[0] + dist_matrix[j][0]
                total_transitions += 1
                if cost < min_cost:
                    min_cost = cost
                    min_prev = j

    if min_prev == -1:
        print("No Hamiltonian cycle found.")
        return [], float('inf'), total_states, 0

    # Reconstruct cycle
    path = [0]
    last = min_prev
    subset_mask = (1 << N) - 1
    while last != -1 and subset_mask:
        path.append(last)
        temp_mask = subset_mask
        subset_mask ^= (1 << last)
        last = C.get((temp_mask, last), (0, -1))[1]

    path.append(0)
    path.reverse()

    return path, min_cost, total_states, 1  # total_states as cycles_checked, 1 valid cycle


def main(N, seed):
    # Start timing
    start_time = time.time()

    # Step 1: Generate random points
    points = generate_random_points(N, seed)

    # Calculate and print the total number of possible edges for N points
    total_possible_edges = (N * (N - 1)) // 2
    print(f"Total possible edges for {N} points (fully connected graph): {total_possible_edges}")

    # Step 2: Compute distance matrix
    dist_matrix = compute_distance_matrix(points)

    # Step 3: Solve TSP using OR-Tools for reference
    print("\nSolving TSP using OR-Tools...")
    tsp_route_ortools = solve_tsp_ortools(dist_matrix)

    if tsp_route_ortools:
        # Compute total distance of TSP solution
        tsp_length = 0.0
        for i in range(len(tsp_route_ortools) - 1):
            tsp_length += dist_matrix[tsp_route_ortools[i]][tsp_route_ortools[i+1]]

        print(f"\nTotal distance of TSP solution (OR-Tools): {tsp_length:.4f}")

        # Extract TSP edges from OR-Tools solution
        tsp_edges_ortools = set()
        for i in range(len(tsp_route_ortools) - 1):
            edge = tuple(sorted((tsp_route_ortools[i], tsp_route_ortools[i+1])))
            tsp_edges_ortools.add(edge)
    else:
        tsp_length = float('inf')
        tsp_edges_ortools = set()

    # Step 4: Collect all possible edges
    print(f"\nCollecting all possible edges.")
    all_edges = set(combinations(range(N), 2))
    print(f"Total collected edges: {len(all_edges)}")

    # Step 5: Eliminate blocking edges based on visibility with spatial partitioning
    print("\nEliminating blocking edges based on visibility criteria.")
    non_blocking_edges = eliminate_blocking_edges_visibility(points, all_edges, dist_matrix, grid_size=10)

    # Check if all TSP edges are collected
    missing_tsp_edges = tsp_edges_ortools - non_blocking_edges
    if not missing_tsp_edges:
        print("All TSP edges have been collected in the non-blocking edges.")
    else:
        print(f"Missing TSP edges: {missing_tsp_edges}")

    # Attempt to find the shortest Hamiltonian cycle within the collected edges using DP
    print("\nStarting to find the shortest Hamiltonian cycle using DP...")
    cycle, cycle_length, cycles_checked, valid_cycles = find_shortest_hamiltonian_cycle_dp(
        N, non_blocking_edges, dist_matrix)

    if cycle:
        print(f"\nFound a Hamiltonian cycle with total length: {cycle_length:.4f}")
        # Compare with OR-Tools' solution
        match = compare_cycles(cycle, tsp_route_ortools)
        if match:
            print(f"Success: The found cycle matches the OR-Tools TSP solution.")
        else:
            print(f"The found cycle does NOT match the OR-Tools TSP solution.")
    else:
        print(f"\nNo Hamiltonian cycle found with the current edge set.")

    # Print cycle statistics
    print("\n--- Cycle Search Statistics ---")
    print(f"Total DP states computed: {cycles_checked}")
    print(f"Total transitions considered: {cycles_checked}")
    print(f"Total valid Hamiltonian cycles found: {valid_cycles}")
    if cycle:
        print(f"Best cycle length found: {cycle_length:.4f}")
        print(f"TSP cycle length (OR-Tools): {tsp_length:.4f}")
    else:
        print("No valid cycles found.")
    print("--------------------------------\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if cycle:
        print(f"Final Result: Found a Hamiltonian cycle.")
    else:
        print(f"Final Result: Could not find a Hamiltonian cycle with the current edge set.")

    print(f"Total collected edges: {len(non_blocking_edges)}")
    print(f"Total TSP edges (OR-Tools): {len(tsp_edges_ortools)}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")

    # Step 6: Plot the results
    plot_results(points, non_blocking_edges, tsp_route_ortools, cycle, "Final")


if __name__ == "__main__":
    N = 20
    seed = 1234
    main(N, seed)
