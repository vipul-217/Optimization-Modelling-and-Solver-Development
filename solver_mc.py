import numpy as np
import random
import math

class MaxCutSolver:
    def __init__(self, file_path):
        self.nodes, self.edges = self._parse_gset(file_path)
        self.adj = self._build_adj_matrix()

    def _parse_gset(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        v, e = map(int, lines[0].split())
        edges = []
        for line in lines[1:]:
            if line.strip():
                u, v_node, w = map(int, line.split())
                # GSET is 1-indexed, converting to 0-indexed
                edges.append((u-1, v_node-1, w))
        return v, edges

    def _build_adj_matrix(self):
        # Using a dictionary for adjacency to handle sparsity efficiently
        adj = {}
        for u, v, w in self.edges:
            adj.setdefault(u, {})[v] = w
            adj.setdefault(v, {})[u] = w
        return adj

    def calculate_cut_weight(self, state):
        """
        Calculates the weight of the cut for a given state (binary array).
        state[i] = 0 or 1
        """
        weight = 0
        for u, v, w in self.edges:
            if state[u] != state[v]:
                weight += w
        return weight

    def solve_simulated_annealing(self, initial_temp=100.0, cooling_rate=0.9995, iterations=50000):
        current_state = np.random.randint(2, size=self.nodes)
        current_weight = self.calculate_cut_weight(current_state)
        
        best_state = np.copy(current_state)
        best_weight = current_weight
        
        temp = initial_temp

        for i in range(iterations):
            node_to_flip = random.randint(0, self.nodes - 1)
            
            delta = 0
            if node_to_flip in self.adj:
                for neighbor, weight in self.adj[node_to_flip].items():
                    if current_state[node_to_flip] == current_state[neighbor]:
                        delta += weight
                    else:
                        delta -= weight

            # Acceptance criteria
            if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
                current_state[node_to_flip] = 1 - current_state[node_to_flip]
                current_weight += delta
                
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_state = np.copy(current_state)

            # Slower cooling
            temp *= cooling_rate
            
            # Periodic Reheating: If temp gets too low, bump it back up
            if temp < 0.01:
                temp = initial_temp * 0.5  # Reheat to half initial temp
                # Optional: current_state = np.copy(best_state) # Start from best known

            if i % 5000 == 0:
                print(f"Iteration {i}: Best Weight = {best_weight}, Temp = {temp:.4f}")

        return best_weight, best_state
    
    def local_search(self, state):
        """
        Polishes the state by performing greedy flips until a local optimum is reached.
        """
        current_state = np.copy(state)
        current_weight = self.calculate_cut_weight(current_state)
        improved = True
        
        while improved:
            improved = False
            # Check every node
            for node in range(self.nodes):
                delta = 0
                if node in self.adj:
                    for neighbor, weight in self.adj[node].items():
                        if current_state[node] == current_state[neighbor]:
                            delta += weight
                        else:
                            delta -= weight
                
                if delta > 0:
                    current_state[node] = 1 - current_state[node]
                    current_weight += delta
                    improved = True
        return current_weight, current_state
