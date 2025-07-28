# enhanced_pauli_noise_user_friendly.py
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
import random

# ================================
# GLOBAL SIMULATION PARAMETERS
# ================================

# Core simulation parameters - MODIFY THESE TO CHANGE THE SIMULATION
QUBIT_RANGES = [3, 4, 5, 6, 7, 8]  # List of qubit counts to test
NUM_TRIALS = 10  # Number of Monte Carlo trials per configuration
RESULTS_FOLDER = "Enhanced_Results"  # Folder name for all outputs

# Graph topologies to test
GRAPH_TYPES = [
    "complete", "ring", "line", "star",
    "grid_2d", "binary_tree", "ladder", "wheel",
    "small_world", "scale_free", "triangular_lattice",
    "custom_ring_with_shortcuts", "custom_double_star",
    "custom_branched_path", "custom_modular", "custom_sparse_random"
]

# Weather models to test
WEATHER_MODELS = ["rain", "dust", "turbulence", "clear"]

# Intensity range for weather effects
INTENSITY_RANGE = np.linspace(0.2, 1.5, 6)  # From 0.2 to 1.5 in 6 steps

# Basic noise analysis parameters
BASIC_NOISE_N_RANGE = range(1, 3)  # Number of errors to test
BASIC_NOISE_P_RANGE = np.linspace(0, 1, 20)  # Probability range
BASIC_NOISE_SAMPLES = 30  # Samples per point

# Visualization parameters
FIGURE_DPI = 300  # DPI for saved figures
FIGURE_SIZE_LARGE = (14, 9)  # Large figure size
FIGURE_SIZE_MEDIUM = (12, 8)  # Medium figure size
FIGURE_SIZE_SMALL = (8, 6)  # Small figure size


# ================================
# ORIGINAL CLASSES (unchanged)
# ================================

class PauliOperators:
    """Constants for Pauli operators."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.array([[1, 0], [0, 1]], dtype=complex)

    @classmethod
    def get_operators(cls) -> Dict[str, np.ndarray]:
        """Get all Pauli operators except identity."""
        return {'X': cls.X, 'Y': cls.Y, 'Z': cls.Z}


class BaseQuantumState:
    """Base class for quantum state operations."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = None
        self.graph = nx.Graph()

    @property
    def density_matrix(self) -> np.ndarray:
        """Convert state vector to density matrix."""
        if self.state is None:
            raise ValueError("State not initialized")
        return np.outer(self.state, self.state.conj())

    def create_state_vector(self, initial_state: int = 0) -> np.ndarray:
        """Create initial state vector."""
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[initial_state] = 1
        return state

    def apply_hadamard_to_all(self, state: np.ndarray) -> np.ndarray:
        """Apply Hadamard gate to all qubits."""
        h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        for i in range(self.num_qubits):
            h_full = np.kron(np.eye(2 ** i),
                             np.kron(h_matrix, np.eye(2 ** (self.num_qubits - i - 1))))
            state = h_full @ state
        return state

    def apply_cz_gate(self, state: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Apply controlled-Z gate between two qubits."""
        full_cz = np.eye(2 ** self.num_qubits)
        mask = 2 ** qubit1 + 2 ** qubit2

        for k in range(2 ** self.num_qubits):
            if bin(k & mask).count('1') == 2:
                full_cz[k, k] = -1

        return full_cz @ state


class EnhancedGraphStateBuilder(BaseQuantumState):
    """Enhanced builder for creating graph states with various topologies including custom ones."""

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)

    # Basic topology builders
    TOPOLOGY_BUILDERS = {
        "complete": lambda n: [(i, j) for i in range(n) for j in range(i + 1, n)],
        "line": lambda n: [(i, i + 1) for i in range(n - 1)],
        "ring": lambda n: [(i, (i + 1) % n) for i in range(n)],
        "star": lambda n: [(0, i) for i in range(1, n)],

        # New custom topologies
        "grid_2d": lambda n: EnhancedGraphStateBuilder._create_2d_grid(n),
        "binary_tree": lambda n: EnhancedGraphStateBuilder._create_binary_tree(n),
        "ladder": lambda n: EnhancedGraphStateBuilder._create_ladder(n),
        "wheel": lambda n: EnhancedGraphStateBuilder._create_wheel(n),
        "random_regular": lambda n: EnhancedGraphStateBuilder._create_random_regular(n, 3),
        "small_world": lambda n: EnhancedGraphStateBuilder._create_small_world(n),
        "scale_free": lambda n: EnhancedGraphStateBuilder._create_scale_free(n),
        "hypercube": lambda n: EnhancedGraphStateBuilder._create_hypercube(n),
        "triangular_lattice": lambda n: EnhancedGraphStateBuilder._create_triangular_lattice(n),
    }

    @staticmethod
    def _create_2d_grid(n: int) -> List[Tuple[int, int]]:
        """Create a 2D grid topology."""
        rows = int(np.sqrt(n))
        cols = n // rows
        if rows * cols < n:
            cols += 1

        edges = []
        for i in range(rows):
            for j in range(cols):
                node = i * cols + j
                if node >= n:
                    break

                if j + 1 < cols and node + 1 < n:
                    edges.append((node, node + 1))

                if i + 1 < rows and node + cols < n:
                    edges.append((node, node + cols))

        return edges

    @staticmethod
    def _create_binary_tree(n: int) -> List[Tuple[int, int]]:
        """Create a binary tree topology."""
        edges = []
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if left_child < n:
                edges.append((i, left_child))
            if right_child < n:
                edges.append((i, right_child))

        return edges

    @staticmethod
    def _create_ladder(n: int) -> List[Tuple[int, int]]:
        """Create a ladder topology."""
        if n < 4:
            return [(i, i + 1) for i in range(n - 1)]

        edges = []
        rungs = n // 2

        for i in range(rungs - 1):
            edges.append((i, i + 1))
            edges.append((i + rungs, i + rungs + 1))

        for i in range(rungs):
            if i + rungs < n:
                edges.append((i, i + rungs))

        return edges

    @staticmethod
    def _create_wheel(n: int) -> List[Tuple[int, int]]:
        """Create a wheel topology (star + ring)."""
        if n < 4:
            return [(0, i) for i in range(1, n)]

        edges = []
        for i in range(1, n):
            edges.append((0, i))

        for i in range(1, n - 1):
            edges.append((i, i + 1))
        edges.append((n - 1, 1))

        return edges

    @staticmethod
    def _create_random_regular(n: int, degree: int = 3) -> List[Tuple[int, int]]:
        """Create a random regular graph."""
        if degree >= n or (n * degree) % 2 != 0:
            degree = min(degree, n - 1)
            if (n * degree) % 2 != 0:
                degree -= 1

        try:
            G = nx.random_regular_graph(degree, n)
            return list(G.edges())
        except:
            edges = []
            for i in range(n):
                for j in range(1, degree + 1):
                    neighbor = (i + j) % n
                    if i < neighbor:
                        edges.append((i, neighbor))
            return edges

    @staticmethod
    def _create_small_world(n: int, k: int = 4, p: float = 0.3) -> List[Tuple[int, int]]:
        """Create a small-world network (Watts-Strogatz model)."""
        if k >= n:
            k = n - 1

        try:
            G = nx.watts_strogatz_graph(n, k, p)
            return list(G.edges())
        except:
            edges = [(i, (i + 1) % n) for i in range(n)]
            for _ in range(min(n // 4, 10)):
                i, j = random.sample(range(n), 2)
                if (i, j) not in edges and (j, i) not in edges:
                    edges.append((i, j))
            return edges

    @staticmethod
    def _create_scale_free(n: int, m: int = 2) -> List[Tuple[int, int]]:
        """Create a scale-free network (Barabasi-Albert model)."""
        if m >= n:
            m = max(1, n // 3)

        try:
            G = nx.barabasi_albert_graph(n, m)
            return list(G.edges())
        except:
            edges = []
            degrees = [0] * n

            for i in range(min(3, n)):
                for j in range(i + 1, min(3, n)):
                    edges.append((i, j))
                    degrees[i] += 1
                    degrees[j] += 1

            for i in range(3, n):
                candidates = list(range(i))
                probs = [degrees[j] + 1 for j in candidates]
                total_prob = sum(probs)
                probs = [p / total_prob for p in probs]

                num_connections = min(m, i)
                chosen = np.random.choice(candidates, size=num_connections,
                                          replace=False, p=probs)

                for j in chosen:
                    edges.append((j, i))
                    degrees[j] += 1
                    degrees[i] += 1

            return edges

    @staticmethod
    def _create_hypercube(n: int) -> List[Tuple[int, int]]:
        """Create a hypercube topology."""
        dim = int(np.log2(n)) if n > 0 else 0
        actual_n = 2 ** dim

        if actual_n != n:
            print(f"Warning: Hypercube requires 2^k nodes. Using {actual_n} nodes instead of {n}")

        edges = []
        for i in range(actual_n):
            for j in range(i + 1, actual_n):
                if bin(i ^ j).count('1') == 1:
                    edges.append((i, j))

        return edges

    @staticmethod
    def _create_triangular_lattice(n: int) -> List[Tuple[int, int]]:
        """Create a triangular lattice topology."""
        rows = int(np.sqrt(n))
        cols = n // rows
        if rows * cols < n:
            cols += 1

        edges = []
        for i in range(rows):
            for j in range(cols):
                node = i * cols + j
                if node >= n:
                    break

                if j + 1 < cols and node + 1 < n:
                    edges.append((node, node + 1))
                if i + 1 < rows and node + cols < n:
                    edges.append((node, node + cols))

                if i + 1 < rows and j + 1 < cols and node + cols + 1 < n:
                    edges.append((node, node + cols + 1))

        return edges

    def create_custom_topology_from_function(self, topology_func: Callable[[int], List[Tuple[int, int]]]) -> nx.Graph:
        """Create a custom topology from a user-defined function."""
        self.graph.clear()
        self.graph.add_nodes_from(range(self.num_qubits))

        edges = topology_func(self.num_qubits)
        self.graph.add_edges_from(edges)

        return self.graph

    def create_custom_topology_from_adjacency_matrix(self, adj_matrix: np.ndarray) -> nx.Graph:
        """Create a custom topology from an adjacency matrix."""
        if adj_matrix.shape != (self.num_qubits, self.num_qubits):
            raise ValueError(f"Adjacency matrix must be {self.num_qubits}x{self.num_qubits}")

        self.graph.clear()
        self.graph.add_nodes_from(range(self.num_qubits))

        edges = []
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if adj_matrix[i, j] != 0:
                    edges.append((i, j))

        self.graph.add_edges_from(edges)
        return self.graph

    def create_random_topology(self, edge_probability: float = 0.3,
                               min_edges: Optional[int] = None,
                               max_edges: Optional[int] = None) -> nx.Graph:
        """Create a random topology with specified edge probability."""
        self.graph.clear()
        self.graph.add_nodes_from(range(self.num_qubits))

        edges = []
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if random.random() < edge_probability:
                    edges.append((i, j))

        if min_edges and len(edges) < min_edges:
            possible_edges = [(i, j) for i in range(self.num_qubits)
                              for j in range(i + 1, self.num_qubits)]
            remaining_edges = [e for e in possible_edges if e not in edges]

            additional_needed = min(min_edges - len(edges), len(remaining_edges))
            additional_edges = random.sample(remaining_edges, additional_needed)
            edges.extend(additional_edges)

        if max_edges and len(edges) > max_edges:
            edges = random.sample(edges, max_edges)

        self.graph.add_edges_from(edges)
        return self.graph

    def create_topology(self, graph_type: str = "complete",
                        custom_edges: Optional[List[Tuple]] = None,
                        **kwargs) -> nx.Graph:
        """Enhanced topology creation with support for new topologies."""
        self.graph.clear()
        self.graph.add_nodes_from(range(self.num_qubits))

        if graph_type == "custom" and custom_edges:
            self.graph.add_edges_from(custom_edges)
        elif graph_type in self.TOPOLOGY_BUILDERS:
            try:
                edges = self.TOPOLOGY_BUILDERS[graph_type](self.num_qubits)
                self.graph.add_edges_from(edges)
            except Exception as e:
                print(f"Warning: Could not create {graph_type} topology: {e}")
                print("Falling back to complete graph")
                edges = self.TOPOLOGY_BUILDERS["complete"](self.num_qubits)
                self.graph.add_edges_from(edges)
        else:
            raise ValueError(f"Unknown graph type '{graph_type}' or missing custom_edges")

        return self.graph

    def create_graph_state(self, graph_type: str = "complete") -> np.ndarray:
        """Create a graph state with specified topology."""
        if not self.graph.edges():
            self.create_topology(graph_type)

        state = self.create_state_vector()
        state = self.apply_hadamard_to_all(state)

        for i, j in self.graph.edges():
            state = self.apply_cz_gate(state, i, j)

        self.state = state
        return state

    def get_topology_info(self) -> Dict:
        """Get detailed information about the current topology."""
        if not self.graph:
            return {"error": "No graph created"}

        info = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "average_clustering": nx.average_clustering(self.graph) if self.graph.number_of_edges() > 0 else 0,
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else "infinite",
            "average_path_length": nx.average_shortest_path_length(self.graph) if nx.is_connected(
                self.graph) else "infinite",
            "degree_sequence": sorted([d for n, d in self.graph.degree()], reverse=True),
            "edge_list": list(self.graph.edges())
        }

        return info

    def visualize_topology(self, title: str = "Graph Topology",
                           figsize: Tuple[int, int] = (8, 6),
                           layout: str = "auto", save_path: str = None) -> None:
        """Enhanced visualization with better layout options."""
        if not self.graph:
            raise ValueError("Graph not created yet.")

        plt.figure(figsize=figsize)

        if layout == "auto":
            if self.graph.number_of_edges() == 0:
                pos = nx.random_layout(self.graph)
            elif nx.is_tree(self.graph):
                pos = nx.spring_layout(self.graph, k=2, iterations=50)
            elif self.graph.number_of_edges() / (self.num_qubits * (self.num_qubits - 1) / 2) > 0.7:
                pos = nx.circular_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "spring":
            pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        elif layout == "shell":
            pos = nx.shell_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                               node_size=500, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray',
                               width=2, alpha=0.7)
        nx.draw_networkx_labels(self.graph, pos, font_size=10,
                                font_weight='bold')

        plt.title(f"{title}\nNodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Graph saved to {save_path}")

        plt.show()


class PauliNoiseModel:
    """Model for applying Pauli noise to quantum states."""

    def __init__(self):
        self.operators = PauliOperators.get_operators()
        self.last_operations = []

    def apply_single_operator(self, state: np.ndarray, op_name: str,
                              target_qubit: int, num_qubits: int) -> np.ndarray:
        """Apply a single Pauli operator to a specific qubit."""
        op = self.operators[op_name]
        n = num_qubits
        full_op = np.eye(2 ** n, dtype=complex)

        if op[0, 1] == 0 and op[1, 0] == 0:  # Diagonal operator (Z)
            for i in range(2 ** n):
                if ((i >> target_qubit) & 1) == 0:
                    full_op[i, i] *= op[0, 0]
                else:
                    full_op[i, i] *= op[1, 1]
        else:  # Off-diagonal operator (X, Y)
            for i in range(2 ** n):
                if (i >> target_qubit) & 1:
                    new_i = i ^ (1 << target_qubit)
                    full_op[i, new_i] = op[1, 0]
                    full_op[new_i, i] = op[0, 1]

        return full_op @ state

    def apply_noise(self, state: np.ndarray, num_qubits: int,
                    n_errors: int, probability: float,
                    allow_repetition: bool = False) -> np.ndarray:
        """Apply Pauli noise to a quantum state."""
        if np.random.random() >= probability:
            self.last_operations = []
            return state.copy()

        n_errors = min(n_errors, num_qubits) if not allow_repetition else n_errors

        operators = list(self.operators.keys())
        chosen_ops = np.random.choice(operators, size=n_errors)
        qubits = np.random.choice(num_qubits, size=n_errors, replace=allow_repetition)

        self.last_operations = list(zip(qubits, chosen_ops))

        noisy_state = state.copy()
        for op, qubit in zip(chosen_ops, qubits):
            noisy_state = self.apply_single_operator(noisy_state, op, qubit, num_qubits)

        return noisy_state


class QuantumStateMetrics:
    """Metrics for quantum states and graph states."""

    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate fidelity between two quantum states."""
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        normalization = (np.vdot(state1, state1).real * np.vdot(state2, state2).real)
        return overlap / normalization

    @staticmethod
    def frobenius_distance(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate Frobenius distance between density matrices."""
        rho1 = np.outer(state1, state1.conj())
        rho2 = np.outer(state2, state2.conj())
        return np.linalg.norm(rho1 - rho2, ord='fro')

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        """Calculate von Neumann entropy of a density matrix."""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    @staticmethod
    def reduced_density_matrix(rho: np.ndarray, qubits_to_keep: List[int],
                               num_qubits: int) -> np.ndarray:
        """Compute reduced density matrix for specified qubits."""
        keep_dims = 2 ** len(qubits_to_keep)
        reduced_rho = np.zeros((keep_dims, keep_dims), dtype=complex)

        qubits_to_trace = [q for q in range(num_qubits) if q not in qubits_to_keep]

        for i, j in product(range(keep_dims), repeat=2):
            temp_sum = 0

            for k in range(2 ** len(qubits_to_trace)):
                full_i = full_j = 0

                for idx, q in enumerate(qubits_to_keep):
                    bit_i = (i >> idx) & 1
                    bit_j = (j >> idx) & 1
                    full_i |= bit_i << q
                    full_j |= bit_j << q

                for idx, q in enumerate(qubits_to_trace):
                    bit = (k >> idx) & 1
                    full_i |= bit << q
                    full_j |= bit << q

                temp_sum += rho[full_i, full_j]

            reduced_rho[i, j] = temp_sum

        return reduced_rho

    @classmethod
    def concurrence(cls, state_vector: np.ndarray, qubits: Tuple[int, int],
                    num_qubits: int) -> float:
        """Calculate concurrence between two qubits."""
        if len(qubits) != 2:
            raise ValueError("Concurrence requires exactly two qubits")

        rho = np.outer(state_vector, state_vector.conj())
        reduced_rho = cls.reduced_density_matrix(rho, list(qubits), num_qubits)

        purity = np.trace(reduced_rho @ reduced_rho).real
        concurrence_approx = max(0, 2 * np.sqrt(max(0, purity - 0.25)))
        return min(concurrence_approx, 1.0)

    @classmethod
    def mutual_information(cls, state_vector: np.ndarray, qubit_a: int,
                           qubit_b: int, num_qubits: int) -> float:
        """Calculate quantum mutual information between two qubits."""
        rho = np.outer(state_vector, state_vector.conj())

        rho_a = cls.reduced_density_matrix(rho, [qubit_a], num_qubits)
        rho_b = cls.reduced_density_matrix(rho, [qubit_b], num_qubits)
        rho_ab = cls.reduced_density_matrix(rho, [qubit_a, qubit_b], num_qubits)

        s_a = cls.von_neumann_entropy(rho_a)
        s_b = cls.von_neumann_entropy(rho_b)
        s_ab = cls.von_neumann_entropy(rho_ab)

        return max(0, s_a + s_b - s_ab)

    @classmethod
    def average_entanglement(cls, state_vector: np.ndarray, graph: nx.Graph,
                             num_qubits: int) -> float:
        """Calculate average entanglement across all edges in the graph."""
        if not graph.edges():
            return 0.0

        total_entanglement = sum(
            cls.concurrence(state_vector, edge, num_qubits)
            for edge in graph.edges()
        )
        return total_entanglement / len(graph.edges())

    @classmethod
    def graph_sensitivity(cls, original_state: np.ndarray, noisy_state: np.ndarray,
                          graph: nx.Graph, num_qubits: int) -> float:
        """Calculate the sensitivity of a graph to noise."""
        if not graph.edges():
            return 0.0

        original_ent = cls.average_entanglement(original_state, graph, num_qubits)
        noisy_ent = cls.average_entanglement(noisy_state, graph, num_qubits)

        if original_ent == 0:
            return 0.0

        return abs(original_ent - noisy_ent) / original_ent


class WeatherNoiseModel:
    """Model for different weather-based noise distributions."""

    DEFAULT_PARAMS = {
        'rain': {'lambda_param': 2.0, 'attenuation': 0.7},
        'dust': {'shape': 1.5, 'scale': 0.8},
        'turbulence': {'mu': 0.0, 'sigma': 0.5},
        'clear': {'base_error': 0.01}
    }

    OPERATOR_DISTRIBUTIONS = {
        'rain': {'X': 0.33, 'Y': 0.33, 'Z': 0.34},
        'dust': {'X': 0.33, 'Y': 0.33, 'Z': 0.34},
        'turbulence': {'X': 0.33, 'Y': 0.33, 'Z': 0.34},
        'clear': {'X': 0.33, 'Y': 0.33, 'Z': 0.34}
    }

    def __init__(self, model_type: str, params: Optional[Dict] = None):
        self.model_type = model_type.lower()
        if self.model_type not in self.DEFAULT_PARAMS:
            raise ValueError(f"Unknown weather model: {model_type}")

        self.params = params or self.DEFAULT_PARAMS[self.model_type].copy()

    def error_probability(self, intensity: float = 1.0) -> float:
        """Get error probability based on weather model and intensity."""
        if self.model_type == 'rain':
            lambda_param = self.params['lambda_param'] * intensity
            num_errors = np.random.poisson(lambda_param)
            prob = 1.0 - (self.params['attenuation'] ** num_errors)

        elif self.model_type == 'dust':
            shape = self.params['shape']
            scale = self.params['scale'] * intensity
            prob = np.random.weibull(shape) * scale

        elif self.model_type == 'turbulence':
            mu = self.params['mu']
            sigma = self.params['sigma'] * intensity
            prob = np.random.lognormal(mu, sigma) / 5.0

        elif self.model_type == 'clear':
            base_error = self.params['base_error'] * intensity
            prob = base_error * (1 + 0.5 * np.random.random())

        return min(prob, 1.0)

    def operator_distribution(self) -> Dict[str, float]:
        """Get the distribution of Pauli operators for this weather model."""
        return self.OPERATOR_DISTRIBUTIONS.get(self.model_type,
                                               self.OPERATOR_DISTRIBUTIONS['clear'])


class GraphStateSimulator:
    """Enhanced simulator for graph states with custom topologies and noise."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.builder = EnhancedGraphStateBuilder(num_qubits)
        self.noise_model = PauliNoiseModel()
        self.original_state = None
        self.noisy_state = None

    def create_graph_state(self, graph_type: str = "complete",
                           custom_edges: Optional[List[Tuple]] = None) -> np.ndarray:
        """Enhanced graph state creation with custom topology support."""
        if graph_type == "custom" and custom_edges:
            self.original_state = self.create_graph_state_from_edges(custom_edges)
        else:
            self.original_state = self.builder.create_graph_state(graph_type)

        return self.original_state

    def create_graph_state_from_edges(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """Create graph state directly from edge list."""
        self.builder.graph.clear()
        self.builder.graph.add_nodes_from(range(self.num_qubits))
        self.builder.graph.add_edges_from(edges)

        state = self.builder.create_state_vector()
        state = self.builder.apply_hadamard_to_all(state)

        for i, j in self.builder.graph.edges():
            state = self.builder.apply_cz_gate(state, i, j)

        self.builder.state = state
        self.original_state = state
        return state

    def apply_pauli_noise(self, n_errors: int, probability: float,
                          allow_repetition: bool = False) -> np.ndarray:
        """Apply standard Pauli noise."""
        if self.original_state is None:
            raise ValueError("Create graph state first!")

        self.noisy_state = self.noise_model.apply_noise(
            self.original_state, self.num_qubits,
            n_errors, probability, allow_repetition
        )
        return self.noisy_state

    def apply_weather_noise(self, weather_model, intensity: float = 1.0) -> np.ndarray:
        """Apply weather-based noise to the graph state."""
        if self.original_state is None:
            raise ValueError("Create graph state first!")

        error_prob = weather_model.error_probability(intensity)

        if np.random.random() >= error_prob:
            self.noisy_state = self.original_state.copy()
            self.noise_model.last_operations = []
            return self.noisy_state

        n_errors = np.random.randint(1, self.num_qubits + 1)
        op_dist = weather_model.operator_distribution()
        operators = list(op_dist.keys())
        op_probs = list(op_dist.values())

        chosen_ops = np.random.choice(operators, size=n_errors, p=op_probs)
        qubits = np.random.choice(self.num_qubits, size=n_errors, replace=False)

        self.noise_model.last_operations = list(zip(qubits, chosen_ops))

        state = self.original_state.copy()
        for op, qubit in zip(chosen_ops, qubits):
            state = self.noise_model.apply_single_operator(state, op, qubit, self.num_qubits)

        self.noisy_state = state
        return self.noisy_state

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all relevant metrics for the current states."""
        if self.original_state is None or self.noisy_state is None:
            raise ValueError("Both original and noisy states must exist!")

        metrics = {
            'fidelity': QuantumStateMetrics.fidelity(self.original_state, self.noisy_state),
            'frobenius_similarity': QuantumStateMetrics.frobenius_distance(
                self.original_state, self.noisy_state
            ),
            'avg_entanglement_original': QuantumStateMetrics.average_entanglement(
                self.original_state, self.builder.graph, self.num_qubits
            ),
            'avg_entanglement_noisy': QuantumStateMetrics.average_entanglement(
                self.noisy_state, self.builder.graph, self.num_qubits
            ),
            'sensitivity': QuantumStateMetrics.graph_sensitivity(
                self.original_state, self.noisy_state,
                self.builder.graph, self.num_qubits
            )
        }

        metrics['entanglement_change'] = (
                metrics['avg_entanglement_original'] - metrics['avg_entanglement_noisy']
        )

        if self.builder.graph.edges():
            edge = list(self.builder.graph.edges())[0]
            metrics['mutual_info'] = QuantumStateMetrics.mutual_information(
                self.noisy_state, edge[0], edge[1], self.num_qubits
            )
        else:
            metrics['mutual_info'] = 0

        # Add topology-specific metrics
        topology_info = self.builder.get_topology_info()
        metrics['graph_density'] = topology_info['density']
        metrics['graph_clustering'] = topology_info['average_clustering']
        metrics['is_connected'] = int(topology_info['is_connected'])

        return metrics

    def visualize_graph(self, title: str = "Graph State",
                        figsize: Tuple[int, int] = (8, 6), save_path: str = None) -> None:
        """Visualize the graph state with noise indicators."""
        if not self.builder.graph:
            raise ValueError("Graph state not created yet.")

        plt.figure(figsize=figsize)

        if nx.is_tree(self.builder.graph):
            pos = nx.shell_layout(self.builder.graph)
        elif nx.is_isomorphic(self.builder.graph, nx.cycle_graph(self.num_qubits)):
            pos = nx.circular_layout(self.builder.graph)
        else:
            pos = nx.spring_layout(self.builder.graph, seed=42)

        noisy_nodes = [q for q, _ in self.noise_model.last_operations]
        node_colors = ['red' if node in noisy_nodes else 'skyblue'
                       for node in self.builder.graph.nodes()]

        nx.draw_networkx_nodes(self.builder.graph, pos,
                               node_color=node_colors, node_size=700)
        nx.draw_networkx_edges(self.builder.graph, pos)
        nx.draw_networkx_labels(self.builder.graph, pos, font_size=12)

        if noisy_nodes:
            labels = {q: op for q, op in self.noise_model.last_operations}
            nx.draw_networkx_labels(self.builder.graph, pos, labels=labels,
                                    font_size=12, font_color='black')

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Graph state visualization saved to {save_path}")

        plt.show()


def define_custom_topologies(num_qubits: int) -> Dict[str, List[Tuple[int, int]]]:
    """Define several custom topologies for testing."""

    topologies = {}

    # 1. Ring with shortcuts (small-world-like)
    ring_edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    shortcuts = [(0, num_qubits // 2)] if num_qubits > 3 else []
    topologies["ring_with_shortcuts"] = ring_edges + shortcuts

    # 2. Double star (two stars connected)
    if num_qubits >= 6:
        center1, center2 = 0, 1
        star1_edges = [(center1, i) for i in range(2, num_qubits // 2 + 1)]
        star2_edges = [(center2, i) for i in range(num_qubits // 2 + 1, num_qubits)]
        bridge = [(center1, center2)]
        topologies["double_star"] = star1_edges + star2_edges + bridge

    # 3. Path with branches
    path_edges = [(i, i + 1) for i in range(num_qubits - 1)]
    if num_qubits > 4:
        branches = [(num_qubits // 3, num_qubits - 1)] if num_qubits > 5 else []
        topologies["branched_path"] = path_edges + branches
    else:
        topologies["branched_path"] = path_edges

    # 4. Modular topology
    module_size = 3
    modules = []
    for module_start in range(0, num_qubits, module_size):
        module_end = min(module_start + module_size, num_qubits)
        for i in range(module_start, module_end):
            for j in range(i + 1, module_end):
                modules.append((i, j))

    inter_module = []
    for i in range(0, num_qubits - module_size, module_size):
        if i + module_size < num_qubits:
            inter_module.append((i, i + module_size))

    topologies["modular"] = modules + inter_module

    # 5. Random sparse topology
    all_possible = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    num_edges = max(num_qubits - 1, len(all_possible) // 3)
    random.seed(42)
    sparse_edges = random.sample(all_possible, min(num_edges, len(all_possible)))
    topologies["sparse_random"] = sparse_edges

    return topologies


# ================================
# ENHANCED VISUALIZATION FUNCTIONS
# ================================

def visualize_all_topologies_for_all_qubits(save_to_results: bool = True) -> None:
    """Visualize all topologies for each qubit count used in simulation."""
    print("=== Visualizing All Topologies for Each Qubit Count ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for num_qubits in QUBIT_RANGES:
        print(f"Creating topology visualizations for {num_qubits} qubits...")

        # Standard topologies (excluding custom ones for this visualization)
        standard_topologies = [t for t in GRAPH_TYPES if not t.startswith("custom_")]

        # Custom topologies
        custom_topologies = define_custom_topologies(num_qubits)

        # Calculate grid size for subplots
        total_topologies = len(standard_topologies) + len(custom_topologies)
        cols = 4
        rows = (total_topologies + cols - 1) // cols

        # Create figure for this qubit count
        fig, axes = plt.subplots(rows, cols, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle(f'All Graph Topologies with {num_qubits} Qubits', fontsize=16)

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        plot_idx = 0

        # Plot standard topologies
        for topology in standard_topologies:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            builder = EnhancedGraphStateBuilder(num_qubits)

            try:
                builder.create_topology(topology)

                # Choose optimal layout for each topology
                pos = get_optimal_layout(builder.graph, topology)

                # Draw the graph
                nx.draw_networkx_nodes(builder.graph, pos, node_color='lightblue',
                                       node_size=400, ax=ax, alpha=0.8)
                nx.draw_networkx_edges(builder.graph, pos, edge_color='gray',
                                       width=1.5, ax=ax, alpha=0.7)
                nx.draw_networkx_labels(builder.graph, pos, font_size=8, ax=ax)

                info = builder.get_topology_info()
                ax.set_title(f'{topology.replace("_", " ").title()}\n'
                             f'Edges: {info["num_edges"]}, Connected: {str(info["is_connected"])[0]}',
                             fontsize=9)
                ax.axis('off')

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{topology}", ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(f'{topology} (Failed)', fontsize=9)
                ax.axis('off')

            plot_idx += 1

        # Plot custom topologies
        for name, edges in custom_topologies.items():
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            # Create graph
            G = nx.Graph()
            G.add_nodes_from(range(num_qubits))
            G.add_edges_from(edges)

            # Choose layout
            pos = get_optimal_layout(G, name)

            # Draw
            nx.draw_networkx_nodes(G, pos, node_color='lightcoral',
                                   node_size=400, ax=ax, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray',
                                   width=1.5, ax=ax, alpha=0.7)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

            ax.set_title(f"{name.replace('_', ' ').title()}\n"
                         f"E: {G.number_of_edges()}, C: {str(nx.is_connected(G))[0]}",
                         fontsize=9)
            ax.axis('off')

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_to_results:
            os.makedirs(RESULTS_FOLDER, exist_ok=True)
            save_path = f"{RESULTS_FOLDER}/all_topologies_{num_qubits}_qubits_{timestamp}.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"  Saved topology overview for {num_qubits} qubits to {save_path}")

        plt.show()


def get_optimal_layout(graph: nx.Graph, topology_name: str) -> Dict:
    """Get optimal layout for a specific topology."""
    num_nodes = graph.number_of_nodes()

    if "grid" in topology_name.lower():
        rows = int(np.sqrt(num_nodes))
        cols = num_nodes // rows
        if rows * cols < num_nodes:
            cols += 1
        pos = {i: (i % cols, -(i // cols)) for i in range(num_nodes)}
    elif "ring" in topology_name.lower() or "wheel" in topology_name.lower():
        pos = nx.circular_layout(graph)
    elif "tree" in topology_name.lower():
        pos = nx.spring_layout(graph, k=2, iterations=50)
    elif "ladder" in topology_name.lower():
        rungs = num_nodes // 2
        pos = {}
        for i in range(rungs):
            pos[i] = (i, 0)
            if i + rungs < num_nodes:
                pos[i + rungs] = (i, 1)
    elif "star" in topology_name.lower():
        pos = nx.spring_layout(graph, k=2)
    elif "complete" in topology_name.lower():
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, k=1.5, iterations=50)

    return pos


def create_topology_comparison_matrix(save_to_results: bool = True) -> None:
    """Create a comparison matrix showing topology properties."""
    print("=== Creating Topology Comparison Matrix ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect data for all topologies and qubit counts
    topology_data = []

    for num_qubits in QUBIT_RANGES:
        # Standard topologies
        standard_topologies = [t for t in GRAPH_TYPES if not t.startswith("custom_")]
        custom_topologies = define_custom_topologies(num_qubits)

        all_topologies = standard_topologies + [f"custom_{name}" for name in custom_topologies.keys()]

        for topology in all_topologies:
            try:
                builder = EnhancedGraphStateBuilder(num_qubits)

                if topology.startswith("custom_"):
                    custom_name = topology.replace("custom_", "")
                    if custom_name in custom_topologies:
                        edges = custom_topologies[custom_name]
                        builder.graph.clear()
                        builder.graph.add_nodes_from(range(num_qubits))
                        builder.graph.add_edges_from(edges)
                else:
                    builder.create_topology(topology)

                info = builder.get_topology_info()

                topology_data.append({
                    'topology': topology,
                    'num_qubits': num_qubits,
                    'num_edges': info['num_edges'],
                    'density': info['density'],
                    'is_connected': info['is_connected'],
                    'clustering': info['average_clustering'],
                    'max_degree': max(dict(builder.graph.degree()).values()) if builder.graph.degree() else 0,
                    'min_degree': min(dict(builder.graph.degree()).values()) if builder.graph.degree() else 0,
                })

            except Exception as e:
                print(f"Error processing {topology} with {num_qubits} qubits: {e}")

    # Create DataFrame
    df = pd.DataFrame(topology_data)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Topology Properties Comparison Matrix', fontsize=16)

    axes = axes.flatten()

    metrics = ['num_edges', 'density', 'clustering', 'max_degree', 'min_degree']

    for idx, metric in enumerate(metrics):
        if idx >= len(axes) - 1:  # Reserve last subplot for connectivity
            break

        # Create pivot table for heatmap
        pivot_data = df.pivot_table(values=metric, index='topology', columns='num_qubits', aggfunc='mean')

        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f', ax=axes[idx])
        axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        axes[idx].set_xlabel('Number of Qubits')
        axes[idx].set_ylabel('Topology')

    # Connectivity matrix (binary)
    connectivity_pivot = df.pivot_table(values='is_connected', index='topology',
                                        columns='num_qubits', aggfunc='mean')
    sns.heatmap(connectivity_pivot, annot=True, cmap='RdYlGn', fmt='.0f',
                ax=axes[5], cbar_kws={'label': 'Connected (1) / Disconnected (0)'})
    axes[5].set_title('Graph Connectivity')
    axes[5].set_xlabel('Number of Qubits')
    axes[5].set_ylabel('Topology')

    plt.tight_layout()

    if save_to_results:
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        save_path = f"{RESULTS_FOLDER}/topology_comparison_matrix_{timestamp}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved topology comparison matrix to {save_path}")

        # Also save the data as CSV
        csv_path = f"{RESULTS_FOLDER}/topology_properties_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved topology properties data to {csv_path}")

    plt.show()

    return df


class MonteCarloSimulator:
    """Enhanced Monte Carlo simulation with support for custom topologies."""

    def __init__(self, results_dir: str = RESULTS_FOLDER):
        self.results_dir = results_dir
        self.results_df = None
        os.makedirs(results_dir, exist_ok=True)
        self.custom_topologies = {}

    def add_custom_topology(self, name: str, edges: List[Tuple[int, int]]):
        """Add a custom topology from an edge list."""
        self.custom_topologies[name] = edges

    def run_simulation(self, graph_types: List[str] = None,
                       qubit_ranges: List[int] = None,
                       weather_models: List[str] = None,
                       intensity_range: np.ndarray = None,
                       num_trials: int = None,
                       custom_topologies: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> pd.DataFrame:
        """Run Monte Carlo simulation with global parameters as defaults."""

        # Use global parameters if not specified
        graph_types = graph_types or GRAPH_TYPES
        qubit_ranges = qubit_ranges or QUBIT_RANGES
        weather_models = weather_models or WEATHER_MODELS
        intensity_range = intensity_range if intensity_range is not None else INTENSITY_RANGE
        num_trials = num_trials or NUM_TRIALS

        if custom_topologies:
            for name, edges in custom_topologies.items():
                self.add_custom_topology(name, edges)

        results = []
        total_sims = (len(graph_types) * len(qubit_ranges) *
                      len(weather_models) * len(intensity_range) * num_trials)

        with tqdm(total=total_sims, desc="Running Enhanced Monte Carlo Simulations") as pbar:
            for graph_type in graph_types:
                for num_qubits in qubit_ranges:
                    # Get custom topologies for this number of qubits
                    current_custom = define_custom_topologies(num_qubits)

                    simulator = GraphStateSimulator(num_qubits)

                    for weather_type in weather_models:
                        weather_model = WeatherNoiseModel(weather_type)

                        for intensity in intensity_range:
                            for trial in range(num_trials):
                                try:
                                    # Create graph state
                                    if graph_type.startswith("custom_"):
                                        topology_name = graph_type.replace("custom_", "")
                                        if topology_name in current_custom:
                                            edges = current_custom[topology_name]
                                            simulator.create_graph_state_from_edges(edges)
                                        elif topology_name in self.custom_topologies:
                                            edges = [(i, j) for i, j in self.custom_topologies[topology_name]
                                                     if i < num_qubits and j < num_qubits]
                                            simulator.create_graph_state_from_edges(edges)
                                        else:
                                            print(f"Unknown custom topology: {topology_name}")
                                            continue
                                    else:
                                        simulator.create_graph_state(graph_type)

                                    # Apply weather noise
                                    simulator.apply_weather_noise(weather_model, intensity)

                                    # Calculate metrics
                                    metrics = simulator.calculate_metrics()

                                    # Store results
                                    result = {
                                        'graph_type': graph_type,
                                        'num_qubits': num_qubits,
                                        'weather_type': weather_type,
                                        'intensity': intensity,
                                        'trial': trial,
                                        'num_errors': len(simulator.noise_model.last_operations),
                                        'resilience': 1 if metrics['fidelity'] >= 0.9 else 0,
                                        **metrics
                                    }
                                    results.append(result)

                                except Exception as e:
                                    print(f"Error in simulation: {graph_type}, {num_qubits} qubits, trial {trial}: {e}")
                                    result = {
                                        'graph_type': graph_type,
                                        'num_qubits': num_qubits,
                                        'weather_type': weather_type,
                                        'intensity': intensity,
                                        'trial': trial,
                                        'num_errors': 0,
                                        'resilience': 0,
                                        'fidelity': 0,
                                        'frobenius_similarity': float('inf'),
                                        'avg_entanglement_original': 0,
                                        'avg_entanglement_noisy': 0,
                                        'sensitivity': 0,
                                        'entanglement_change': 0,
                                        'mutual_info': 0,
                                        'graph_density': 0,
                                        'graph_clustering': 0,
                                        'is_connected': 0
                                    }
                                    results.append(result)

                                pbar.update(1)

        self.results_df = pd.DataFrame(results)
        self._save_results()
        return self.results_df

    def _save_results(self) -> None:
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/weather_simulation_results_{timestamp}.csv"
        self.results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def analyze_topology_performance(self) -> Dict:
        """Analyze performance of different topologies."""
        if self.results_df is None:
            raise ValueError("No results available. Run simulations first.")

        topology_stats = self.results_df.groupby('graph_type').agg({
            'fidelity': ['mean', 'std'],
            'sensitivity': ['mean', 'std'],
            'resilience': 'mean',
            'graph_density': 'mean',
            'graph_clustering': 'mean',
            'is_connected': 'mean'
        }).round(4)

        topology_stats.columns = ['_'.join(col).strip() for col in topology_stats.columns]

        return topology_stats.to_dict()

    def compare_topologies(self, metric: str = 'fidelity', save_plot: bool = True) -> None:
        """Compare topologies across a specific metric."""
        if self.results_df is None:
            raise ValueError("No results available. Run simulations first.")

        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        sns.boxplot(data=self.results_df, x='graph_type', y=metric, hue='weather_type')

        plt.xticks(rotation=45, ha='right')
        plt.title(f'Comparison of Graph Topologies by {metric.replace("_", " ").title()}')
        plt.xlabel('Graph Topology')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend(title='Weather Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.results_dir}/topology_comparison_{metric}_{timestamp}.png",
                        dpi=FIGURE_DPI, bbox_inches='tight')
        plt.show()

    def analyze_results(self, save_plots: bool = True) -> Dict:
        """Analyze and visualize simulation results."""
        if self.results_df is None:
            raise ValueError("No results available. Run simulations first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._plot_sensitivity_heatmap(save_plots, timestamp)
        self._plot_fidelity_by_weather(save_plots, timestamp)
        self._plot_fidelity_by_topology(save_plots, timestamp)
        self._plot_resilience_scores(save_plots, timestamp)
        self._plot_entanglement_change(save_plots, timestamp)
        self._plot_mutual_information(save_plots, timestamp)

        summary = self._calculate_summary_statistics()
        self._save_summary(summary, timestamp)

        return summary

    def _plot_sensitivity_heatmap(self, save: bool, timestamp: str) -> None:
        """Plot sensitivity heatmap by graph type and qubit count."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        sensitivity_pivot = (self.results_df.groupby(['graph_type', 'num_qubits'])
                             ['sensitivity'].mean().reset_index()
                             .pivot(index='graph_type', columns='num_qubits',
                                    values='sensitivity'))

        sns.heatmap(sensitivity_pivot, annot=True, cmap='YlGnBu', fmt=".3f")
        plt.title('Average Sensitivity by Graph Type and Qubit Count')
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.results_dir}/sensitivity_heatmap_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _plot_fidelity_by_weather(self, save: bool, timestamp: str) -> None:
        """Plot fidelity vs intensity by weather type."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        for weather_type in self.results_df['weather_type'].unique():
            weather_data = self.results_df[self.results_df['weather_type'] == weather_type]
            sns.lineplot(data=weather_data, x='intensity', y='fidelity',
                         label=weather_type, ci=95)

        plt.xlabel('Weather Intensity')
        plt.ylabel('Fidelity')
        plt.title('Fidelity vs. Weather Intensity by Weather Type')
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save:
            plt.savefig(f"{self.results_dir}/fidelity_by_weather_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _plot_fidelity_by_topology(self, save: bool, timestamp: str) -> None:
        """Plot fidelity vs intensity by graph topology."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        for topology in self.results_df['graph_type'].unique():
            topo_data = self.results_df[self.results_df['graph_type'] == topology]
            sns.lineplot(data=topo_data, x='intensity', y='fidelity',
                         label=topology, errorbar=('ci', 95))

        plt.xlabel('Weather Intensity')
        plt.ylabel('Fidelity')
        plt.title('Fidelity vs. Weather Intensity by Graph Topology')
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save:
            plt.savefig(f"{self.results_dir}/fidelity_by_topology_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _plot_resilience_scores(self, save: bool, timestamp: str) -> None:
        """Plot resilience scores by graph type and weather."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        resilience_data = (self.results_df.groupby(['graph_type', 'weather_type'])
                           ['resilience'].mean().reset_index())

        ax = sns.barplot(data=resilience_data, x='graph_type', y='resilience',
                         hue='weather_type')

        plt.xlabel('Graph Type')
        plt.ylabel('Resilience Score (Probability of Fidelity > 0.9)')
        plt.title('Resilience by Graph Type and Weather')
        plt.xticks(rotation=45)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.results_dir}/resilience_score_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _plot_entanglement_change(self, save: bool, timestamp: str) -> None:
        """Plot entanglement change by weather type and graph type."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        sns.boxplot(data=self.results_df, x='weather_type', y='entanglement_change',
                    hue='graph_type')

        plt.xlabel('Weather Type')
        plt.ylabel('Entanglement Change')
        plt.title('Entanglement Change by Weather Type and Graph Type')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.results_dir}/entanglement_change_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _plot_mutual_information(self, save: bool, timestamp: str) -> None:
        """Plot mutual information by intensity and parameters."""
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)

        sns.lineplot(data=self.results_df, x='intensity', y='mutual_info',
                     hue='num_qubits', style='weather_type', ci=95)

        plt.xlabel('Weather Intensity')
        plt.ylabel('Mutual Information')
        plt.title('Mutual Information vs. Weather Intensity by Qubit Count')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Qubits / Weather')

        if save:
            plt.savefig(f"{self.results_dir}/mutual_info_{timestamp}.png", dpi=FIGURE_DPI)
        plt.show()

    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics from simulation results."""
        summary = {
            'best_sensor_graph': self._find_best_sensor_graph(),
            'most_resilient_graph': self._find_most_resilient_graph(),
            'weather_impact': self._analyze_weather_impact(),
            'optimal_qubit_count': self._find_optimal_qubit_count()
        }

        self._print_summary(summary)
        return summary

    def _find_best_sensor_graph(self) -> Dict:
        """Find the graph with highest sensitivity (best for sensing)."""
        sensitivity_avg = (self.results_df.groupby(['graph_type', 'num_qubits'])
                           ['sensitivity'].mean().reset_index())
        best_sensor = sensitivity_avg.loc[sensitivity_avg['sensitivity'].idxmax()]

        return {
            'graph_type': best_sensor['graph_type'],
            'num_qubits': int(best_sensor['num_qubits']),
            'sensitivity': float(best_sensor['sensitivity'])
        }

    def _find_most_resilient_graph(self) -> Dict:
        """Find the graph with highest resilience (best for communication)."""
        resilience_avg = (self.results_df.groupby(['graph_type', 'num_qubits'])
                          ['resilience'].mean().reset_index())
        most_resilient = resilience_avg.loc[resilience_avg['resilience'].idxmax()]

        return {
            'graph_type': most_resilient['graph_type'],
            'num_qubits': int(most_resilient['num_qubits']),
            'resilience': float(most_resilient['resilience'])
        }

    def _analyze_weather_impact(self) -> Dict[str, str]:
        """Analyze the impact of different weather types."""
        weather_impact = {}

        for weather in self.results_df['weather_type'].unique():
            weather_data = self.results_df[self.results_df['weather_type'] == weather]
            avg_fidelity = weather_data['fidelity'].mean()

            if avg_fidelity > 0.9:
                impact = "Low impact"
            elif avg_fidelity > 0.7:
                impact = "Moderate impact"
            elif avg_fidelity > 0.5:
                impact = "High impact"
            else:
                impact = "Severe impact"

            weather_impact[weather] = impact

        return weather_impact

    def _find_optimal_qubit_count(self) -> Dict[str, Dict]:
        """Find optimal qubit counts for different purposes."""
        sensing_data = (self.results_df.groupby(['graph_type', 'num_qubits'])
                        ['sensitivity'].mean().reset_index())
        best_sensing = sensing_data.loc[sensing_data['sensitivity'].idxmax()]

        comm_data = (self.results_df.groupby(['graph_type', 'num_qubits'])
                     ['resilience'].mean().reset_index())
        best_comm = comm_data.loc[comm_data['resilience'].idxmax()]

        self.results_df['norm_sensitivity'] = (
                (self.results_df['sensitivity'] - self.results_df['sensitivity'].min()) /
                (self.results_df['sensitivity'].max() - self.results_df['sensitivity'].min())
        )
        self.results_df['balanced_score'] = (
                self.results_df['norm_sensitivity'] * 0.5 +
                self.results_df['resilience'] * 0.5
        )

        balanced_data = (self.results_df.groupby(['graph_type', 'num_qubits'])
                         ['balanced_score'].mean().reset_index())
        best_balanced = balanced_data.loc[balanced_data['balanced_score'].idxmax()]

        return {
            'sensing': {
                'count': int(best_sensing['num_qubits']),
                'graph_type': best_sensing['graph_type']
            },
            'communication': {
                'count': int(best_comm['num_qubits']),
                'graph_type': best_comm['graph_type']
            },
            'balanced': {
                'count': int(best_balanced['num_qubits']),
                'graph_type': best_balanced['graph_type']
            }
        }

    def _print_summary(self, summary: Dict) -> None:
        """Print formatted summary of results."""
        print("\n===== SIMULATION SUMMARY =====")
        print(f"Best sensor graph: {summary['best_sensor_graph']['graph_type']} with "
              f"{summary['best_sensor_graph']['num_qubits']} qubits")
        print(f"Most resilient graph: {summary['most_resilient_graph']['graph_type']} with "
              f"{summary['most_resilient_graph']['num_qubits']} qubits")

        print("\nWeather Impact Analysis:")
        for weather, impact in summary['weather_impact'].items():
            print(f"  {weather}: {impact}")

        print("\nOptimal qubit counts:")
        for purpose, details in summary['optimal_qubit_count'].items():
            print(f"  For {purpose}: {details['count']} qubits ({details['graph_type']} topology)")

    def _save_summary(self, summary: Dict, timestamp: str) -> None:
        """Save summary to text file."""
        filename = f"{self.results_dir}/simulation_summary_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write("===== SIMULATION SUMMARY =====\n\n")

            f.write(f"Best sensor graph: {summary['best_sensor_graph']['graph_type']} with "
                    f"{summary['best_sensor_graph']['num_qubits']} qubits\n")
            f.write(f"Most resilient graph: {summary['most_resilient_graph']['graph_type']} with "
                    f"{summary['most_resilient_graph']['num_qubits']} qubits\n\n")

            f.write("Weather Impact Analysis:\n")
            for weather, impact in summary['weather_impact'].items():
                f.write(f"  {weather}: {impact}\n")

            f.write("\nOptimal qubit counts:\n")
            for purpose, details in summary['optimal_qubit_count'].items():
                f.write(f"  For {purpose}: {details['count']} qubits ({details['graph_type']} topology)\n")

        print(f"Summary saved to {filename}")


# ================================
# USER-FRIENDLY ANALYSIS FUNCTIONS
# ================================

def run_basic_noise_analysis(num_qubits: int = None, n_range: range = None,
                             p_range: np.ndarray = None, samples: int = None,
                             save_results: bool = True) -> None:
    """Run basic Pauli noise analysis with global parameters."""

    # Use global parameters if not specified
    if num_qubits is None:
        num_qubits = QUBIT_RANGES[0] if QUBIT_RANGES else 5
    n_range = n_range or BASIC_NOISE_N_RANGE
    p_range = p_range if p_range is not None else BASIC_NOISE_P_RANGE
    samples = samples or BASIC_NOISE_SAMPLES

    print(f"=== Running Basic Noise Analysis ===")
    print(f"Qubits: {num_qubits}, Error range: {list(n_range)}, Samples: {samples}")

    results_fidelity = {}
    results_frobenius = {}

    for n in n_range:
        print(f"Processing N={n}")
        fidelities = []
        frobenius_similarities = []

        for p in p_range:
            simulator = GraphStateSimulator(num_qubits)
            simulator.create_graph_state("complete")

            sample_fidelities = []
            sample_frobenius = []

            for _ in range(samples):
                simulator.apply_pauli_noise(n, p)
                metrics = simulator.calculate_metrics()
                sample_fidelities.append(metrics['fidelity'])
                sample_frobenius.append(metrics['frobenius_similarity'])

            fidelities.append(np.mean(sample_fidelities))
            frobenius_similarities.append(np.mean(sample_frobenius))

        results_fidelity[n] = fidelities
        results_frobenius[n] = frobenius_similarities

    plt.figure(figsize=FIGURE_SIZE_MEDIUM)

    plt.subplot(1, 2, 1)
    for n, fidelities in results_fidelity.items():
        plt.plot(p_range, fidelities, label=f'N={n}')
    plt.xlabel('Probability (p)')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Probability')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for n, frobenius in results_frobenius.items():
        plt.plot(p_range, frobenius, label=f'N={n}', linestyle='--')
    plt.xlabel('Probability (p)')
    plt.ylabel('Frobenius Similarity')
    plt.title('Frobenius Similarity vs Probability')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_results:
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{RESULTS_FOLDER}/basic_noise_analysis_{timestamp}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Basic noise analysis saved to {save_path}")

    plt.show()


def run_full_simulation() -> Tuple[pd.DataFrame, Dict]:
    """Run the complete simulation with all global parameters."""
    print("=== Running Full Enhanced Simulation ===")
    print(f"Qubit ranges: {QUBIT_RANGES}")
    print(f"Graph types: {len(GRAPH_TYPES)} types")
    print(f"Weather models: {WEATHER_MODELS}")
    print(f"Intensity range: {INTENSITY_RANGE.min():.2f} to {INTENSITY_RANGE.max():.2f}")
    print(f"Trials per configuration: {NUM_TRIALS}")
    print(f"Results folder: {RESULTS_FOLDER}")

    # Run simulation
    simulator = MonteCarloSimulator()
    results = simulator.run_simulation()

    # Analyze results
    topology_stats = simulator.analyze_topology_performance()
    print("\nTopology Performance Summary (first 5):")
    for topology, stats in list(topology_stats.items())[:5]:
        print(f"{topology}: {stats}")

    # Create comparison plots
    simulator.compare_topologies('fidelity')
    simulator.compare_topologies('sensitivity')

    # Full analysis with all plots
    summary = simulator.analyze_results()

    return results, summary


def run_custom_topology_simulation() -> Tuple[pd.DataFrame, Dict]:
    """Example of running simulations with custom topologies."""
    print("=== Running Custom Topology Simulation ===")

    # Use global parameters
    simulator = MonteCarloSimulator()
    results = simulator.run_simulation()

    # Analyze results
    topology_stats = simulator.analyze_topology_performance()
    print("Topology Performance Summary:")
    for topology, stats in list(topology_stats.items())[:5]:  # Print first 5
        print(f"{topology}: {stats}")

    # Create comparison plots
    simulator.compare_topologies('fidelity')
    simulator.compare_topologies('sensitivity')

    # Analyze results
    summary = simulator.analyze_results()

    return results, summary


def demonstrate_custom_topology_creation():
    """Demonstrate different ways to create custom topologies."""

    print("=== Demonstrating Custom Topology Creation ===\n")

    # Use a fixed number of qubits for consistent demonstration
    num_qubits = 6
    builder = EnhancedGraphStateBuilder(num_qubits)

    # Method 1: From edge list
    print("1. Creating topology from edge list...")
    custom_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    builder.create_topology("custom", custom_edges=custom_edges)
    print(f"   Created graph with {builder.graph.number_of_edges()} edges")

    # Method 2: From adjacency matrix
    print("\n2. Creating topology from adjacency matrix...")
    adj_matrix = np.array([
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0]
    ])
    builder.create_custom_topology_from_adjacency_matrix(adj_matrix)
    print(f"   Created graph with {builder.graph.number_of_edges()} edges")

    # Method 3: Random topology
    print("\n3. Creating random topology...")
    builder.create_random_topology(edge_probability=0.4, min_edges=5)
    print(f"   Created random graph with {builder.graph.number_of_edges()} edges")

    # Method 4: From function
    print("\n4. Creating topology from custom function...")

    def my_custom_topology(n):
        """Create a custom 'bow-tie' topology."""
        edges = []
        center = n // 2
        # Left triangle
        for i in range(center):
            for j in range(i + 1, center):
                edges.append((i, j))
        # Right triangle
        for i in range(center + 1, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        # Connect triangles through center
        edges.append((center - 1, center))
        edges.append((center, center + 1))
        return edges

    builder.create_custom_topology_from_function(my_custom_topology)
    print(f"   Created bow-tie graph with {builder.graph.number_of_edges()} edges")

    # Visualize the final topology
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{RESULTS_FOLDER}/custom_bowtie_topology_{timestamp}.png"
    # builder.visualize_topology("Custom Bow-Tie Topology", save_path=save_path)


def run_performance_comparison() -> Dict:
    """Compare performance of different topology categories."""

    print("=== Running Performance Comparison ===\n")

    # Define topology categories
    topology_categories = {
        "Standard": ["complete", "ring", "line", "star"],
        "Enhanced": ["grid_2d", "binary_tree", "ladder", "wheel", "small_world"],
        "Custom": ["custom_ring_with_shortcuts", "custom_double_star", "custom_modular"]
    }

    # num_qubits = QUBIT_RANGES[0] if QUBIT_RANGES else 6
    num_qubits = 6
    num_trials = max(30, NUM_TRIALS // 3)  # Reduce trials for comparison

    results_by_category = {}

    for category, topologies in topology_categories.items():
        print(f"Testing {category} topologies...")

        simulator = MonteCarloSimulator()
        results = simulator.run_simulation(
            topologies, [num_qubits], WEATHER_MODELS,
            INTENSITY_RANGE, num_trials
        )

        # Calculate average performance
        avg_performance = {
            'fidelity': results['fidelity'].mean(),
            'sensitivity': results['sensitivity'].mean(),
            'resilience': results['resilience'].mean(),
            'connectivity': results['is_connected'].mean()
        }

        results_by_category[category] = avg_performance
        print(f"   Average fidelity: {avg_performance['fidelity']:.3f}")
        print(f"   Average sensitivity: {avg_performance['sensitivity']:.3f}")
        print(f"   Average resilience: {avg_performance['resilience']:.3f}\n")

    # Plot comparison
    categories = list(results_by_category.keys())
    metrics = ['fidelity', 'sensitivity', 'resilience', 'connectivity']

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_MEDIUM)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results_by_category[cat][metric] for cat in categories]

        bars = axes[idx].bar(categories, values, alpha=0.7, color=['steelblue', 'darkorange', 'forestgreen'])
        axes[idx].set_title(f'Average {metric.title()}')
        axes[idx].set_ylabel(metric.title())

        # Fix axis scaling - set y-limits to fit data properly
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val > 0:
            # Add 10% padding above and below
            padding = range_val * 0.15
            axes[idx].set_ylim(min_val - padding, max_val + padding)
        else:
            # If all values are the same, center around the value
            axes[idx].set_ylim(min_val - 0.1, max_val + 0.1)

        # Add value labels on bars with better positioning
        for bar, value in zip(bars, values):
            # Position label at 95% of bar height, inside the bar
            label_y = bar.get_height() * 0.5  # Middle of bar
            axes[idx].text(bar.get_x() + bar.get_width() / 2, label_y,
                           f'{value:.3f}', ha='center', va='center',
                           fontweight='bold', color='white', fontsize=10)

    plt.tight_layout()

    # Save the comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{RESULTS_FOLDER}/performance_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Performance comparison saved to {save_path}")

    plt.show()

    return results_by_category


# ================================
# MAIN EXECUTION FUNCTION
# ================================

def main():
    """Main function to run all analyses with global parameters."""
    print("=" * 60)
    print("ENHANCED PAULI NOISE ANALYZER - USER FRIENDLY VERSION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Qubit ranges: {QUBIT_RANGES}")
    print(f"  Graph types: {len(GRAPH_TYPES)} types")
    print(f"  Weather models: {WEATHER_MODELS}")
    print(f"  Trials per config: {NUM_TRIALS}")
    print(f"  Results folder: {RESULTS_FOLDER}")
    print("=" * 60)

    # Create results directory
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # 1. Visualize all topologies for all qubit counts
    print("\n1. Visualizing all topologies for each qubit count...")
    visualize_all_topologies_for_all_qubits()

    # 2. Create topology comparison matrix
    print("\n2. Creating topology comparison matrix...")
    topology_df = create_topology_comparison_matrix()

    # 3. Demonstrate custom topology creation
    print("\n3. Demonstrating custom topology creation methods...")
    demonstrate_custom_topology_creation()

    # 4. Run performance comparison
    print("\n4. Running performance comparison...")
    performance_results = run_performance_comparison()

    # # 5. Run basic noise analysis
    # print("\n5. Running basic noise analysis...")
    # run_basic_noise_analysis()

    # 6. Run full simulation with custom topologies
    print("\n6. Running full simulation with all topologies...")
    results, summary = run_full_simulation()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("Key findings:")
    print(f"- Best topology for sensing: {summary['best_sensor_graph']['graph_type']}")
    print(f"- Most resilient topology: {summary['most_resilient_graph']['graph_type']}")
    print(f"- Total topologies tested: {len(results['graph_type'].unique())}")
    print(f"- All results saved to: {RESULTS_FOLDER}/")
    print("=" * 60)

    return results, summary, performance_results, topology_df


if __name__ == "__main__":
    # Run the complete analysis
    results, summary, performance_results, topology_df = main()