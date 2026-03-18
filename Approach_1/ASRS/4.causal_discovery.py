"""
10_causal_discovery_fci_bootstrap.py

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Input produced by Program 08
INPUT_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\ASRS\binary_document_topics_adaptive.csv")

# Output directory
OUT_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\ASRS\causal_output"
)

# Conservative rule of thumb for constraint-based methods
MIN_SAMPLES_PER_FEATURE = 5

# Significance level for conditional independence tests
ALPHA = 0.05

# Visualization settings
GRAPH_DPI = 300  # High resolution for publications
GRAPH_FIGSIZE = (12, 8)  # Figure size in inches
NODE_SIZE = 2000  # Size of nodes in the graph
FONT_SIZE = 7  # Font size for node labels


# =============================================================================
# Utility functions
# =============================================================================

def sufficient_sample_size(n_samples: int, n_features: int) -> bool:
    """
    Check whether sample size is sufficient for FCI / PC style algorithms.

    This is intentionally conservative.
    """
    return n_samples >= MIN_SAMPLES_PER_FEATURE * n_features


def ensure_output_dir() -> None:
    """Create output directory if it does not already exist."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_and_clean_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleaning for a binary feature matrix.

    Steps:
    1) Convert boolean columns to {0,1}
    2) Drop rows with missing values (strict, avoids CI-test issues)
    3) Attempt to cast all features to integer
    """
    df2 = df.copy()

    # Convert booleans to integers
    for col in df2.columns:
        if df2[col].dtype == bool:
            df2[col] = df2[col].astype(int)

    # Drop rows with any missing values
    n_before = len(df2)
    df2 = df2.dropna(axis=0)
    n_after = len(df2)

    if n_after < n_before:
        print(
            f"[WARN] Dropped {n_before - n_after} rows due to missing values.")

    # Attempt to cast to int
    try:
        df2 = df2.astype(int)
    except Exception:
        print("[WARN] Couldn't cast all columns to int")

    return df2


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant (degenerate) columns."""
    keep_cols = []
    dropped_cols = []

    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            dropped_cols.append(col)
        else:
            keep_cols.append(col)

    if dropped_cols:
        print(f"[WARN] Dropping constant features: {dropped_cols}")

    return df[keep_cols]


def extract_edges_from_pag(cg, var_names: List[str]) -> pd.DataFrame:
    """
    Convert a causal-learn PAG object into a flat edge list.

    Each edge is represented as:
        source | target | endpoints
    """
    edges: List[Tuple[str, str, str]] = []

    # Handle GeneralGraph object directly
    if hasattr(cg, 'graph'):
        G = cg.graph
    elif hasattr(cg, 'G') and hasattr(cg.G, 'graph'):
        G = cg.G.graph
    else:
        G = cg

    # Get dimensions
    if hasattr(G, 'shape'):
        p = G.shape[0]
    else:
        print(f"[ERROR] Can't determine graph dim from type {type(G)}")
        return pd.DataFrame(edges, columns=["source", "target", "endpoints"])

    # Extract edges from adjacency matrix
    for i in range(p):
        for j in range(i + 1, p):
            # Check if there's any connection between i and j
            if hasattr(G[i, j], 'name') and hasattr(G[j, i], 'name'):
                ei_name = getattr(G[i, j], 'name', str(G[i, j]))
                ej_name = getattr(G[j, i], 'name', str(G[j, i]))
                if ei_name != "NO_EDGE" or ej_name != "NO_EDGE":
                    edges.append((var_names[i], var_names[j], f"{ei_name}--{ej_name}"))
            else:
                ei_val = G[i, j]
                ej_val = G[j, i]
                if ei_val != 0 or ej_val != 0:
                    edges.append((var_names[i], var_names[j], f"{ei_val}--{ej_val}"))

    return pd.DataFrame(edges, columns=["source", "target", "endpoints"])


def visualize_causal_graph(edges_df: pd.DataFrame, var_names: List[str],
                           output_path: Path) -> None:
    """
    Generate a publication-quality visualization of the causal graph.

    Args:
        edges_df: DataFrame containing edges with endpoints
        var_names: List of all variable names
        output_path: Path to save the visualization
    """
    if edges_df.empty:
        print("[VISUALIZATION] No edges to visualize.")
        return

    try:
        import networkx as nx
        print("[VISUALIZATION] Generating causal graph visualization...")

        # Create directed graph with NetworkX
        G = nx.DiGraph()

        # Add nodes
        for var in var_names:
            G.add_node(var)

        # Process edges and add them to graph
        edge_types = []
        for _, row in edges_df.iterrows():
            source, target, endpoints = row['source'], row['target'], row['endpoints']

            # Parse endpoint types
            if '--' in endpoints:
                left_end, right_end = endpoints.split('--')

                # Determine edge type based on endpoints
                if '1' in right_end and '-1' in left_end:
                    # Directed edge (causal)
                    G.add_edge(source, target, style='solid', color='red',
                               arrowstyle='->')
                    edge_types.append('directed')
                elif '1' in left_end and '-1' in right_end:
                    # Reverse directed edge
                    G.add_edge(target, source, style='solid', color='red',
                               arrowstyle='->')
                    edge_types.append('reverse_directed')
                elif '1' in left_end and '1' in right_end:
                    # Bidirectional edge (unoriented confounding)
                    G.add_edge(source, target, style='solid', color='purple',
                               arrowstyle='<->')
                    G.add_edge(target, source, style='solid', color='purple')
                    edge_types.append('bidirectional')
                elif '1' in right_end and '2' in left_end:
                    # Partially directed edge (possible causation)
                    G.add_edge(source, target, style='solid', color='blue',
                               arrowstyle='->')
                    edge_types.append('partially_directed')
                elif '1' in left_end and '2' in right_end:
                    # Reverse partially directed edge
                    G.add_edge(target, source, style='solid', color='blue',
                               arrowstyle='->')
                    edge_types.append('reverse_partial')
                elif '2' in left_end and '2' in right_end:
                    # Non-directed edge (unoriented)
                    G.add_edge(source, target, style='dotted', color='gray',
                               arrowstyle='-')
                    G.add_edge(target, source, style='dotted', color='gray')
                    edge_types.append('non_directed')
            else:
                # Default to simple directed edge
                G.add_edge(source, target, style='solid', color='black')
                edge_types.append('default')

        # Create figure with better aesthetics
        plt.figure(figsize=GRAPH_FIGSIZE, dpi=GRAPH_DPI)

        # Use spring layout for better node positioning
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=NODE_SIZE,
            alpha=0.5,
            linewidths=1.5
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=FONT_SIZE,
            font_weight='bold',
            font_family='Arial'
        )

        # Draw edges with different styles based on type
        edge_colors = [G[u][v].get('color') for u, v in G.edges()]
        edge_styles = [G[u][v].get('style') for u, v in G.edges()]
        edge_widths = [2.0 if G[u][v].get('color') == 'red' else 1.5 for u, v in G.edges()]

        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            style=edge_styles,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )

        # Add title and legend
        plt.title('Causal Graph (PAG) from FCI Algorithm', fontsize=14,
                  fontweight='bold', pad=20)

        # Create custom legend for edge types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='darkblue',
                  label='Variable Node'),
            Patch(facecolor='none', edgecolor='red',
                  label='Directed Edge (Causal)'),
            Patch(facecolor='none', edgecolor='blue',
                  label='Partially Directed Edge'),
            Patch(facecolor='none', edgecolor='purple',
                  label='Bidirected Edge (Confounding)'),
            Patch(facecolor='none', edgecolor='gray',
                  label='Non-directed Edge'),
        ]

        plt.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            framealpha=0.9,
            fontsize=9
        )

        # Add statistics text box
        stats_text = f"""
        Graph Statistics:
        • Variables: {len(var_names)}
        • Edges: {len(edges_df)}
        • Directed edges: {edge_types.count('directed') + edge_types.count('reverse_directed')}
        • Partially directed: {edge_types.count('partially_directed') + edge_types.count('reverse_partial')}
        • Bidirected: {edge_types.count('bidirectional')}
        """

        plt.text(
            0.02, 0.02, stats_text,
            transform=plt.gca().transAxes,
            fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Remove axes and set tight layout
        plt.axis('off')
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=GRAPH_DPI, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        print(f"[VISUALIZATION] Graph saved → {output_path}")

    except ImportError:
        print("[WARN] NetworkX not available")
        print("[INFO] Skipping graph visualization.")
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}")
        print("[INFO] Continuing without graph visualization.")


# =============================================================================
# Main logic
# =============================================================================

def main():
    # Guard 0: input exists
    if not INPUT_CSV.exists():
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        return

    # Load accident-level features
    df = pd.read_csv(INPUT_CSV, index_col=0)
    print(f"[INFO] Loaded feature matrix: {INPUT_CSV}")
    print(f"[INFO] Original shape: {df.shape} (samples, features)")

    # Defensive preprocessing
    df = encode_and_clean_binary(df)
    df = drop_constant_columns(df)
    n_samples, n_features = df.shape
    print(f"[INFO] After cleaning: {n_samples} samples, {n_features} features")

    # Guard 1: sample size sufficiency
    if not sufficient_sample_size(n_samples, n_features):
        print(
            "\n[SKIP] Insufficient samples for causal discovery.\n"
            f"       Required ≥ {MIN_SAMPLES_PER_FEATURE} × {n_features} = "
            f"{MIN_SAMPLES_PER_FEATURE * n_features}\n"
            f"       Found    = {n_samples}\n"
            "       Program 09 exits safely."
        )
        return

    print("[OK] Sample size sufficient. Running FCI...")
    print("[INFO] CI test: discrete G-squared (CIT('gsq'))")
    print(f"[INFO] Alpha  : {ALPHA}")

    # Prepare data for causal-learn
    data = df.values.astype(float)
    # just extract the word between [""] in the column names
    var_names: List[str] = list(df.columns)

    # Run FCI (causal-learn 0.1.4.3)
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import CIT

    # Primary CI test for discrete/binary data
    try:
        cit = CIT(data, "gsq")
    except ValueError:
        print("[WARN] CIT method 'gsq' unavailable; falling back to 'chisq'.")
        cit = CIT(data, "chisq")

    # Handle fci() return value
    fci_result = fci(
        data,
        independence_test=cit,
        alpha=ALPHA,
        verbose=True,
        variable_names=var_names,
    )

    # Extract graph from result
    if isinstance(fci_result, tuple):
        cg = fci_result[0]
        print(f"[INFO] fci() returned tuple of length {len(fci_result)}")
    else:
        cg = fci_result

    ensure_output_dir()

    # Save PAG (text form)
    pag_path = OUT_DIR / "causal_pag.txt"
    with open(pag_path, "w", encoding="utf-8") as f:
        f.write(str(cg))
    print(f"[OK] PAG written → {pag_path}")

    # Save edge list
    edges_df = extract_edges_from_pag(cg, var_names)
    edges_path = OUT_DIR / "causal_edges.csv"
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Edge list written → {edges_path}")
    print(f"[INFO] Number of edges: {len(edges_df)}")

    # Generate visualization
    graph_path = OUT_DIR / "causal_graph_asrs.png"
    visualize_causal_graph(edges_df, var_names, graph_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CAUSAL DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples × {n_features} features")
    print(f"Significance level (alpha): {ALPHA}")
    print(f"Discovered edges: {len(edges_df)}")

    if len(edges_df) > 0:
        print("\nTop 5 causal relationships (by alphabetical order):")
        for i, (_, row) in enumerate(edges_df.head(5).iterrows()):
            print(f"{i+1}. {row['source']} → {row['target']} ({row['endpoints']})")

    print(f"\nOutput files saved to: {OUT_DIR}")
    print("  1. causal_pag.txt     - Text representation of PAG")
    print("  2. causal_edges.csv   - Edge list with endpoints")
    print("  3. causal_graph.png   - Publication-quality visualization")
    print("=" * 60)
    print("\n[DONE] Program completed successfully")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
