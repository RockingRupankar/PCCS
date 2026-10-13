"""
Reporting and visualization module for PCCS.
"""
from pccs.reporting.plots import (
    plot_fig1_synthetic_performance,
    plot_fig2_real_world_benchmarks,
    plot_fig3_cluster_fingerprints,
    plot_fig4_interactions,
    plot_fig5_falsification_psi,
    generate_all_publication_figures,
)
from pccs.reporting.tables import (
    export_summary_table,
    generate_statistical_comparison_table,
)

__all__ = [
    "plot_fig1_synthetic_performance",
    "plot_fig2_real_world_benchmarks",
    "plot_fig3_cluster_fingerprints",
    "plot_fig4_interactions",
    "plot_fig5_falsification_psi",
    "generate_all_publication_figures",
    "export_summary_table",
    "generate_statistical_comparison_table",
]
