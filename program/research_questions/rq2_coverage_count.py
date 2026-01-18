import sys
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from configparser import ConfigParser
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, shapiro
import statistics

# --- Configuration ---
# Define constants for file paths and parameters to avoid hardcoding.
MODULE_PATH = 'program/__module'
DB_CONFIG_FILE = 'program/envFile.ini'
OUTPUT_DIR = 'data/result_data/rq2'
PROJECT_FIGURE_DIR = os.path.join(OUTPUT_DIR, 'projects')

# --- Main Script ---

def plot_project_coverage_trend(coverage_data, output_pdf_path="coverage_chart.pdf"):
    """
    Generates and saves a PDF chart showing the coverage trend for a single project.
    The chart includes coverage percentage, total lines, and covered lines over time.

    Args:
        coverage_data (list of tuples): A list where each tuple contains (covered_line, total_line).
        output_pdf_path (str): The path to save the output PDF file.

    Returns:
        str: The path to the saved PDF file, or None if plotting was skipped.
    """
    if not coverage_data:
        print("Warning: No data provided to plot. Skipping graph creation.")
        return None
    # Ensure the output directory for project-specific figures exists.
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    # 1. Prepare DataFrame
    df = pd.DataFrame(coverage_data, columns=["covered_line", "total_line"])
    if df.empty:
        print("Warning: DataFrame is empty. Skipping graph creation.")
        return None

    # Calculate coverage percentage, handling division by zero.
    df["coverage_percent"] = np.divide(
        df["covered_line"], df["total_line"],
        out=np.zeros_like(df["covered_line"], dtype=float),
        where=df["total_line"] != 0
    ) * 100
    df["session_index"] = range(len(df))

    # 2. Setup Plot Style
    sns.set_theme(style="white")
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 3. Prepare dual axes
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # print(os.path.basename(output_pdf_path), len(df["coverage_percent"]))
    # 4. Right Y-axis: Bar/Fill chart for line counts
    palette = sns.color_palette("muted")
    total_color, covered_color = palette[4], palette[2]

    # Use a filled area chart for many data points, and a bar chart for fewer.
    if len(df) > 150:
        ax2.fill_between(df.session_index, 0, df["total_line"], color=total_color, alpha=0.5, label="Total Lines")
        ax2.fill_between(df.session_index, 0, df["covered_line"], color=covered_color, alpha=0.9, label="Covered Lines")
    else:
        ax2.bar(df.session_index, df["total_line"], width=0.7, label="Total Lines", color=total_color, alpha=0.5)
        ax2.bar(df.session_index, df["covered_line"], width=0.7, label="Covered Lines", color=covered_color, alpha=0.9)

    ax2.set_ylabel("Number of Lines", fontsize=10)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.grid(False)

    # 5. Left Y-axis: Line chart for coverage percentage
    line_color = palette[0]
    line = ax1.plot(
        df.session_index, df["coverage_percent"],
        color='red',
        alpha=0.7,
        label="Coverage (%)",
        linewidth=1.3,
        zorder=10,
        solid_capstyle='round'
    )
    # Add a white stroke effect to the line for better visibility
    plt.setp(line, path_effects=[
        path_effects.Stroke(linewidth=0.3, foreground='white'),
        path_effects.Normal()
    ])

    ax1.set_ylabel("Coverage (%)", fontsize=10, color=line_color)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', colors=line_color, labelsize=8)
    ax1.set_xlabel("Coverage Measurement Count", fontsize=10)
    ax1.grid(False)

    # 6. Despine axes for a cleaner look
    sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
    sns.despine(ax=ax2, top=True, right=False, left=True, bottom=False)

    # 7. Create a shared legend and adjust layout
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()

    # Save the figure to PDF
    fig.savefig(output_pdf_path, bbox_inches='tight')
    plt.close(fig)

    return output_pdf_path


def plot_coverage_distribution_trend(sessions_data, output_pdf_path):
    """
    Plots the distribution of coverage rates per coverage measurement count (percentiles, mean, median)
    and saves it as a PDF.

    Args:
        sessions_data (list of lists): Each inner list contains coverage rate data
                                        for a specific session index.
        output_pdf_path (str): The path for the output PDF file.
    """
    if not sessions_data:
        print("Warning: No session data provided. Skipping distribution trend plot.")
        return

    print(f"Generating coverage distribution trend plot... (Data points: {len(sessions_data)} sessions)")

    # 1. Calculate statistics
    session_indices = list(range(len(sessions_data)))
    num_projects = [len(d) for d in sessions_data]
    
    # Percentiles to calculate
    percentiles_to_calc = [5, 25, 50, 75, 95]
    percentiles = {}
    
    print("Calculating percentiles for distribution plot...")
    # np.percentile needs to be executed for each session data (d)
    for p in tqdm(percentiles_to_calc, desc="Calculating Percentiles", leave=False):
        percentiles[p] = [np.percentile(d, p) for d in sessions_data]
        
    mean_values = [np.mean(d) for d in sessions_data]
    
    # 2. Set up graph (stack two subplots vertically)
    sns.set_theme(style="whitegrid")
    fig, (ax_num, ax_cov) = plt.subplots(
        2, 1, 
        figsize=(10, 6), 
        sharex=True, # Share X-axis
        gridspec_kw={'height_ratios': [1, 3]} # Height ratio of top and bottom plots
    )
    
    # --- 3. Upper plot: Number of projects ---
    ax_num.plot(session_indices, num_projects, color='tab:blue', linewidth=1.5)
    ax_num.set_ylabel('#Projects')
    ax_num.set_ylim(bottom=0)
    ax_num.set_title('Coverage Percentage across Fuzzing Sessions')
    
    # --- 4. Lower plot: Coverage distribution ---
    
    # Color definitions (shades of blue to match the image)
    cmap = plt.get_cmap('Blues')
    # Dark -> Light
    colors = [cmap(0.8), cmap(0.4)] 

    # Fill percentile regions (25% steps)
    ax_cov.fill_between(
        session_indices,
        percentiles[25],
        percentiles[75],
        color=colors[0],
        alpha=0.35,
        label='Percentile 25-75%',
        zorder=1,
    )
    # Outer 5-95% with light fill, border changed to thick blue line
    ax_cov.fill_between(
        session_indices,
        percentiles[5],
        percentiles[95],
        color=colors[1],
        alpha=0.28,
        zorder=0,
    )
    ax_cov.plot(
        session_indices,
        percentiles[5],
        color="#6889df",
        linewidth=1.3,
        label='Percentile 5-95%',
        zorder=3,
    )
    ax_cov.plot(
        session_indices,
        percentiles[95],
        color='#6889df',
        linewidth=1.3,
        zorder=3,
    )

    # Median (Green)
    ax_cov.plot(session_indices, percentiles[50], color='#2ca02c', linewidth=2, label='Median', zorder=4)
    
    # Mean (Red)
    ax_cov.plot(session_indices, mean_values, color="#ffb43b", linewidth=2, label='Mean', zorder=4)

    # Add vertical lines every 100
    for x in range(0, len(session_indices), 100):
        ax_cov.axvline(x=x, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Fix X-axis ticks to every 200
    ax_cov.set_xticks(range(0, len(session_indices), 200))

    ax_cov.set_ylabel('Line Coverage %')
    ax_cov.set_xlabel('Coverage Measurement Count (Sessions)')
    ax_cov.set_ylim(0, 100)
    ax_cov.set_xlim(left=0, right=len(session_indices)-1)
    
    # --- 5. Adjust legend and layout ---
    
    handles, labels = ax_cov.get_legend_handles_labels()
    order = [2, 1, 3, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], 
                loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    fig.savefig(output_pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Coverage distribution trend plot saved to: {output_pdf_path}")

def main():
    """
    Main function to perform RQ2 analysis.
    This script fetches coverage data from a database, calculates correlations,
    and generates plots to analyze coverage trends over fuzzing sessions.
    """
    print("--- Main process started ---")

    # Add the module path to sys.path for custom module imports
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)
    from dbFile import DB
    import queries1

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Database Connection and Data Fetching ---
    config = ConfigParser()
    config.read(DB_CONFIG_FILE)
    db_config = config["POSTGRES"]

    db = DB(database=db_config["POSTGRES_DB"], user=db_config["POSTGRES_USER"],
            password=db_config["POSTGRES_PASSWORD"], host=db_config["POSTGRES_IP"],
            port=db_config["POSTGRES_PORT"])
    db.connect()

    # Query to select projects with at least 365 coverage measurements
    query = """
        SELECT project
        FROM total_coverage
        WHERE coverage IS NOT NULL AND coverage > 0 AND date < '2025-01-08'
        GROUP BY project
        HAVING COUNT(*) >= 365
    """
    project_records = db.executeQuery("select", query)
    projects = [project[0] for project in project_records]
    

    # --- 2. Process Each Project ---
    all_project_correlations = []
    coverage_by_session_index = [[]] # List of lists, where index `i` holds coverage values for the i-th session
    
    normal_project_count = 0
    projects_tested_for_normality = 0
    
    print(f"\n--- Starting to process {len(projects)} projects ---")
    for project_name in tqdm(projects, desc="Processing projects"):
        query = queries1.GET_TOTAL_COVERAGE_EACH_PROJECT(project_name, 'coverage')
        
        raw_coverage_data = db.executeQuery("select", query)
        
        if not raw_coverage_data:
            continue

        # Calculate coverage percentage for each session
        coverage_trend = [
            (float(x[0]) / float(x[1])) * 100
            for x in raw_coverage_data if x[1] != 0  # Avoid division by zero
        ]
        
        if len(coverage_trend) >= 3:
            projects_tested_for_normality += 1
            try:
                # Test the "distribution" of coverage_trend (time series of coverage)
                sw_stat, sw_p = shapiro(coverage_trend)
                if sw_p > 0.05:
                    normal_project_count += 1
            except Exception as e:
                # Example: cases where all values are the same, etc.
                print(f"Warning: Shapiro test failed for {project_name}. Error: {e}")

        # Calculate Spearman correlation between session index and coverage trend
        if len(coverage_trend) < 2:
            corr = np.nan
        else:
            corr, _ = spearmanr(range(len(coverage_trend)), coverage_trend)

        all_project_correlations.append(corr)

        # Plot and save a figure for projects with high correlation
        if not np.isnan(corr) and abs(corr) > 0.5:
            figure_path = os.path.join(PROJECT_FIGURE_DIR, f"{corr:.4f}_{project_name}.pdf")
            plot_project_coverage_trend(raw_coverage_data, figure_path)

        # Aggregate coverage data by session index across all projects
        for i, cov in enumerate(coverage_trend):
            if len(coverage_by_session_index) <= i:
                coverage_by_session_index.append([])
            coverage_by_session_index[i].append(cov)
    
    print("\n--- Project processing finished ---\n")

    print("\n--- Analysis of Project Coverage Normality (Shapiro-Wilk) ---")
    if projects_tested_for_normality > 0:
        normality_percentage = (normal_project_count / projects_tested_for_normality) * 100
        print(f"Projects tested for normality (N >= 3 sessions): {projects_tested_for_normality}")
        print(f"Projects whose coverage trend follows normal distribution (p > 0.05): {normal_project_count}")
        print(f"Percentage of normally distributed projects: {normality_percentage:.2f}%")
    else:
        print("No projects had sufficient data (N >= 3) for normality testing.")
    
    # --- 3. Save Aggregated Data ---
    csv_path = os.path.join(OUTPUT_DIR, "coverage_by_session_index.csv")
    print(f"Saving coverage data per session index to: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(coverage_by_session_index)
    print(f"Successfully saved. Total rows (max sessions): {len(coverage_by_session_index)}")

    # --- 4. Overall Correlation Analysis ---
    print("\n--- Analysis of All Project Correlations ---")
    correlations_with_nan = np.array(all_project_correlations)
    valid_correlations = correlations_with_nan[~np.isnan(correlations_with_nan)]
    
    print(f"Total projects processed: {len(correlations_with_nan)}")
    print(f"Number of projects with valid correlation: {len(valid_correlations)}")
    print(f"Average correlation: {np.mean(valid_correlations):.4f}, Median correlation: {np.median(valid_correlations):.4f}")

    # Generate and save Violin+Box plot for correlation coefficients (DISABLED)
    # plt.figure(figsize=(6, 4))
    # plt.violinplot(valid_correlations, showmeans=False, showmedians=True)
    # plt.boxplot(valid_correlations, positions=[1.15], widths=0.15, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    # plt.xticks([1, 1.15], ['Violin', 'Box'])
    # plt.ylabel('Correlation')
    # plt.tight_layout(pad=0.2)
    # violin_path = os.path.join(OUTPUT_DIR, 'all_project_corr_violin_box.pdf')
    # plt.savefig(violin_path, format='pdf')
    # plt.close()
    # print(f"Violin+Box plot saved to: {violin_path}")

    # Generate and save a histogram of correlation coefficients
    plt.figure(figsize=(5, 3))
    sns.histplot(valid_correlations, bins=40, color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=0.2)
    hist_path = os.path.join(OUTPUT_DIR, 'all_project_corr_hist.pdf')
    plt.savefig(hist_path, format='pdf')
    plt.close()
    print(f"Correlation histogram saved to: {hist_path}")

    # --- 5. Boxplot of Coverage vs. Fuzzing Sessions ---
    print("\n--- Generating Boxplot of Coverage vs. Session Count ---")
    
    # Filter for sessions with at least 100 data points
    sessions_with_enough_data = [d for d in coverage_by_session_index if len(d) >= 100]
    print(f'Number of sessions with >= 100 projects: {len(sessions_with_enough_data)}')

    # Sample data for boxplot (e.g., one box every 100 sessions)
    n_step = 100
    boxplot_data = [coverage_by_session_index[i] for i in range(0, len(coverage_by_session_index), n_step) if len(coverage_by_session_index[i]) >= 100]
    
    # Define X-axis ticks and labels
    xtick_labels_full = [i for i in range(1, len(coverage_by_session_index) + 1, n_step) if len(coverage_by_session_index[i-1]) >= 100]
    label_step = 2
    xtick_positions = list(range(1, len(boxplot_data) + 1))[::label_step]
    xtick_labels = xtick_labels_full[::label_step]
    
    plt.figure(figsize=(7.5,4.5))
    ax1 = plt.gca()
    
    # Bar chart for the number of projects (background)
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax2.bar(range(1, len(boxplot_data) + 1), [len(data) for data in boxplot_data], color='#88c778', alpha=0.6, zorder=1)
    ax2.set_ylabel('Number of Projects')
    
    # Boxplot for coverage distribution (foreground)
    box = ax1.boxplot(boxplot_data, vert=True, patch_artist=True, zorder=3)
    for patch in box['boxes']:
        patch.set_facecolor('#e3eefa')
    for median in box['medians']:
        median.set_color('#000000')
    
    # Plot mean values as triangles
    for i, data in enumerate(boxplot_data, start=1):
        mean_value = np.mean(data)
        ax1.scatter(i, mean_value, color='#215F9A', marker='^', zorder=4, s=8)
    
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Coverage Measurement Count')
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=45)
    
    plt.tight_layout(pad=0.2)
    boxplot_path = os.path.join(OUTPUT_DIR, 'session_coverage_boxplot.pdf')
    plt.savefig(boxplot_path, format='pdf', transparent=True)
    plt.close()
    print(f"Boxplot saved to: {boxplot_path}")

    # --- 6. Correlation of Average/Median Coverage Over Time ---
    print("\n--- Correlation of Average/Median Coverage over Time ---")
    average_trend = [statistics.mean(s) for s in sessions_with_enough_data]
    median_trend = [statistics.median(s) for s in sessions_with_enough_data]
    session_indices = list(range(len(sessions_with_enough_data)))
    
    if len(median_trend) > 1:
        spearman_median = spearmanr(session_indices, median_trend)
        print("Spearman correlation (Session Index vs. Median):", spearman_median)
    else:
        print("Not enough data points to calculate correlation of coverage trends.")
    
    print("\n--- Normality Test for Median Trend (Shapiro-Wilk) ---")
    if len(median_trend) >= 3:
        sw_stat_median, sw_p_median = shapiro(median_trend)
        print(f"Shapiro-Wilk test for 'median_trend' (N={len(median_trend)}): p-value = {sw_p_median:.4f}")
        if sw_p_median > 0.05:
            print("-> The distribution of median coverage values (median_trend) CAN be considered normal.")
        else:
            print("-> The distribution of median coverage values (median_trend) is NOT normal.")
    else:
        print(f"Not enough median values (N={len(median_trend)}, required >= 3) to run Shapiro-Wilk test.")
    
    # Generate and save line plot for average and median trends
    print("Generating average/median line plot...")
    plt.figure(figsize=(6, 4))
    plt.plot(session_indices, average_trend, label='Average', marker='o', color='blue', markersize=1, linewidth=1)
    plt.plot(session_indices, median_trend, label='Median', marker='s', color='orange', markersize=1, linewidth=1)
    plt.xlabel('Session Index (with >= 100 projects)')
    plt.ylabel('Coverage (%)')
    plt.title('Average and Median Coverage Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    lineplot_path = os.path.join(OUTPUT_DIR, 'average_median_lineplot.pdf')
    plt.savefig(lineplot_path, format='pdf')
    plt.close()
    print(f"Line plot saved to: {lineplot_path}")

    # --- 7. (NEW) Distribution Trend Plot ---
    # sessions_with_enough_data uses the data defined in step 5 (sessions with 100 or more data points)
    print("\n--- Generating Coverage Distribution Trend Plot ---")
    distribution_plot_path = os.path.join(OUTPUT_DIR, 'session_coverage_distribution_trend.pdf')
    plot_coverage_distribution_trend(sessions_with_enough_data, distribution_plot_path)


    print("\n--- Main process finished ---")


if __name__ == '__main__':
    main()