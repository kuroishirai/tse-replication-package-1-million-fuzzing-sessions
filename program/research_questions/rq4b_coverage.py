import sys
import os
from configparser import ConfigParser
import pandas as pd
import numpy as np
import logging
import csv 
from tqdm import tqdm 

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys
from matplotlib.ticker import ScalarFormatter, LogLocator, NullLocator 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Statistical analysis libraries
from scipy.stats import mannwhitneyu, brunnermunzel, levene, spearmanr

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration (Assuming CWD=/app) ---

CWD = os.getcwd() 
logger.info(f"Using Current Working Directory: {CWD}")

MODULE_PATH = os.path.join(CWD, 'program/__module')
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
    logger.info(f"Added to sys.path: {MODULE_PATH}")

try:
    from dbFile import DB
except ImportError:
    logger.error(f"Error: Could not import 'dbFile' from '{MODULE_PATH}'.")
    sys.exit(1)

DB_CONFIG_FILE = os.path.join(CWD, 'program/envFile.ini')
CORPUS_ANALYSIS_CSV = os.path.join(CWD, 'data/processed_data/csv/project_corpus_analysis.csv')
OUTPUT_DIR = os.path.join(CWD, 'data/result_data/rq4/coverage')
FILE_FORMAT = 'pdf'  # Graph save format ('pdf' or 'png')

# *** Global Variables ***
ANALYSIS_ITERATIONS = 7
DAYS_THRESHOLD = 7  # Day threshold (used for classifying Group 3 and Group 4)
INCLUDE_MISSING_PRE_IN_G2 = False  # Whether to add Group 3/4 projects with missing Pre data to Group 2 in Analysis 1

# *** User-defined: Integrated mapping of percentiles, colors, and line styles (GLOBAL) ***
PERCENTILE_STYLE_MAP = {
    # Q1: Vivid blue
    25: {'color': '#0000FF', 'linewidth': 1.0, 'linestyle': '-', 'label': 'Q1'}, 
    # Median: Vivid green
    50: {'color': '#00FF00', 'linewidth': 1.5, 'linestyle': '-',  'label': 'Median'}, 
    # Q3: Vivid red
    75: {'color': '#FF0000', 'linewidth': 1.0, 'linestyle': '-', 'label': 'Q3'}, 
}

# List of percentile values for plotting and aggregation
PERCENTILES_TO_CALCULATE = sorted([int(k) for k in PERCENTILE_STYLE_MAP.keys()])

# Global step for boxplot sampling
BOXPLOT_STEP = 100

# *** Nested Boxplot Configuration ***
NESTED_BOXPLOT_CONFIG = {
    'color_a': "#1f77b4",  # Group A (Wide) fill color
    'edge_a': "#104e8b",   # Group A (Wide) edge color
    'color_b': "#ff7f0e",  # Group B (Narrow) fill color
    'edge_b': "#d65f00",   # Group B (Narrow) edge color
    'step': BOXPLOT_STEP,
    'alpha_a': 0.2,
    'alpha_b': 0.6,
    'linewidth_a': 1.8,
    'linewidth_b': 1.1,
    'width_a': 0.5,
    'width_b': 0.25,
    'zorder_a': 1,
    'zorder_b': 2,
    'min_projects': 100,
    'ylim': (0, 100),
    'yticks': [0, 20, 40, 60, 80, 100],
    'ax2_ylim': (0, 750),
    'ax2_yticks': [0, 150, 300, 450, 600, 750],
    'ax2_alpha': 0.7,
    'ax2_linewidth': 1.5,
    'ax2_linestyle': ':',
}

# --- Simple two-color boxplot scheme ---
# Color 1: Fill (specified within each plot function or auto-set by Hue)
# Color 2: Edge/Lines (unified below)
BOXPLOT_EDGE_COLOR = '#333333'  # Dark gray (almost black): for border, whiskers, median
# Comparative boxplot unified edge/whisker/cap linewidth
COMPARATIVE_EDGE_LINEWIDTH = 1.0
# Delta plot unified edge/whisker/cap/median linewidth
DELTA_EDGE_LINEWIDTH = 1.2

# --- Helper Function for Color Adjustment ---
def adjust_color_hls_reduction(hex_color, l_reduction_percent, s_reduction_percent):
    """
    Takes a HEX color and adjusts HLS L (lightness) and S (saturation) by specified percentage reduction rates.
    """
    try:
        c = mcolors.to_rgb(hex_color)
        h, l, s = colorsys.rgb_to_hls(*c)
        l_factor = 1.0 - (l_reduction_percent / 100.0)
        s_factor = 1.0 - (s_reduction_percent / 100.0)
        new_l = max(0.0, l * l_factor)
        new_s = max(0.0, s * s_factor)
        return mcolors.to_hex(colorsys.hls_to_rgb(h, new_l, new_s))
    except Exception as e:
        logger.error(f"Color adjustment failed: {e}")
        return hex_color

def apply_two_color_boxplot(ax, fill_color=None, edge_color=BOXPLOT_EDGE_COLOR, alpha=0.6, linewidth=1.2):
    """
    Force-convert existing boxplot style to "two-color composition".
    
    Args:
        ax: Matplotlib Axes
        fill_color: Box fill color (if None, keep original color and apply only alpha)
        edge_color: Unified color for border, whiskers, caps, and median
        alpha: Fill transparency
        linewidth: Line thickness
    """
    try:
        # 1. Box (Patch) settings: fill color and border color
        for patch in getattr(ax, 'artists', []):
            # Determine fill color
            if fill_color is not None:
                fc = mcolors.to_rgba(fill_color, alpha)
            else:
                # Get original color and apply only Alpha
                original_fc = patch.get_facecolor()
                fc = mcolors.to_rgba(original_fc, alpha)
            
            patch.set_facecolor(fc)
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(linewidth)

        # 2. Lines settings: Unify whiskers, caps, median all to edge_color
        for line in getattr(ax, 'lines', []):
            line.set_color(edge_color)
            line.set_linewidth(linewidth)
            line.set_alpha(alpha)
            # Don't change lines that originally have no line style (linestyle='None' for Outliers, etc.)
            if line.get_linestyle() != 'None':
                try:
                    line.set_linestyle('-')
                except Exception:
                    pass
            
    except Exception as e:
        logger.warning(f"Failed to apply two-color style: {e}")

def get_eligible_projects_from_db(db):
    """Get projects that meet RQ1 criteria (coverage 365 days or more)"""
    logger.info("Connecting to DB to fetch eligible projects (RQ1 criteria)...")
    try:
        query = """
            SELECT project
            FROM total_coverage
            WHERE coverage IS NOT NULL AND coverage > 0 AND date < '2025-01-08'
            GROUP BY project
            HAVING COUNT(*) >= 365
        """
        projects_result = db.executeQuery("select", query)
        eligible_projects = {project[0] for project in projects_result}
        logger.info(f"Found {len(eligible_projects)} eligible projects in DB.")
        return eligible_projects
    except Exception as e:
        logger.error(f"Error fetching projects from DB: {e}")
        return None

def categorize_projects_and_get_times(eligible_projects_set, corpus_csv_path):
    """Load CSV and classify into G1-G4"""
    logger.info(f"Loading corpus analysis data from '{corpus_csv_path}'...")
    try:
        df = pd.read_csv(corpus_csv_path)
        df['corpus_commit_time'] = pd.to_datetime(df['corpus_commit_time'], errors='coerce', utc=True)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None, None, None
        
    filtered_df = df[df['project_name'].isin(eligible_projects_set)].copy()
    if filtered_df.empty:
        return None, None, None

    cat_null_g1 = filtered_df['time_elapsed_seconds'].isna() 
    cat_same_time_g2 = (filtered_df['time_elapsed_seconds'] == 0) & (~cat_null_g1) 
    cat_under_1_day_g3 = (filtered_df['time_elapsed_seconds'] > 0) & (filtered_df['time_elapsed_seconds'] < DAYS_THRESHOLD * 86400) & (~cat_null_g1)
    cat_over_1_day_g4 = (filtered_df['time_elapsed_seconds'] >= DAYS_THRESHOLD * 86400) & (~cat_null_g1)

    project_groups = {
        'group1': set(filtered_df[cat_null_g1]['project_name']),
        'group2': set(filtered_df[cat_same_time_g2]['project_name']),
        'group3': set(filtered_df[cat_under_1_day_g3]['project_name']),
        'group4': set(filtered_df[cat_over_1_day_g4]['project_name'])
    }
    
    print("\n=== Number of Projects by Group ===")
    print(f"Group 1 (No Corpus): {len(project_groups['group1'])} projects")
    print(f"Group 2 (Same Time): {len(project_groups['group2'])} projects")
    print(f"Group 3 (< {DAYS_THRESHOLD} day): {len(project_groups['group3'])} projects")
    print(f"Group 4 (>= {DAYS_THRESHOLD} day): {len(project_groups['group4'])} projects")
    print(f"Total: {sum(len(v) for v in project_groups.values())} projects\n")
    
    group_2_3_4_df = filtered_df[~cat_null_g1][['project_name', 'corpus_commit_time']].set_index('project_name')
    g2_g3_g4_times_df = filtered_df[~cat_null_g1][['project_name', 'time_elapsed_seconds']].set_index('project_name')

    return project_groups, group_2_3_4_df, g2_g3_g4_times_df

def get_initial_coverage(db, project_list):
    """Get initial coverage"""
    if not project_list: return []
    try:
        in_clause = ", ".join(f"""'{proj.replace("'", "''")}'""" for proj in project_list)
    except Exception: return []
    
    if not in_clause: return []

    query = f"""
    WITH RankedCoverage AS (
        SELECT project, coverage,
            ROW_NUMBER() OVER(PARTITION BY project ORDER BY date ASC) as rn
        FROM total_coverage
        WHERE project IN ({in_clause})
          AND coverage IS NOT NULL AND coverage > 0
          AND date < '2025-01-08'
    )
    SELECT coverage FROM RankedCoverage WHERE rn = 1;
    """
    try:
        results = db.executeQuery("select", query)
        return [float(row[0]) for row in results if row and row[0] is not None]
    except Exception as e:
        logger.error(f"Error executing initial coverage query: {e}")
        return []

def analyze_g2_vs_g1_initial_coverage(db, group2_projects, group1_projects, additional_group2_projects=None):
    """Analysis 1: G2 vs G1 Initial Coverage Comparison"""
    final_group2_projects = set(group2_projects)
    if additional_group2_projects:
        final_group2_projects.update(additional_group2_projects)
    
    print("\n=== Analysis 1: G2 vs G1 Initial Coverage Comparison ===")
    print(f"Groups used: Group 2 (G2) vs Group 1 (G1)")
    print(f"Number of Group 2 projects: {len(final_group2_projects)}")
    print(f"Number of Group 1 projects: {len(group1_projects)}\n")
        
    group2_coverage = get_initial_coverage(db, final_group2_projects)
    group1_coverage = get_initial_coverage(db, group1_projects)
    n1, n2 = len(group2_coverage), len(group1_coverage)

    if n1 > 0 and n2 > 0:
        try:
            u_stat, p_value_mw = mannwhitneyu(group2_coverage, group1_coverage, alternative='two-sided')
            logger.info(f"[RESULT] Mann-Whitney U (G2 vs G1): p-value={p_value_mw:.4f}")
            
            u1_stat, _ = mannwhitneyu(group2_coverage, group1_coverage, alternative='greater')
            d_statistic = (2 * u1_stat) / (n1 * n2) - 1
            logger.info(f"[RESULT] Cliff's Delta: {d_statistic:.4f}")
            
            bm_stat, p_value_bm = brunnermunzel(group2_coverage, group1_coverage, alternative='two-sided')
            logger.info(f"[RESULT] Brunner-Munzel (G2 vs G1): p-value={p_value_bm:.4f}, BM-statistic={bm_stat:.4f}")
            
            levene_stat, p_value_levene = levene(group2_coverage, group1_coverage)
            logger.info(f"[RESULT] Levene's Test (G2 vs G1): p-value={p_value_levene:.4f}, statistic={levene_stat:.4f}")
        except Exception as e:
            logger.error(f"Stats error during BM/MWU: {e}")

    # Prepare stats dict to return and save
    stats = None
    if n1 > 0 and n2 > 0:
        try:
            stats = {
                'n_g2': n1,
                'n_g1': n2,
                'mannwhitney_p_two_sided': float(p_value_mw) if 'p_value_mw' in locals() else np.nan,
                'cliffs_delta': float(d_statistic) if 'd_statistic' in locals() else np.nan,
                'brunner_stat': float(bm_stat) if 'bm_stat' in locals() else np.nan,
                'brunner_p': float(p_value_bm) if 'p_value_bm' in locals() else np.nan,
                'levene_stat': float(levene_stat) if 'levene_stat' in locals() else np.nan,
                'levene_p': float(p_value_levene) if 'p_value_levene' in locals() else np.nan,
            }
        except Exception:
            stats = None

    # Save stats to CSV for easier review (DISABLED)
    # try:
    #     os.makedirs(OUTPUT_DIR, exist_ok=True)
    #     stats_file = os.path.join(OUTPUT_DIR, 'initial_coverage_stats.csv')
    #     with open(stats_file, 'w', newline='') as fh:
    #         writer = csv.DictWriter(fh, fieldnames=['metric', 'value'])
    #         writer.writeheader()
    #         if stats:
    #             for k, v in stats.items():
    #                 writer.writerow({'metric': k, 'value': v})
    #         else:
    #             writer.writerow({'metric': 'note', 'value': 'Insufficient data to compute stats'})
    #     logger.info(f"Saved initial coverage stats to {stats_file}")
    # except Exception as e:
    #     logger.warning(f"Failed to save initial coverage stats: {e}")

    return group2_coverage, group1_coverage, final_group2_projects, stats

def get_full_coverage_trend(db, project_name):
    """Get all time series data"""
    query = f"""
        SELECT coverage FROM total_coverage 
        WHERE project = '{project_name.replace("'", "''")}'
          AND coverage IS NOT NULL AND coverage > 0 AND date < '2025-01-08'
        ORDER BY date ASC;
    """
    try:
        results = db.executeQuery("select", query)
        return [float(row[0]) for row in results if row and row[0] is not None]
    except: return []

def plot_nested_variable_width_boxplot(db, group_a_projects, group_b_projects, output_dir, file_format='pdf'):
    """
    Function to display and compare two groups with "different width boxplots" overlaid.
    Style: Fill color (Group A/B) + Unified edge color (BOXPLOT_EDGE_COLOR)
    """
    logger.info("Generating Nested Variable-Width Boxplot (Group A vs Group B)...")
    
    sessions_a, sessions_b = [], []
    
    # Group A (Wide) Data Fetch
    for project_name in tqdm(group_a_projects, desc="Fetching Group A (Wide)", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(sessions_a) <= i: sessions_a.append([])
            sessions_a[i].append(cov)

    # Group B (Narrow) Data Fetch
    for project_name in tqdm(group_b_projects, desc="Fetching Group B (Narrow)", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(sessions_b) <= i: sessions_b.append([])
            sessions_b[i].append(cov)

    STEP = NESTED_BOXPLOT_CONFIG['step']
    max_len = max(len(sessions_a), len(sessions_b))
    indices = range(0, max_len, STEP)
    
    data_a, data_b = [], []
    count_data = []

    for idx in indices:
        session_label = idx + 1
        
        count_a = 0
        if idx < len(sessions_a) and len(sessions_a[idx]) >= NESTED_BOXPLOT_CONFIG['min_projects']:
            vals = sessions_a[idx]
            count_a = len(vals)
            for v in vals:
                data_a.append({'Session': session_label, 'Coverage': v})

        count_b = 0
        if idx < len(sessions_b) and len(sessions_b[idx]) >= NESTED_BOXPLOT_CONFIG['min_projects']:
            vals = sessions_b[idx]
            count_b = len(vals)
            for v in vals:
                data_b.append({'Session': session_label, 'Coverage': v})

        if count_a > 0 or count_b > 0:
            count_data.append({
                'Session': session_label, 
                'Count_A': count_a, 
                'Count_B': count_b
            })

    if not data_a and not data_b:
        logger.warning("No sufficient data for nested boxplot.")
        return

    df_a = pd.DataFrame(data_a)
    df_b = pd.DataFrame(data_b)
    df_count = pd.DataFrame(count_data)

    # --- Drawing ---
    fig, ax1 = plt.subplots(figsize=(5, 3))
    sns.set_style("whitegrid")
    
    fill_a = NESTED_BOXPLOT_CONFIG['color_a']
    fill_b = NESTED_BOXPLOT_CONFIG['color_b']
    edge_a = NESTED_BOXPLOT_CONFIG['edge_a']
    edge_b = NESTED_BOXPLOT_CONFIG['edge_b']
    linewidth_a = NESTED_BOXPLOT_CONFIG['linewidth_a']
    linewidth_b = NESTED_BOXPLOT_CONFIG['linewidth_b']
    alpha_a = NESTED_BOXPLOT_CONFIG['alpha_a']
    alpha_b = NESTED_BOXPLOT_CONFIG['alpha_b']
    
    # Property settings for each group: Apply alpha only to fill, not to lines
    line_props_a = dict(color=edge_a, linewidth=linewidth_a)
    line_props_b = dict(color=edge_b, linewidth=linewidth_b)
    box_props_a = dict(facecolor=mcolors.to_rgba(fill_a, alpha_a), edgecolor=edge_a, linewidth=linewidth_a)
    box_props_b = dict(facecolor=mcolors.to_rgba(fill_b, alpha_b), edgecolor=edge_b, linewidth=linewidth_b)

    # 1. Group A (Wide Box)
    if not df_a.empty:
        sns.boxplot(data=df_a, x='Session', y='Coverage', ax=ax1,
                    width=NESTED_BOXPLOT_CONFIG['width_a'],
                    showfliers=False,
                    zorder=NESTED_BOXPLOT_CONFIG['zorder_a'],
                    boxprops=box_props_a,
                    whiskerprops=line_props_a,
                    capprops=line_props_a,
                    medianprops=line_props_a)

    # 2. Group B (Narrow Box)
    if not df_b.empty:
        sns.boxplot(data=df_b, x='Session', y='Coverage', ax=ax1,
                    width=NESTED_BOXPLOT_CONFIG['width_b'],
                    showfliers=False,
                    zorder=NESTED_BOXPLOT_CONFIG['zorder_b'],
                    boxprops=box_props_b,
                    whiskerprops=line_props_b,
                    capprops=line_props_b,
                    medianprops=line_props_b)

    # --- Right axis (Number of projects) ---
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    if not df_count.empty:
        unique_sessions = sorted(list(set(df_count['Session'])))
        session_map = {val: i for i, val in enumerate(unique_sessions)}
        x_indices = [session_map[s] for s in df_count['Session']]

        # Use group colors for line plots as well
        ax2.plot(x_indices, df_count['Count_A'], color=fill_a, linestyle=NESTED_BOXPLOT_CONFIG['ax2_linestyle'], linewidth=NESTED_BOXPLOT_CONFIG['ax2_linewidth'], label='Count A (Wide)', alpha=NESTED_BOXPLOT_CONFIG['ax2_alpha'])
        ax2.plot(x_indices, df_count['Count_B'], color=fill_b, linestyle=NESTED_BOXPLOT_CONFIG['ax2_linestyle'], linewidth=NESTED_BOXPLOT_CONFIG['ax2_linewidth'], label='Count B (Narrow)', alpha=NESTED_BOXPLOT_CONFIG['ax2_alpha'])

        ax1.set_xticks(list(range(len(unique_sessions))))
        ax1.set_xticklabels(unique_sessions, rotation=45)
        ax1.set_xlim(left=-0.5, right=len(unique_sessions) - 0.5)

    ax2.set_ylabel('Number of Projects')
    ax2.set_ylim(NESTED_BOXPLOT_CONFIG['ax2_ylim'])
    ax2.set_yticks(NESTED_BOXPLOT_CONFIG['ax2_yticks'])

    # --- Legend and axis settings ---
    ax1.set_xlabel('Coverage Measurement Count')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(NESTED_BOXPLOT_CONFIG['ylim'])
    ax1.set_yticks(NESTED_BOXPLOT_CONFIG['yticks'])
    
    # Custom legend: Use edge color for each group
    legend_elements = [
        # Box Legends
        Patch(facecolor=fill_a, edgecolor=edge_a, alpha=NESTED_BOXPLOT_CONFIG['alpha_a'], label='Group A (No Seed)'),
        Patch(facecolor=fill_b, edgecolor=edge_b, alpha=NESTED_BOXPLOT_CONFIG['alpha_b'], label='Group B (Initial Seed)'),
        # Line Legends
        Line2D([0], [0], color=fill_a, linestyle=NESTED_BOXPLOT_CONFIG['ax2_linestyle'], linewidth=NESTED_BOXPLOT_CONFIG['ax2_linewidth'], alpha=NESTED_BOXPLOT_CONFIG['ax2_alpha'], label='Group A (Projects)'),
        Line2D([0], [0], color=fill_b, linestyle=NESTED_BOXPLOT_CONFIG['ax2_linestyle'], linewidth=NESTED_BOXPLOT_CONFIG['ax2_linewidth'], alpha=NESTED_BOXPLOT_CONFIG['ax2_alpha'], label='Group B (Projects)')
    ]
    
    # Reorder: Left column [A Cov, A Proj], Right column [B Cov, B Proj]
    ordered_handles = [legend_elements[0], legend_elements[2], legend_elements[1], legend_elements[3]]
    ax1.legend(handles=ordered_handles, loc='upper left', fontsize='small', ncol=2)

    # Axis limits
    min_projects = 100
    if not df_count.empty:
        valid_sessions = df_count[(df_count['Count_A'] >= min_projects) & (df_count['Count_B'] >= min_projects)]
        if not valid_sessions.empty:
            min_pos = session_map.get(valid_sessions['Session'].min(), 0)
            max_pos = session_map.get(valid_sessions['Session'].max(), len(unique_sessions) - 1)
            ax1.set_xlim(left=max(min_pos - 0.5, -0.5), right=min(max_pos + 0.5, len(unique_sessions) - 0.5))

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'nested_boxplot_comparison.{file_format}')
    plt.savefig(save_path, format=file_format, bbox_inches='tight')
    logger.info(f"Saved nested boxplot to {save_path}")
    plt.close()

def plot_g2_g1_comparative_boxplot(db, g2_project_list, g1_project_list, output_dir, file_format='pdf', overlap_fraction=0.5, total_span=1.5, width_scale=0.5):
    """
    G2 vs G1 distribution comparison (Side-by-side).
    With project count (line plot).
    Display only periods where both groups have 100 or more projects.
    """
    logger.info("Generating G2 vs G1 Comparative Boxplot...")
    
    g2_sessions, g1_sessions = [], []
    
    # G2 Data Fetch
    for project_name in tqdm(g2_project_list, desc="Fetching G2 Data", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(g2_sessions) <= i: g2_sessions.append([])
            g2_sessions[i].append(cov)

    # G1 Data Fetch
    for project_name in tqdm(g1_project_list, desc="Fetching G1 Data", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(g1_sessions) <= i: g1_sessions.append([])
            g1_sessions[i].append(cov)

    STEP = BOXPLOT_STEP
    max_len = max(len(g2_sessions), len(g1_sessions))
    indices = range(0, max_len, STEP)
    min_projects_limit = 100
    
    unique_sessions = []
    counts_a = []
    counts_b = []
    data_a_list = []
    data_b_list = []
    
    for idx in indices:
        session_label = idx + 1
        
        cnt_a = len(g1_sessions[idx]) if idx < len(g1_sessions) else 0
        cnt_b = len(g2_sessions[idx]) if idx < len(g2_sessions) else 0
        
        # Cutoff: stop if either group has fewer than 100 projects
        if cnt_a < min_projects_limit or cnt_b < min_projects_limit:
            break

        vals_a = g1_sessions[idx] if idx < len(g1_sessions) else []
        vals_b = g2_sessions[idx] if idx < len(g2_sessions) else []
        
        unique_sessions.append(session_label)
        counts_a.append(cnt_a)
        counts_b.append(cnt_b)
        data_a_list.append(vals_a)
        data_b_list.append(vals_b)

    if not unique_sessions:
        logger.warning("No sufficient data for boxplot.")
        return

    # --- Drawing ---
    fig, ax1 = plt.subplots(figsize=(5, 3))
    sns.set_style("whitegrid")



    central_pos = np.arange(len(unique_sessions))

    f = max(0.0, min(0.99, overlap_fraction))
    S = max(0.1, float(total_span))
    w = S / (2.0 - f)
    try:
        scale = float(width_scale)
    except Exception:
        scale = 0.5
    scale = max(0.01, min(1.0, scale))
    w = w * scale
    w = max(0.02, w)
    d = w * (1.0 - f)
    positions_a = central_pos - (d / 2.0)
    positions_b = central_pos + (d / 2.0)

    gA_color = '#66b3ff'
    gB_color = '#ff9999'



    bp_a = ax1.boxplot(data_a_list, positions=positions_a, widths=w, patch_artist=True, showfliers=False)
    bp_b = ax1.boxplot(data_b_list, positions=positions_b, widths=w, patch_artist=True, showfliers=False)

    edge_a = NESTED_BOXPLOT_CONFIG.get('edge_a', '#104e8b')
    edge_b = NESTED_BOXPLOT_CONFIG.get('edge_b', '#d65f00')
    unified_lw = COMPARATIVE_EDGE_LINEWIDTH
    median_lw = max(1.2, unified_lw)
    z_a = NESTED_BOXPLOT_CONFIG.get('zorder_a', 1)
    z_b = NESTED_BOXPLOT_CONFIG.get('zorder_b', 2)

    for box in bp_a['boxes']:
        box.set(facecolor=gA_color, edgecolor=edge_a, linewidth=unified_lw, alpha=0.6)
        box.set_zorder(z_a)
        try: box.set_linestyle('--')
        except: pass
    for whisker in bp_a['whiskers']:
        whisker.set(color=edge_a, linewidth=unified_lw, linestyle='--')
        whisker.set_zorder(z_a)
    for cap in bp_a['caps']:
        cap.set(color=edge_a, linewidth=unified_lw, linestyle='--')
        cap.set_zorder(z_a)
    for median in bp_a['medians']:
        median.set(color=edge_a, linewidth=median_lw)
        median.set_zorder(z_a)

    for box in bp_b['boxes']:
        box.set(facecolor=gB_color, edgecolor=edge_b, linewidth=unified_lw, alpha=0.6)
        box.set_zorder(z_b)
        try: box.set_linestyle('-')
        except: pass
    for whisker in bp_b['whiskers']:
        whisker.set(color=edge_b, linewidth=unified_lw, linestyle='-')
        whisker.set_zorder(z_b)
    for cap in bp_b['caps']:
        cap.set(color=edge_b, linewidth=unified_lw, linestyle='-')
        cap.set_zorder(z_b)
    for median in bp_b['medians']:
        median.set(color=edge_b, linewidth=median_lw)
        median.set_zorder(z_b)

    ax1.set_ylabel('Coverage (%)')
    ax1.set_xlabel('Coverage Measurement Count')
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_xticks(central_pos)
    ax1.set_xticklabels(unique_sessions, rotation=45)
    ax1.set_xlim(left=-0.5, right=len(unique_sessions) - 0.5)

    try: ax1.get_legend().remove()
    except: pass
    patch_cov_a = Patch(facecolor=gA_color, edgecolor=BOXPLOT_EDGE_COLOR, alpha=0.6, label='Group A (No Seed)')
    patch_cov_b = Patch(facecolor=gB_color, edgecolor=BOXPLOT_EDGE_COLOR, alpha=0.6, label='Group B (Initial Seed)')
    ordered_handles = [patch_cov_a, patch_cov_b]
    ax1.legend(handles=ordered_handles, loc='upper left', fontsize='small', ncol=2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'g2_g1_boxplot_comparison.{file_format}')
    plt.savefig(save_path, format=file_format, bbox_inches='tight')
    logger.info(f"Saved comparative boxplot to {save_path}")
    plt.close()

def plot_g2_g1_comparative_violin(db, g2_project_list, g1_project_list, output_dir, file_format='pdf'):
    """
    G2 vs G1 comparison (Split Violin Plot).
    Does not display project count (line plot).
    Display only periods where both groups have 100 or more projects.
    """
    logger.info("Generating G2 vs G1 Comparative Violin Plot...")
    
    g2_sessions, g1_sessions = [], []
    
    for project_name in tqdm(g2_project_list, desc="Fetching G2 Data (Violin)", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(g2_sessions) <= i: g2_sessions.append([])
            g2_sessions[i].append(cov)

    for project_name in tqdm(g1_project_list, desc="Fetching G1 Data (Violin)", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        for i, cov in enumerate(trend):
            while len(g1_sessions) <= i: g1_sessions.append([])
            g1_sessions[i].append(cov)

    STEP = BOXPLOT_STEP
    max_len = max(len(g2_sessions), len(g1_sessions))
    plot_data = []
    
    indices = range(0, max_len, STEP)
    min_projects_limit = 100 

    for idx in indices:
        session_label = idx + 1
        
        cnt_a = len(g1_sessions[idx]) if idx < len(g1_sessions) else 0
        cnt_b = len(g2_sessions[idx]) if idx < len(g2_sessions) else 0
        
        if cnt_a < min_projects_limit or cnt_b < min_projects_limit:
            break
            
        for v in g2_sessions[idx]:
            plot_data.append({'Session': session_label, 'Coverage': v, 'Group': 'G2 (Initial)'})
        for v in g1_sessions[idx]:
            plot_data.append({'Session': session_label, 'Coverage': v, 'Group': 'G1 (No Corpus)'})
            
    if not plot_data:
        logger.warning("No data sufficient for violin plot.")
        return

    df_plot = pd.DataFrame(plot_data)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    my_pal = {'G1 (No Corpus)': '#66b3ff', 'G2 (Initial)': '#ff9999'}
    
    sns.violinplot(
        data=df_plot,
        x='Session',
        y='Coverage',
        hue='Group',
        split=True,
        inner='quartile', 
        palette=my_pal,
        ax=ax1,
        cut=0, 
        linewidth=1.0
    )

    ax1.set_ylabel('Coverage (%)')
    ax1.set_xlabel('Coverage Measurement Count')
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    
    unique_sessions = sorted(list(set(df_plot['Session'])))
    if len(unique_sessions) > 10:
         plt.xticks(rotation=45)

    ax1.legend(loc='upper right', fontsize='small', title=None)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'g2_g1_violin_comparison.{file_format}')
    plt.savefig(save_path, format=file_format, bbox_inches='tight')
    logger.info(f"Saved comparative violin plot to {save_path}")
    plt.close()

def get_coverage_deltas(db, group_2_3_4_df, groups):
    """Analysis 2: Pre/Post Difference"""
    target_group_c = groups['group3'].union(groups['group4'])

    print("\n=== Analysis 2: Pre/Post Corpus Introduction Difference Analysis (Group C: Strict Filter Applied) ===")
    
    deltas = {
        'pre_deltas': {i: [] for i in range(ANALYSIS_ITERATIONS)}, 
        'post_deltas': {i: [] for i in range(1, ANALYSIS_ITERATIONS + 1)},
        'pre_groups': {i: [] for i in range(ANALYSIS_ITERATIONS)}, 
        'post_groups': {i: [] for i in range(1, ANALYSIS_ITERATIONS + 1)},
        'pre_coverages': {i: [] for i in range(ANALYSIS_ITERATIONS)},    # Added: Raw Coverage Values (Pre)
        'post_coverages': {i: [] for i in range(1, ANALYSIS_ITERATIONS + 1)} # Added: Raw Coverage Values (Post)
    } 
    
    projects_missing_pre_coverage = set() 
    processed_projects_list = set() 
    valid_project_count = 0

    for project_name, row in group_2_3_4_df.iterrows():
        if project_name not in target_group_c:
            continue

        corpus_time = row['corpus_commit_time']
        if pd.isna(corpus_time): continue
        corpus_date = corpus_time.date()

        group_num = None
        if project_name in groups['group4']: group_num = 4
        elif project_name in groups['group3']: group_num = 3
        
        if group_num is None: continue

        query_pre = f"""
            SELECT coverage FROM total_coverage 
            WHERE project = '{project_name.replace("'", "''")}' AND date < '{corpus_date}'
              AND coverage IS NOT NULL AND coverage > 0
            ORDER BY date DESC LIMIT {ANALYSIS_ITERATIONS}; 
        """
        results_pre = db.executeQuery("select", query_pre)
        pre_coverages = [float(res[0]) for res in results_pre] 

        query_post = f"""
            SELECT coverage FROM total_coverage 
            WHERE project = '{project_name.replace("'", "''")}' AND date >= '{corpus_date}' 
            AND coverage IS NOT NULL AND coverage > 0
            ORDER BY date ASC LIMIT {ANALYSIS_ITERATIONS}; 
        """
        results_post = db.executeQuery("select", query_post)
        post_coverages = [float(res[0]) for res in results_post]

        if len(pre_coverages) < ANALYSIS_ITERATIONS or len(post_coverages) < ANALYSIS_ITERATIONS:
            if len(pre_coverages) == 0:
                projects_missing_pre_coverage.add(project_name)
            continue
        
        valid_project_count += 1
        processed_projects_list.add(project_name)
        
        pre_1_baseline = pre_coverages[0]

        for i in range(ANALYSIS_ITERATIONS):
            deltas['pre_deltas'][i].append(pre_1_baseline - pre_coverages[i])
            deltas['pre_groups'][i].append(group_num)
            deltas['pre_coverages'][i].append(pre_coverages[i]) # Added
        
        for i in range(ANALYSIS_ITERATIONS):
            deltas['post_deltas'][i + 1].append(post_coverages[i] - pre_1_baseline)
            deltas['post_groups'][i + 1].append(group_num)
            deltas['post_coverages'][i + 1].append(post_coverages[i]) # Added
            
    print(f"Number of projects meeting conditions and analyzed: {valid_project_count}")
    return deltas, projects_missing_pre_coverage, processed_projects_list

def summarize_p_value_trends_and_stats(p_values, g2_stats_list, g1_stats_list, alpha=0.05):
    """
    Summarize and display the significant difference rate of Brunner-Munzel test, 
    and the ratio of G2 > G1 in each statistic (Q1, Median, Q3).
    Assumes p_values, g2_stats_list, g1_stats_list are all filtered data for valid periods.
    Also calculates Spearman rank correlation for Group A vs Group B time series data.
    """
    logger.info("Summarizing trends and stats...")
    
    valid_n = len(p_values)
    if valid_n == 0:
        logger.warning("No valid data to summarize.")
        return

    # 1. Significant difference (p < 0.05)
    sig_count = 0
    valid_p_count = 0
    for p in p_values:
        if not np.isnan(p):
            valid_p_count += 1
            if p < alpha:
                sig_count += 1
    
    # 2. Stats comparison (G2 > G1) & Data extraction
    q1_win_count = 0
    med_win_count = 0
    q3_win_count = 0
    comparison_n = 0
    
    g2_q1_seq, g2_med_seq, g2_q3_seq = [], [], []
    g1_q1_seq, g1_med_seq, g1_q3_seq = [], [], []

    for s2, s1 in zip(g2_stats_list, g1_stats_list):
        if s2 and s1 and len(s2) == 3 and len(s1) == 3:
            if np.isnan(s2).any() or np.isnan(s1).any():
                continue
            
            comparison_n += 1
            if s2[0] > s1[0]: q1_win_count += 1
            if s2[1] > s1[1]: med_win_count += 1
            if s2[2] > s1[2]: q3_win_count += 1
            
            g2_q1_seq.append(s2[0])
            g2_med_seq.append(s2[1])
            g2_q3_seq.append(s2[2])
            
            g1_q1_seq.append(s1[0])
            g1_med_seq.append(s1[1])
            g1_q3_seq.append(s1[2])

    print("\n=== Trend Analysis Summary (Trend Summary) ===")
    print(f"Target Valid Period: 1 ~ {valid_n} Sessions")
    
    # BM Test
    if valid_p_count > 0:
        print(f"Brunner-Munzel Test Significant Difference (p<0.05) Rate: {sig_count}/{valid_p_count} ({sig_count/valid_p_count*100:.2f}%)")
        
        # Find the first occurrence of significant difference
        first_sig_idx = -1
        first_sig_p = None
        for i, p in enumerate(p_values):
            if not np.isnan(p) and p < alpha:
                first_sig_idx = i + 1
                first_sig_p = p
                break
        
        if first_sig_idx != -1:
            print(f"First significant difference detected at: {first_sig_idx}th session (p={first_sig_p:.4e})")
        else:
            print("No significant difference detected.")
    else:
        print("Brunner-Munzel Test: No valid calculation results")

    # Comparison
    if comparison_n > 0:
        print(f"Group B > Group A Ratio (N={comparison_n}):")
        print(f"  - Q1               : {q1_win_count}/{comparison_n} ({q1_win_count/comparison_n*100:.2f}%)")
        print(f"  - Median           : {med_win_count}/{comparison_n} ({med_win_count/comparison_n*100:.2f}%)")
        print(f"  - Q3               : {q3_win_count}/{comparison_n} ({q3_win_count/comparison_n*100:.2f}%)")
        
        # 3. Spearman Correlation (Stats vs Iteration Count)
        try:
            # Iteration sequence (1, 2, ..., N)
            iterations = np.arange(1, comparison_n + 1)
            
            print(f"\nSpearman Rank Correlation with Coverage Measurement Count (N={comparison_n}):")
            
            # Helper to print
            def print_corr(name, data):
                c, p = spearmanr(iterations, data)
                print(f"  - {name:<15} : corr={c:.4f}, p-value={p:.4e}")

            print(" [Group A (No Corpus)]")
            print_corr("Q1", g1_q1_seq)
            print_corr("Median", g1_med_seq)
            print_corr("Q3", g1_q3_seq)

            print(" [Group B (Initial Corpus)]")
            print_corr("Q1", g2_q1_seq)
            print_corr("Median", g2_med_seq)
            print_corr("Q3", g2_q3_seq)
            
        except Exception as e:
            logger.error(f"Failed to calculate spearmanr: {e}")
            print("Spearman Rank Correlation: Calculation Error")

    else:
        print("Stats Comparison: No valid data")
        
    print("============================================\n")

def analyze_g2_g1_trends(db, g2_project_list, g1_project_list, output_dir, percentiles_list):
    """Analysis 3: G2 vs G1 Median/IQR Trend Calculation + Brunner-Munzel Test"""
    print("\n=== Analysis 3: G2 vs G1 Coverage Trend Analysis ===")
    
    g2_sessions, g1_sessions = [[]], [[]]
    max_sessions = 0
    
    for project_name in tqdm(g2_project_list, desc="G2 Projects", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        max_sessions = max(max_sessions, len(trend))
        for i, cov in enumerate(trend):
            while len(g2_sessions) <= i: g2_sessions.append([])
            g2_sessions[i].append(cov)
    
    for project_name in tqdm(g1_project_list, desc="G1 Projects", leave=False):
        trend = get_full_coverage_trend(db, project_name)
        if not trend: continue
        max_sessions = max(max_sessions, len(trend))
        for i, cov in enumerate(trend):
            while len(g1_sessions) <= i: g1_sessions.append([])
            g1_sessions[i].append(cov)

    if len(g2_sessions) < max_sessions:
        g2_sessions.extend([[] for _ in range(max_sessions - len(g2_sessions))])
    if len(g1_sessions) < max_sessions:
        g1_sessions.extend([[] for _ in range(max_sessions - len(g1_sessions))])
    
    csv_header = ["Session"]
    for group in ['G2', 'G1']:
        for p in percentiles_list:
            csv_header.append(f"{group}_{p}")
        csv_header.append(f"{group}_Count")
        
    csv_data = [csv_header]
    p_values = []
    
    # Stats history (before filtering)
    g2_stats_raw = []
    g1_stats_raw = []
    
    # Counts for filtering
    counts_g2 = []
    counts_g1 = []
    
    for i in range(max_sessions):
        g2_d, g1_d = g2_sessions[i], g1_sessions[i]
        c2, c1 = len(g2_d), len(g1_d)
        counts_g2.append(c2)
        counts_g1.append(c1)
        
        row = [i + 1]
        
        # Calculate Stats
        current_g2_stats = [np.nan] * len(percentiles_list)
        if g2_d: 
            current_g2_stats = list(np.percentile(g2_d, percentiles_list))
        row.extend(current_g2_stats + [c2])
        g2_stats_raw.append(current_g2_stats)
        
        current_g1_stats = [np.nan] * len(percentiles_list)
        if g1_d: 
            current_g1_stats = list(np.percentile(g1_d, percentiles_list))
        row.extend(current_g1_stats + [c1])
        g1_stats_raw.append(current_g1_stats)
        
        csv_data.append(row)

        # Brunner-Munzel Test
        p_val = np.nan
        if c2 >= 5 and c1 >= 5:
            try:
                _, p_val = brunnermunzel(g2_d, g1_d, alternative='two-sided')
            except Exception:
                pass
        p_values.append(p_val)

    # Filtering: Up to the last point where both maintain 100 or more
    # i.e., find the last index where (c2 >= 100 AND c1 >= 100) holds
    last_valid_idx = -1
    for i in range(len(p_values)):
        if counts_g2[i] >= 100 and counts_g1[i] >= 100:
            last_valid_idx = i
            
    if last_valid_idx != -1:
        # Slice (0 to last_valid_idx inclusive)
        target_p_values = p_values[:last_valid_idx+1]
        target_g2_stats = g2_stats_raw[:last_valid_idx+1]
        target_g1_stats = g1_stats_raw[:last_valid_idx+1]
        
        final_idx = last_valid_idx
        logger.info(f"Filtering analysis up to session {final_idx+1} (Limit: BOTH G1 and G2 >= 100).")
        logger.info(f"At limit ({final_idx+1}): G1 Count={counts_g1[final_idx]}, G2 Count={counts_g2[final_idx]}")
        if final_idx + 1 < len(counts_g1):
             logger.info(f"Next ({final_idx+2}): G1 Count={counts_g1[final_idx+1]}, G2 Count={counts_g2[final_idx+1]}")
    else:
        # If no session meets the condition
        target_p_values = []
        target_g2_stats = []
        target_g1_stats = []
        logger.warning("No sessions met the condition (Either G1 or G2 >= 100). No summary reported.")

    summarize_p_value_trends_and_stats(target_p_values, target_g2_stats, target_g1_stats)

    df = pd.DataFrame(csv_data[1:], columns=csv_header)
    return df

def plot_coverage_distribution(group2_data, group1_data, output_dir, file_format='pdf'):
    """Analysis 1 Visualization: Violin Plot (Note: Only boxplots were styled changes, existing logic is maintained)"""
    data2 = pd.DataFrame({'coverage': group2_data, 'group': 'Group 2 (Initial Corpus)'})
    data1 = pd.DataFrame({'coverage': group1_data, 'group': 'Group 1 (No Corpus)'})
    if data2.empty and data1.empty: return

    plot_df = pd.concat([data2, data1], ignore_index=True)
    plt.figure(figsize=(5, 3), dpi=100)
    sns.set_style("whitegrid")
    
    group_order = ['Group 1 (No Corpus)', 'Group 2 (Initial Corpus)']
    palette = {'Group 2 (Initial Corpus)': '#66b3ff', 'Group 1 (No Corpus)': '#ff9999'}
    
    ax = sns.violinplot(x='group', y='coverage', hue='group', legend=False, data=plot_df, 
                        palette=palette, inner="quartile", cut=0, order=group_order)
    
    ax.set_ylabel('Initial Coverage (%)', fontsize=12)
    ax.set_xlabel('')
    ax.set_xticklabels([f"Group 1 (No Corpus)\n(N={len(group1_data)})", f"Group 2 (Initial Corpus)\n(N={len(group2_data)})"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'initial_coverage_g2_g1_comparison.{file_format}'), format=file_format)
    plt.close()

def plot_coverage_deltas(deltas, output_dir, file_format='pdf'):
    """Analysis 2 Visualization: Pre/Post Delta Boxplot (Two-color version)"""
    plot_data = []
    group_keys = []
    
    for i in range(ANALYSIS_ITERATIONS - 1, -1, -1):
        key = f"t=-{i+1}"
        group_keys.append(key)
        for j, val in enumerate(deltas['pre_deltas'][i]): 
            plot_data.append({'key': key, 'val': val, 'type': 'Pre', 'group': deltas['pre_groups'][i][j]})

    for i in range(1, ANALYSIS_ITERATIONS + 1):
        key = f"t={i}"
        group_keys.append(key)
        for j, val in enumerate(deltas['post_deltas'][i]): 
            plot_data.append({'key': key, 'val': val, 'type': 'Post', 'group': deltas['post_groups'][i][j]})

    if not plot_data: return
    
    # Calculation and display of median values (Chronological order: Pre-N -> ... -> Pre-1 -> Post-1 -> ... -> Post-N)
    print("\n--- Coverage Median for Each Step (Group C) ---")
    
    # Pre (Pre-N ... Pre-1)
    # pre_coverages[i] is data for Pre-(i+1).
    # Time series is Pre-N, Pre-(N-1), ..., Pre-1, so process indices in reverse order
    for i in reversed(range(ANALYSIS_ITERATIONS)):
        step_label = f"Pre-{i+1}"
        cov_data = deltas['pre_coverages'][i]
        if cov_data:
            median_val = np.median(cov_data)
            print(f" {step_label:<7}: {median_val:.2f} (N={len(cov_data)})")
        else:
            print(f" {step_label:<7}: N/A")
            
    # Post (Post-1 ... Post-N)
    for i in range(1, ANALYSIS_ITERATIONS + 1):
        step_label = f"Post-{i}"
        cov_data = deltas['post_coverages'][i]
        if cov_data:
            median_val = np.median(cov_data)
            print(f" {step_label:<7}: {median_val:.2f} (N={len(cov_data)})")
        else:
            print(f" {step_label:<7}: N/A")
            
    print("----------------------------------\n")

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(5, 3))
    sns.set_style("whitegrid")
    
    # Pre and Post color definitions
    color_pre = "#ffcc99"
    color_post = "#99ff99"
    
    sns.boxplot(x='key', y='val', hue='type', dodge=False, legend=False, data=plot_df, 
                order=group_keys, 
                palette={'Pre': color_pre, 'Post': color_post}, 
                fliersize=2, linewidth=DELTA_EDGE_LINEWIDTH)

    # *** Force style unification: Maintain Hue with fill_color=None, unify edge to black ***
    apply_two_color_boxplot(plt.gca(), fill_color=None, edge_color=BOXPLOT_EDGE_COLOR, alpha=0.6, linewidth=DELTA_EDGE_LINEWIDTH)
    
    xtick_labels = []
    for key in group_keys:
        xtick_labels.append(f"{key[2:]}")

    plt.xticks(range(len(group_keys)), xtick_labels)
    
    plt.ylim(-50, 50)
    plt.ylabel('Coverage Delta (Relative to Pre-1)')
    plt.xlabel('Time Step (t)')
    plt.axhline(0, ls='--', color='black', linewidth=1.0)
    plt.axvline(ANALYSIS_ITERATIONS - 0.5, ls=':', color='red', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'coverage_delta_timeseries_linear.{file_format}'), format=file_format)
    plt.close()

def plot_log2_histogram(time_data_df, output_dir, file_format='pdf'):
    """Analysis 4 Visualization: Days Histogram"""
    if time_data_df.empty: return
    
    days = time_data_df['time_elapsed_seconds'] / 86400.0
    days = days.clip(lower=0.01)
    
    plt.figure(figsize=(5, 3))
    sns.set_style("whitegrid")
    
    bins = np.logspace(np.log2(days.min()), np.log2(days.max()), num=30, base=2.0)
    ax = sns.histplot(x=days, bins=bins, kde=False)
    
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Days Elapsed (Log2 Scale)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'g2_g3_g4_days_histogram.{file_format}'), format=file_format)
    plt.close()

def plot_g2_g1_trends(df, output_dir, file_format='pdf'):
    """Analysis 3 Visualization: G2 vs G1 Trend"""
    logger.info("Generating G2 vs G1 trend plots...")
    
    if df.empty: return

    G1_S_REDUCTION = 80
    G1_L_REDUCTION = 0
    G2_S_REDUCTION = 0
    G2_L_REDUCTION = 40
    
    default_style = {'color': '#000000', 'linewidth': 1.0, 'linestyle': '-'}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 

    PLOT_ORDER = sorted(PERCENTILE_STYLE_MAP.keys(), reverse=True) 

    handles_a, labels_a = [], []
    handles_b, labels_b = [], []
    
    for p in PLOT_ORDER:
        col_name_g2 = f'G2_{p}'
        col_name_g1 = f'G1_{p}'
        
        if col_name_g2 not in df.columns or p not in PERCENTILE_STYLE_MAP: continue
            
        style_config = PERCENTILE_STYLE_MAP.get(p, default_style)
        base_color = style_config['color']
        label_suffix = style_config.get('label', f'{p}%')
        
        line_conf = {
            'linewidth': style_config.get('linewidth', 1.0),
            'linestyle': style_config.get('linestyle', '-')
        }
        
        g2_color = adjust_color_hls_reduction(base_color, G2_L_REDUCTION, G2_S_REDUCTION)
        line_g2, = ax.plot(df['Session'], df[col_name_g2], color=g2_color, label='_nolegend_', **line_conf)

        g1_color = adjust_color_hls_reduction(base_color, G1_L_REDUCTION, G1_S_REDUCTION)
        line_g1, = ax.plot(df['Session'], df[col_name_g1], color=g1_color, label='_nolegend_', **line_conf)

        handles_a.append(line_g1)
        labels_a.append(f'A {label_suffix}')
        handles_b.append(line_g2)
        labels_b.append(f'B {label_suffix}')

    legend_handles = handles_a + handles_b
    legend_labels = labels_a + labels_b

    ax.set_xlabel('Coverage Measurement Session (Count)')
    ax.set_ylabel('Coverage (%)')
    ax.legend(legend_handles, legend_labels, ncol=2, loc='upper left', fontsize='small')
    
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    
    min_projects = 100
    valid_sessions = df[(df['G2_Count'] >= min_projects) & (df['G1_Count'] >= min_projects)]
    if not valid_sessions.empty:
        max_valid_session = valid_sessions['Session'].max()
        ax.set_xlim(left=0, right=max_valid_session)

    plt.tight_layout() 
    output_path = os.path.join(output_dir, f'g2_g1_coverage_trends_custom_color.{file_format}')
    plt.savefig(output_path, format=file_format)
    plt.close() 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    config = ConfigParser()
    config.read(DB_CONFIG_FILE)
    try:
        db_conf = config["POSTGRES"]
        db = DB(database=db_conf.get("POSTGRES_DB"), user=db_conf.get("POSTGRES_USER"),
                password=db_conf.get("POSTGRES_PASSWORD"), host=db_conf.get("POSTGRES_IP"),
                port=db_conf.get("POSTGRES_PORT"))
        db.connect()
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        sys.exit(1)

    eligible_projects = get_eligible_projects_from_db(db)
    if not eligible_projects: return

    groups, g234_df, times_df = categorize_projects_and_get_times(eligible_projects, CORPUS_ANALYSIS_CSV)
    if not groups: return

    trend_df = analyze_g2_g1_trends(db, groups['group2'], groups['group1'], OUTPUT_DIR, PERCENTILES_TO_CALCULATE)

    deltas, missing_pre, processed_projects = get_coverage_deltas(db, g234_df, groups)
    
    add_g2 = missing_pre if missing_pre and INCLUDE_MISSING_PRE_IN_G2 else set()
    g2_cov, g1_cov, final_g2, initial_cov_stats = analyze_g2_vs_g1_initial_coverage(db, groups['group2'], groups['group1'], add_g2)

    if initial_cov_stats:
        logger.info(f"Initial coverage stats: {initial_cov_stats}")

    if deltas: plot_coverage_deltas(deltas, OUTPUT_DIR, FILE_FORMAT)
    # plot_coverage_distribution(g2_cov, g1_cov, OUTPUT_DIR, FILE_FORMAT)  # initial_coverage_g2_g1_comparison.pdf (DISABLED)
    
    # if processed_projects and not times_df.empty:  # g2_g3_g4_days_histogram.pdf (DISABLED)
    #     plot_log2_histogram(times_df.loc[list(processed_projects)], OUTPUT_DIR, FILE_FORMAT)
        
    # if trend_df is not None:  # g2_g1_coverage_trends_custom_color.pdf (DISABLED)
    #     plot_g2_g1_trends(trend_df, OUTPUT_DIR, FILE_FORMAT)

    plot_g2_g1_comparative_boxplot(db, groups['group2'], groups['group1'], OUTPUT_DIR, FILE_FORMAT)
    
    # plot_g2_g1_comparative_violin(db, groups['group2'], groups['group1'], OUTPUT_DIR, FILE_FORMAT)  # g2_g1_violin_comparison.pdf (DISABLED)

    # plot_nested_variable_width_boxplot(  # nested_boxplot_comparison.pdf (DISABLED)
    #     db, 
    #     groups['group1'],  # Group A (Wide)
    #     groups['group2'],  # Group B (Narrow)
    #     OUTPUT_DIR, 
    #     FILE_FORMAT
    # )

    logger.info("--- Analysis Finished ---")

if __name__ == "__main__":
    main()