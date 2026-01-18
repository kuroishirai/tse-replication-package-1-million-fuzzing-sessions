import sys
import os
import csv
from configparser import ConfigParser
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import MaxNLocator
try:
    from matplotlib_venn import venn2
except Exception:
    venn2 = None
    logging.getLogger(__name__).warning("Optional package 'matplotlib-venn' not found — Venn diagram will be skipped. Install with: pip install matplotlib-venn")
import matplotlib.colors as mcolors
from PIL import Image
from collections import defaultdict

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration (Assuming CWD=/app) ---
CWD = os.getcwd()
MODULE_PATH = os.path.join(CWD, 'program/__module')
DB_CONFIG_FILE = os.path.join(CWD, 'program/envFile.ini')
CORPUS_ANALYSIS_CSV = os.path.join(CWD, 'data/processed_data/csv/project_corpus_analysis.csv')

# --- User-specified output directory ---
OUTPUT_DIR = os.path.join(CWD, 'data/result_data/rq4/bug')

# --- Graph save format (pdf or png) ---
FILE_FORMAT = 'pdf'

# --- Global variables ---
ANALYSIS_ITERATIONS = 7 
DAYS_THRESHOLD = 7
DATE_LIMIT = '2025-01-08'
INCLUDE_MISSING_PRE_IN_G2 = False  # Whether to add Group 4 projects with missing Pre data to Group 2 in analysis (True: add, False: do not add)

# Add module path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

try:
    from dbFile import DB
except ImportError:
    logger.error(f"Error: Could not import 'dbFile' from '{MODULE_PATH}'. Ensure it is accessible.")
    sys.exit(1)


def get_group_name(group_key):
    """Helper function to return readable name from group key"""
    if group_key == 'group1': return 'Group A (No Corpus)'
    if group_key == 'group2': return 'Group B (Initial Corpus)'
    if group_key == 'group3': return 'Group D (1-5 Day Corpus)'
    if group_key == 'group4': return 'Group C (>5 Day Corpus)'
    return group_key


def get_eligible_projects_from_db(db):
    """
    Get projects from DB that meet RQ1 criteria (coverage 365 days or more).
    """
    query = f"""
        SELECT project
        FROM total_coverage
        WHERE coverage IS NOT NULL AND coverage > 0 AND date < '{DATE_LIMIT}'
        GROUP BY project
        HAVING COUNT(*) >= 365
    """
    projects_result = db.executeQuery("select", query)
    return {project[0] for project in projects_result}

def categorize_projects_and_get_g4_time(eligible_projects_set, corpus_csv_path):
    """
    Load CSV, classify target projects into G1, G2, G4, and return G4 introduction time DF.
    Use RQ1's full project set; projects not in CSV are assigned to G1.
    """
    try:
        df = pd.read_csv(corpus_csv_path)
        df['corpus_commit_time'] = pd.to_datetime(df['corpus_commit_time'], errors='coerce', utc=True)
    except FileNotFoundError:
        logger.error(f"Error: Corpus analysis file not found: '{corpus_csv_path}'")
        return None, None
    
    filtered_df = df[df['project_name'].isin(eligible_projects_set)].copy()

    # Classification of projects in CSV
    cat_null_g1 = filtered_df['time_elapsed_seconds'].isna()
    cat_same_time_g2 = (filtered_df['time_elapsed_seconds'] == 0) & (~cat_null_g1)
    cat_over_1_day_under_threshold_g3 = (filtered_df['time_elapsed_seconds'] > 0) & (filtered_df['time_elapsed_seconds'] < DAYS_THRESHOLD * 86400) & (~cat_null_g1)
    cat_over_threshold_g4 = (filtered_df['time_elapsed_seconds'] >= DAYS_THRESHOLD * 86400) & (~cat_null_g1)

    # Groups of projects in CSV
    csv_projects_in_groups = {
        'group1': set(filtered_df[cat_null_g1]['project_name']),
        'group2': set(filtered_df[cat_same_time_g2]['project_name']),
        'group3': set(filtered_df[cat_over_1_day_under_threshold_g3]['project_name']),
        'group4': set(filtered_df[cat_over_threshold_g4]['project_name'])
    }

    # All projects in eligible_projects_set not in CSV are G1
    csv_project_names = set(filtered_df['project_name'])
    missing_projects = eligible_projects_set - csv_project_names
    csv_projects_in_groups['group1'].update(missing_projects)

    project_groups = csv_projects_in_groups
    
    g4_time_df = filtered_df[cat_over_threshold_g4][['project_name', 'corpus_commit_time']].set_index('project_name')

    logger.info(f"Projects categorized: G1={len(project_groups['group1'])}, G2={len(project_groups['group2'])}, G3={len(project_groups['group3'])}, G4={len(project_groups['group4'])}")
    
    return project_groups, g4_time_df


def get_project_fuzzing_builds(db, project_name):
    """
    Get all Fuzzing execution logs (datetime) for 1 project (ascending). Regardless of success/failure.
    """
    query = f"""
        SELECT timecreated
        FROM buildlog_data 
        WHERE project = '{project_name.replace("'", "''")}'
          AND build_type = 'Fuzzing'
          AND timecreated < '{DATE_LIMIT}'
        ORDER BY timecreated ASC;
    """
    results = db.executeQuery("select", query)
    return [(i + 1, pd.to_datetime(row[0], utc=True)) for i, row in enumerate(results) if row[0] is not None]


def get_project_fixed_issues(db, project_name):
    """
    Get all fixed bugs (report datetime) for 1 project.
    """
    query = f"""
        SELECT number, rts
        FROM issues
        WHERE project = '{project_name.replace("'", "''")}'
          AND rts < '{DATE_LIMIT}'
          AND status IN ('Fixed','Fixed (Verified)')
        ORDER BY rts ASC;
    """
    results = db.executeQuery("select", query)
    return [(row[0], pd.to_datetime(row[1], utc=True)) for row in results if row[1] is not None]


def calculate_and_save_stats(g1_stats, g2_stats, output_dir):
    """
    Calculate G1/G2 statistics and save to CSV.
    Same as RQ1, remove iterations with fewer than 100 projects.
    """
    csv_data = []
    
    # Fix: Dynamically determine max iteration count from actual data
    max_iter_g1 = max(g1_stats.keys()) if g1_stats else 0
    max_iter_g2 = max(g2_stats.keys()) if g2_stats else 0
    max_iter = max(max_iter_g1, max_iter_g2)
    
    logger.info(f"Max iteration found in data: {max_iter}")

    # Same filtering as RQ1: Remove if G1 or G2 total_projects < 100 per iteration
    min_project_threshold = 100
    valid_iterations = set()
    for i in range(1, max_iter + 1):
        g1_total, _ = g1_stats.get(i, [0, set()])
        g2_total, _ = g2_stats.get(i, [0, set()])
        if g1_total >= min_project_threshold and g2_total >= min_project_threshold:
            valid_iterations.add(i)
    
    logger.info(f"Filtering iterations with fewer than {min_project_threshold} projects in either group. Retained {len(valid_iterations)} iterations.")

    logger.info("\n--- G1/G2 Detection Trend Statistics ---")
    logger.info(f"| {'Iter':<4} | {'G1 Total':<8} | {'G1 Rate':<7} | {'G2 Total':<8} | {'G2 Rate':<7} |")
    logger.info(f"|{'-'*6}|{'-'*10}|{'-'*9}|{'-'*10}|{'-'*9}|")

    user_log_max = 100
    for i in sorted(valid_iterations):
        g1_total, g1_detected_set = g1_stats.get(i, [0, set()])
        g2_total, g2_detected_set = g2_stats.get(i, [0, set()])
        
        g1_rate = len(g1_detected_set) / g1_total * 100 if g1_total > 0 else 0
        g2_rate = len(g2_detected_set) / g2_total * 100 if g2_total > 0 else 0
        
        csv_data.append([i, g1_total, len(g1_detected_set), g1_rate, g2_total, len(g2_detected_set), g2_rate])
        
        if i <= user_log_max:
            logger.info(f"| {i:<4} | {g1_total:<8} | {g1_rate:>6.2f}% | {g2_total:<8} | {g2_rate:>6.2f}% |")

    stats_csv_path = os.path.join(output_dir, 'rq4_g1_g2_detection_trend.csv')
    csv_header = ['Iteration', 'G1_Total_Projects', 'G1_Detected_Count', 'G1_Detection_Rate_pct', 
                  'G2_Total_Projects', 'G2_Detected_Count', 'G2_Detection_Rate_pct']
    with open(stats_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    logger.info(f"Saved G1/G2 trend statistics to: {stats_csv_path}")

    return pd.DataFrame(csv_data, columns=csv_header)


def create_detection_rate_trend_graph(df, output_path, file_format='pdf'):
    """
    Create and save G1 and G2 bug detection rate trend as line graph.
    """
    if df.empty:
        logger.warning("No data available to create the trend graph.")
        return
        
    df_filtered = df.copy()
    
    plt.figure(figsize=(5, 3))
    
    plt.plot(df_filtered['Iteration'], df_filtered['G1_Detection_Rate_pct'], 
             color='#1f77b4', linestyle='-', label='Group A (No Corpus)', linewidth=1,
             marker='o', markersize=1)
    
    plt.plot(df_filtered['Iteration'], df_filtered['G2_Detection_Rate_pct'], 
             color='#ff7f0e', linestyle='-', label='Group B (Initial Corpus)', linewidth=1, alpha=0.7,
             marker='o', markersize=1) 
    
    plt.xlabel('Fuzzing Session')
    plt.ylabel('Percentage of Projects Detecting Bugs', y=0.45)
    # plt.title('Bug Detection Rate Trend (A vs B)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xlim(left=1)
    
    if df_filtered['Iteration'].max() > 500:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='upper'))
    
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, format=file_format)
    plt.close()
    logger.info(f"Saved detection rate trend graph to: {output_path}")


def analyze_g4_corpus_introduction_iteration(db, project_groups, g4_time_df):
    """
    Calculate at which Fuzzing iteration corpus was introduced for G4 projects.
    """
    g4_projects = project_groups['group4']
    introduction_iterations = {}

    logger.info("\n--- Analyzing Group C Corpus Introduction Iteration ---")
    
    for project_name in tqdm(g4_projects, desc="Calculating G4 Iterations"):
        if project_name not in g4_time_df.index:
            continue
            
        corpus_time = g4_time_df.loc[project_name]['corpus_commit_time']
        if pd.isna(corpus_time):
            continue
            
        builds = get_project_fuzzing_builds(db, project_name) 
        
        if not builds:
            introduction_iterations[project_name] = 0
            continue
            
        k = sum(1 for _, build_time in builds if build_time < corpus_time)
        introduction_iterations[project_name] = k
        
    df_result = pd.DataFrame(
        list(introduction_iterations.items()), 
        columns=['Project', 'Introduction_Iteration']
    ).sort_values(by='Introduction_Iteration', ascending=True)

    valid_data = df_result[df_result['Introduction_Iteration'] > 0]
    
    logger.info(f"[RESULT] Total Group C Projects analyzed: {len(df_result)}")
    if not valid_data.empty:
        logger.info(f"[RESULT] Introduction Iteration (N={len(valid_data)}):")
        logger.info(f"  - Mean: {valid_data['Introduction_Iteration'].mean():.2f}")
        logger.info(f"  - Median: {valid_data['Introduction_Iteration'].median():.1f}")
        logger.info(f"  - Min: {valid_data['Introduction_Iteration'].min()}")
        logger.info(f"  - Max: {valid_data['Introduction_Iteration'].max()}")
    else:
        logger.info("[RESULT] No projects found with corpus introduction after the first fuzzing session.")
        
    csv_path = os.path.join(OUTPUT_DIR, 'rq4_gc_introduction_iteration.csv')
    df_result.to_csv(csv_path, index=False)
    logger.info(f"Saved Group C introduction iteration data to: {csv_path}")

    logger.info("\n[RESULT] Top 5 Projects (Earliest Corpus Introduction):")
    logger.info(df_result.head(5).to_string(index=False)) 
    
    logger.info("\n[RESULT] Bottom 5 Projects (Latest Corpus Introduction):")
    logger.info(df_result.tail(5).to_string(index=False)) 

    return df_result


def analyze_rq4_detection_trends(db, project_groups, g4_time_df, additional_group2_projects=None):
    """
    Execute G1/G2 bug detection trend and G4 pre/post N analysis.
    Fix: For G4, only target projects with complete data for ANALYSIS_ITERATIONS before and after.
    """
    if additional_group2_projects is None:
        additional_group2_projects = set()
    
    # For G1, G2
    g1_stats = defaultdict(lambda: [0, set()])
    g2_stats = defaultdict(lambda: [0, set()])
    
    # For G4: { relative_step (int): [True, False, False, ...] }
    g4_dynamic_data = defaultdict(list)
    
    # G4 Transition Analysis: List of {'project': name, 'pre': bool, 'post': bool}
    g4_transition_data = []
    
    missing_pre = set()
    
    logger.info("Processing G1 and G2 projects for detection trend...")
    
    for group_key in ['group1', 'group2']:
        stats = g1_stats if group_key == 'group1' else g2_stats
        
        projects = project_groups[group_key]
        if group_key == 'group2':
            projects = projects.union(additional_group2_projects)
        
        for project_name in tqdm(projects, desc=f"Processing {get_group_name(group_key)}"):
            builds = get_project_fuzzing_builds(db, project_name)
            issues = get_project_fixed_issues(db, project_name)
            
            if not builds:
                continue

            # (1) Update total_projects
            for iteration, _ in builds:
                stats[iteration][0] += 1
            
            # (2) Update detected_projects
            for issue_id, issue_time in issues:
                k = sum(1 for _, build_time in builds if build_time < issue_time)
                if k > 0:
                    stats[k][1].add(project_name)

    logger.info("Processing G4 projects for pre/post analysis (Fixed N filtering)...")
    
    N = ANALYSIS_ITERATIONS
    for project_name in tqdm(project_groups['group4'], desc="Processing G4 Projects"):
        if project_name not in g4_time_df.index:
            continue
            
        corpus_time = g4_time_df.loc[project_name]['corpus_commit_time']
        if pd.isna(corpus_time):
            continue
            
        builds = get_project_fuzzing_builds(db, project_name) 
        issues = get_project_fixed_issues(db, project_name) 
        
        # Identify builds before corpus introduction
        pre_build_indices = [i for i, (_, build_time) in enumerate(builds) if build_time < corpus_time]
        
        if not pre_build_indices:
            continue
            
        # idx_pre_last: Index of build immediately before corpus introduction
        idx_pre_last = pre_build_indices[-1]
        
        # --- Change: Check if N data points before and after are complete ---
        # Pre side: idx_pre_last Check if we can go back N items including this.
        
        if (idx_pre_last - (N - 1) < 0) or ((idx_pre_last + N) >= len(builds) - 1):
            missing_pre.add(project_name)
            continue # Excluding this project due to insufficient data

        # ----------------------------------------------------
        
        is_detected_pre_any = False
        is_detected_post_any = False

        # Aggregate for each relative step (Data existence is guaranteed at this point)
        for k in range(1, N + 1):
            # --- Pre side (Pre-k) ---
            idx_curr = idx_pre_last - (k - 1)
            
            # Start and end points of interval
            T_start = builds[idx_curr][1]
            T_end = builds[idx_curr + 1][1]
            
            detected_pre = any(T_start <= T_R < T_end for _, T_R in issues)
            g4_dynamic_data[-k].append(detected_pre)
            if detected_pre:
                is_detected_pre_any = True
            
            # --- Post side (Post-k) ---
            idx_curr_post = idx_pre_last + k
            
            T_start_post = builds[idx_curr_post][1]
            T_end_post = builds[idx_curr_post + 1][1]
            
            detected_post = any(T_start_post <= T_R < T_end_post for _, T_R in issues)
            g4_dynamic_data[k].append(detected_post)
            if detected_post:
                is_detected_post_any = True

        g4_transition_data.append({
            'project': project_name,
            'pre': is_detected_pre_any,
            'post': is_detected_post_any
        })
        
    return g1_stats, g2_stats, g4_dynamic_data, missing_pre, g4_transition_data


def analyze_g4_trend(g4_dynamic_data, output_dir, g4_transition_data=None):
    """
    Calculate G4 pre/post N trend, output logs and graph.
    Due to strict filtering, denominator (n) for each step should be constant.
    """
    N = ANALYSIS_ITERATIONS
    
    if not g4_dynamic_data:
        logger.warning(f"Skipping G4 Trend Analysis: No data available.")
        return 0, 0
        
    trend_data = [] 
    
    logger.info(f"\n--- Group C (Introduced Corpus) Pre-N/Post-N Trend Analysis (Fixed n) ---")
    logger.info(f"| {'Step':<7} | {'n (Total)':<9} | {'DetCnt':<6} | {'Rate':<6} |")
    logger.info(f"|{'-'*9}|{'-'*11}|{'-'*8}|{'-'*8}|")

    # Step order: -N, ..., -1, 1, ..., N
    steps = sorted([s for s in g4_dynamic_data.keys() if -N <= s <= N and s != 0])
    
    # Build data for graph
    for step in steps:
        results = g4_dynamic_data[step]
        n_total = len(results)
        
        if n_total == 0:
            continue

        det_count = sum(1 for r in results if r)
        rate = (det_count / n_total) * 100
        
        label_prefix = "Pre" if step < 0 else "Post"
        step_abs = abs(step)
        session_label = f"{label_prefix}-{step_abs}"

        
        # Sort index
        sort_idx = (step + N) if step < 0 else (step + N - 1)

        trend_data.append({
            'Sort_Index': sort_idx,
            'Step_Raw': step,
            'Session': session_label,
            'Total_Projects_at_Session': n_total, 
            'Session_Detected_Count': det_count,
            'Session_Detection_Rate_pct': rate
        })
        
        logger.info(f"| {session_label:<7} | {n_total:<9} | {det_count:<6} | {rate:>5.2f}% |")
        
    trend_df = pd.DataFrame(trend_data).sort_values('Sort_Index')

    # Overall Pre/Post average rate
    all_pre_results = []
    for s in range(-N, 0):
        all_pre_results.extend(g4_dynamic_data.get(s, []))
    
    all_post_results = []
    for s in range(1, N + 1):
        all_post_results.extend(g4_dynamic_data.get(s, []))

    overall_pre_rate = (sum(all_pre_results) / len(all_pre_results) * 100) if all_pre_results else 0
    overall_post_rate = (sum(all_post_results) / len(all_post_results) * 100) if all_post_results else 0
    
    max_n = trend_df['Total_Projects_at_Session'].max() if not trend_df.empty else 0

    # Compute transition counts if transition data is provided
    transition_counts = None
    if g4_transition_data:
        c_i_iii = 0 # Pre Yes, Post Yes (Intersection)
        c_i_iv  = 0 # Pre Yes, Post No (Pre Only)
        c_ii_iii = 0 # Pre No, Post Yes (Post Only)
        c_ii_iv  = 0 # Pre No, Post No (Outside)
        for item in g4_transition_data:
            pre = item.get('pre')
            post = item.get('post')
            if pre and post:
                c_i_iii += 1
            elif pre and not post:
                c_i_iv += 1
            elif not pre and post:
                c_ii_iii += 1
            elif not pre and not post:
                c_ii_iv += 1
        transition_counts = {
            'no_detection': c_ii_iv,
            'pre_only': c_i_iv,
            'pre_and_post': c_i_iii,
            'post_only': c_ii_iii
        }

    create_g4_trend_graph(trend_df, max_n, N, os.path.join(output_dir, f'rq4_gc_detection_trend.{FILE_FORMAT}'), file_format=FILE_FORMAT, transition_counts=transition_counts)
    
    return overall_pre_rate, overall_post_rate


def create_g4_trend_graph(df, max_n, N, output_path, file_format='pdf', transition_counts=None):
    """
    Create and save G4 pre/post N trend as line graph.
    """
    if df.empty:
        return

    plt.figure(figsize=(5, 3)) 
    
    plt.plot(df['Sort_Index'], df['Session_Detection_Rate_pct'], 
             color='#2ca02c', linestyle='-', marker='o', markersize=5, linewidth=1.5)
    
    # Boundary line
    boundary_x = (N - 1) + 0.5
    plt.axvline(x=boundary_x, color='r', linestyle='--', linewidth=1.0, label='Corpus Specification')
    
    plt.xlabel('Fuzzing Session (Relative Step: Pre/Post)')
    plt.ylabel('Percentage of Projects Detecting Bugs', y=0.45)
    
    # Create X-axis labels: "Pre-1\n(n=120)" 
    xticks_labels = []
    xticks_indices = []

    for _, row in df.iterrows():
        label = row['Session']
        n_val = row['Total_Projects_at_Session']
        short_label = label.replace("Pre-", "-").replace("Post-", "+")
        xticks_labels.append(f"{short_label}")
        xticks_indices.append(row['Sort_Index'])

    plt.xticks(xticks_indices, xticks_labels, rotation=0)

    plt.ylim(0, 32)
    # Place legend in the upper-left (user requested)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=0.1)
    # If transition counts are provided, display them in the bottom-right corner
    if transition_counts:
        ax = plt.gca()
        text_lines = [
            f"no detection: {transition_counts.get('no_detection', 0):>2} project",
            f"pre only detection: {transition_counts.get('pre_only', 0):>2} project",
            f"pre&post detection: {transition_counts.get('pre_and_post', 0):>2} project",
            f"post only detection: {transition_counts.get('post_only', 0):>2} project",
        ]
        text = "\n".join(text_lines)
        # Use RGBA for edgecolor to achieve semi-transparent border
        ax.text(0.98, 0.05, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=(0,0,0,0.35), linewidth=0.8))

    plt.savefig(output_path, format=file_format)
    plt.close()
    logger.info(f"Saved Group C trend graph to: {output_path}")

def create_detection_rate_difference_graph(df, output_path, file_format='pdf'):
    """
    Create and save G2 minus G1 detection rate difference trend as line graph.
    """
    if df.empty:
        logger.warning("No data available to create the difference trend graph.")
        return

    df_filtered = df.copy()

    # Calculate difference: G2 - G1
    df_filtered['Difference_Rate_pct'] = df_filtered['G2_Detection_Rate_pct'] - df_filtered['G1_Detection_Rate_pct']

    plt.figure(figsize=(5, 3))

    plt.plot(df_filtered['Iteration'], df_filtered['Difference_Rate_pct'],
             color='#d62728', linestyle='-', label='Difference: B Rate - A Rate', linewidth=1.5)

    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)

    plt.xlabel('Fuzzing Session')
    plt.ylabel('Detection Rate Difference (B - A, %)')
    # plt.title('Bug Detection Rate Difference Trend (B Rate - A Rate)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=1)

    if df_filtered['Iteration'].max() > 500:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='upper'))

    plt.tight_layout()
    plt.savefig(output_path, format=file_format)
    plt.close()
    logger.info(f"Saved detection rate difference graph to: {output_path}")
    
def analyze_g2_vs_g1_superiority(trend_df, output_dir):
    """
    Calculate count and ratio of G2 exceeding G1 detection rate and output to log.
    """
    logger.info("\n--- G2 vs G1 Superiority Analysis ---")
    
    df_filtered = trend_df[(trend_df['G1_Total_Projects'] >= 100) | (trend_df['G2_Total_Projects'] >= 100)].copy()

    if df_filtered.empty:
        logger.warning("[RESULT] No iterations met the >= 100 project count filter.")
        return

    g2_superior_count = len(df_filtered[df_filtered['G2_Detection_Rate_pct'] > df_filtered['G1_Detection_Rate_pct']])
    total_iterations_analyzed = len(df_filtered)
    superiority_rate_pct = (g2_superior_count / total_iterations_analyzed) * 100 if total_iterations_analyzed > 0 else 0
    
    logger.info(f"[FILTER] Analyzed Total Iterations (>=100 projects): {total_iterations_analyzed}")
    logger.info(f"[RESULT] G2 > G1 Iterations: {g2_superior_count}/{total_iterations_analyzed} ({superiority_rate_pct:.2f}%)")

    summary_data = {
        'Metric': ['G2_Superior_Count', 'Total_Analyzed_Iterations', 'G2_Superiority_Rate_pct'],
        'Value': [g2_superior_count, total_iterations_analyzed, superiority_rate_pct]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'rq4_g2_g1_superiority_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved G2 superiority summary to: {csv_path}")
    
    
def analyze_and_report_g4_delta(pre_rate, post_rate, n_total):
    """
    Compare G4 pre/post average detection rates and report improvement effect.
    """
    logger.info("\n--- Group C Corpus Introduction Effect Analysis ---")
    logger.info(f"Number of Projects: {n_total}")
    logger.info(f"Average Pre-Introduction Detection Rate:  {pre_rate:.2f}%")
    logger.info(f"Average Post-Introduction Detection Rate: {post_rate:.2f}%")
    
    delta = post_rate - pre_rate
    logger.info(f"Effect (Post - Pre): {delta:+.2f} points")
    
    if pre_rate > 0:
        improvement_ratio = (delta / pre_rate) * 100
        logger.info(f"Relative Improvement: {improvement_ratio:+.2f}%")
    else:
        logger.info(f"Relative Improvement: Undefined (Pre-rate is 0%)")


def main():
    """Main execution function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"--- Starting RQ4 Bug Detection Trend Analysis ---")
    logger.info(f"Graph save format: {FILE_FORMAT}")
    
    config = ConfigParser()
    if not os.path.exists(DB_CONFIG_FILE):
        logger.error(f"Error: DB config file '{DB_CONFIG_FILE}' not found.")
        sys.exit(1)
    config.read(DB_CONFIG_FILE)

    try:
        db_config = config["POSTGRES"]
        db = DB(database=db_config.get("POSTGRES_DB"), user=db_config.get("POSTGRES_USER"),
                password=db_config.get("POSTGRES_PASSWORD"), host=db_config.get("POSTGRES_IP"),
                port=db_config.get("POSTGRES_PORT"))
        db.connect()
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

    eligible_projects_set = get_eligible_projects_from_db(db)
    if not eligible_projects_set:
        logger.error("Could not retrieve eligible projects from DB. Aborting.")
        return

    project_groups, g4_time_df = categorize_projects_and_get_g4_time(eligible_projects_set, CORPUS_ANALYSIS_CSV)
    if project_groups is None:
        logger.error("Failed during project categorization. Aborting.")
        return

    g1_stats, g2_stats, g4_dynamic_data, missing_pre, g4_transition_data = analyze_rq4_detection_trends(db, project_groups, g4_time_df)
    
    # If INCLUDE_MISSING_PRE_IN_G2 is True, add Group 4 projects with missing Pre data to Group 2
    add_g2 = missing_pre if missing_pre and INCLUDE_MISSING_PRE_IN_G2 else set()
    
    if add_g2:
        logger.info(f"Re-running analysis with additional G2 projects: {len(add_g2)}")
        g1_stats, g2_stats, g4_dynamic_data, _, _ = analyze_rq4_detection_trends(db, project_groups, g4_time_df, add_g2)
    
    trend_df = calculate_and_save_stats(g1_stats, g2_stats, OUTPUT_DIR)
    print(f"Groups used: {get_group_name('group1')} ({len(project_groups['group1'])} projects), {get_group_name('group2')} ({len(project_groups['group2']) + len(add_g2)} projects)")
    
    # Calculate count and ratio of G2 > G1 within valid data range
    g2_superior_count = len(trend_df[trend_df['G2_Detection_Rate_pct'] > trend_df['G1_Detection_Rate_pct']])
    total_iterations = len(trend_df)
    superiority_rate_pct = (g2_superior_count / total_iterations) * 100 if total_iterations > 0 else 0
    print(f"Count of Group B exceeding Group A within valid data range: {g2_superior_count}/{total_iterations} ({superiority_rate_pct:.2f}%)")
    
    g1_rates = trend_df['G1_Detection_Rate_pct'].tolist()
    g2_rates = trend_df['G2_Detection_Rate_pct'].tolist()
    
    def find_first_below_5(rates):
        for idx, rate in enumerate(rates):
            if rate < 5:
                return idx
        return len(rates)
    
    first_below_g1 = find_first_below_5(g1_rates)
    first_below_g2 = find_first_below_5(g2_rates)
    
    # Display iteration that fell below 5% and its value
    if first_below_g1 < len(g1_rates):
        iter_g1 = trend_df.iloc[first_below_g1]['Iteration']
        rate_g1 = g1_rates[first_below_g1]
        print(f"Group A: {iter_g1}th iteration fell below 5% (value: {rate_g1:.2f}%)")
    else:
        print("Group A: No iteration fell below 5%")
    
    if first_below_g2 < len(g2_rates):
        iter_g2 = trend_df.iloc[first_below_g2]['Iteration']
        rate_g2 = g2_rates[first_below_g2]
        print(f"Group B: {iter_g2}th iteration fell below 5% (value: {rate_g2:.2f}%)")
    else:
        print("Group B: No iteration fell below 5%")
    
    rates_after_g1 = g1_rates[first_below_g1:]
    rates_after_g2 = g2_rates[first_below_g2:]
    
    if rates_after_g1:
        median_g1 = np.median(rates_after_g1)
        iqr_g1 = np.subtract(*np.percentile(rates_after_g1, [75, 25]))
        print(f"Group A: median {median_g1:.2f}, IQR {iqr_g1:.2f}")
        print(f"Group A: Last valid data count {trend_df.iloc[-1]['Iteration']}th")
    else:
        print("Group A: No data below 5%")
    
    if rates_after_g2:
        median_g2 = np.median(rates_after_g2)
        iqr_g2 = np.subtract(*np.percentile(rates_after_g2, [75, 25]))
        print(f"Group B: median {median_g2:.2f}, IQR {iqr_g2:.2f}")
        print(f"Group B: Last valid data count {trend_df.iloc[-1]['Iteration']}th")
    else:
        print("Group B: No data below 5%")
    
    # --- Identify and display/draw range where both groups satisfy >= 100 condition ---
    
    valid_condition = (trend_df['G1_Total_Projects'] >= 100) & (trend_df['G2_Total_Projects'] >= 100)
    valid_rows = trend_df[valid_condition]

    if not valid_rows.empty:
        max_valid_iteration = valid_rows['Iteration'].max()
    else:
        max_valid_iteration = 0
        
    print(f"\n[Graph Limit Info] Max iteration where both groups maintained >= 100 projects: {max_valid_iteration}")
    
    print("Data around end:")
    
    if max_valid_iteration > 0:
        row_last = trend_df[trend_df['Iteration'] == max_valid_iteration]
        if not row_last.empty:
            g1_total = int(row_last['G1_Total_Projects'].values[0])
            g2_total = int(row_last['G2_Total_Projects'].values[0])
            print(f"{max_valid_iteration}: Group A {g1_total}, Group B {g2_total}")
    
    next_iter = max_valid_iteration + 1
    
    g1_next_data = g1_stats.get(next_iter)
    g2_next_data = g2_stats.get(next_iter)

    if g1_next_data or g2_next_data:
        g1_next_total = g1_next_data[0] if g1_next_data else 0
        g2_next_total = g2_next_data[0] if g2_next_data else 0
        print(f"{next_iter}: Group A {g1_next_total}, Group B {g2_next_total} (Outside filter)")
    else:
        print(f"(No data exists after iteration {max_valid_iteration})")

    df_for_graph = trend_df[trend_df['Iteration'] <= max_valid_iteration].copy()

    create_detection_rate_trend_graph(df_for_graph, os.path.join(OUTPUT_DIR, f'rq4_g1_g2_detection_trend.{FILE_FORMAT}'), file_format=FILE_FORMAT)
    # create_detection_rate_difference_graph(df_for_graph, os.path.join(OUTPUT_DIR, f'rq4_g2_minus_g1_difference_trend.{FILE_FORMAT}'), file_format=FILE_FORMAT)  # (DISABLED)
    
    
    # G4 Analysis
    df_g4_iteration = analyze_g4_corpus_introduction_iteration(db, project_groups, g4_time_df)
    
    # Analyze with filtered data (g4_dynamic_data should have same project count for each step)
    # Since n doesn't change, n display in analyze_g4_trend will be nearly constant
    overall_pre_rate, overall_post_rate = analyze_g4_trend(g4_dynamic_data, OUTPUT_DIR, g4_transition_data)
    
    # Report denominator: Can use element count in Pre-1 (Should be same for all steps)
    n_analyzed = len(g4_dynamic_data[-1]) if g4_dynamic_data and -1 in g4_dynamic_data else 0
    analyze_and_report_g4_delta(overall_pre_rate, overall_post_rate, n_analyzed)
    
    report_g4_pre_post_transition(g4_transition_data, OUTPUT_DIR)

    print(f"Valid project count for Group C: {n_analyzed}")


    logger.info("\n--- RQ4 Bug Detection Trend Analysis Finished ---")

def report_g4_pre_post_transition(g4_transition_data, output_dir):
    """
    Aggregate and report G4 project detection state transitions (Pre detected/not -> Post detected/not).
    Also draw and save Venn diagram.
    """
    if not g4_transition_data:
        return

    c_i_iii = 0 # Pre Yes, Post Yes (Intersection)
    c_i_iv  = 0 # Pre Yes, Post No (Pre Only)
    c_ii_iii = 0 # Pre No, Post Yes (Post Only)
    c_ii_iv  = 0 # Pre No, Post No (Outside)
    
    for item in g4_transition_data:
        pre = item['pre']
        post = item['post']
        
        if pre and post:
            c_i_iii += 1
        elif pre and not post:
            c_i_iv += 1
        elif not pre and post:
            c_ii_iii += 1
        elif not pre and not post:
            c_ii_iv += 1
            
    total = len(g4_transition_data)
    
    print("\n=== Group C Pre/Post Detection Transition ===")
    print(f"Total Projects: {total}")
    print(f" (i)-(iii) Detected in Pre AND Detected in Post: {c_i_iii}")
    print(f" (i)-(iv)  Detected in Pre AND NOT Detected in Post: {c_i_iv}")
    print(f" (ii)-(iii) NOT Detected in Pre AND Detected in Post: {c_ii_iii}")
    print(f" (ii)-(iv)  NOT Detected in Pre AND NOT Detected in Post: {c_ii_iv}")
    print(f" Sum check: {c_i_iii + c_i_iv + c_ii_iii + c_ii_iv}")
    print("=============================================\n")

    # --- Venn Diagram ---
    if venn2 is None:
        logger.warning("Optional package 'matplotlib-venn' not found — skipping Venn diagram. Install with: pip install matplotlib-venn")
    else:
        try:
            plt.figure(figsize=(5, 4))
            # subset order for venn2 is (Ab, aB, AB)
            # Ab: Pre Yes, Post No = c_i_iv
            # aB: Pre No, Post Yes = c_ii_iii
            # AB: Pre Yes, Post Yes = c_i_iii
            
            v = venn2(subsets=(c_i_iv, c_ii_iii, c_i_iii), set_labels=('Detected in Pre', 'Detected in Post'))
            
            # Style adjustments
            if v.get_patch_by_id('10'): # Pre Only
                v.get_patch_by_id('10').set_alpha(0.5)
                v.get_patch_by_id('10').set_color('skyblue')
            if v.get_patch_by_id('01'): # Post Only
                v.get_patch_by_id('01').set_alpha(0.5)
                v.get_patch_by_id('01').set_color('lightgreen')
            if v.get_patch_by_id('11'): # Both
                v.get_patch_by_id('11').set_alpha(0.5)
                v.get_patch_by_id('11').set_color('violet')

            plt.title('Bug Detection Overlap (Group C)')
            
            # Note regarding 'Neither'
            plt.text(0, -0.65, f"Neither Detected: {c_ii_iv}\n(Total: {total})", ha='center', fontsize=9)
            
            # output_dir validation
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            save_path = os.path.join(output_dir, "rq4_gc_bug_detection_venn.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Venn diagram to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to create Venn diagram: {e}")

if __name__ == "__main__":
    main()