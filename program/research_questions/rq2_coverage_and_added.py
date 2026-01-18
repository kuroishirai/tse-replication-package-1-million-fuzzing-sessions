import sys
import os
import csv
import numpy as np
import pandas as pd
from configparser import ConfigParser
from tqdm import tqdm
from datetime import datetime, timedelta
import statistics

# --- Configuration ---
MODULE_PATH = 'program/__module'
DB_CONFIG_FILE = 'program/envFile.ini'
OUTPUT_DIR = 'data/result_data/rq3'
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'change_analysis')

# --- DB Query Definitions ---

# 1. Query to get project list from total_coverage (RQ2 logic)
QUERY_PROJECTS = """
    SELECT project
    FROM total_coverage
    WHERE coverage IS NOT NULL AND coverage > 0 AND date < '2025-01-08'
    GROUP BY project
    HAVING COUNT(*) >= 365
    ORDER BY project
"""

# 2. Query to get coverage data from total_coverage
def GET_COVERAGE_DATA(project):
    """
    Get coverage data for the specified project.
    Assumes date column is date type (YYYY-MM-DD).
    """
    return f"""
        SELECT 
            date, 
            covered_line, 
            total_line
        FROM 
            total_coverage
        WHERE 
            project = '{project}'
            AND date < '2025-01-08'
        ORDER BY 
            date ASC
    """

# 3. Query to get successful coverage build logs from buildlog_data
def GET_BUILD_LOGS(project):
    """
    Get successful coverage build logs (modules, revisions, timecreated).
    Assumes timecreated is timestamp type (datetime).
    """
    return f"""
        SELECT 
            timecreated, 
            modules, 
            revisions
        FROM 
            buildlog_data
        WHERE 
            project = '{project}'
            AND build_type = 'Coverage' 
            AND result IN ('HalfWay', 'Finish')
            AND timecreated < '2025-01-08'
        ORDER BY 
            timecreated ASC
    """

# --- Main Analysis Function ---

def analyze_coverage_change(db):
    """
    For each project, detect change points in build info (modules/revisions),
    join with total_coverage to analyze coverage changes, and save results as CSV.

    Args:
        db (DB): Database connection object.
    """
    print("--- RQ3 Coverage Change Analysis Started ---")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    # 1. Get list of projects for analysis
    project_records = db.executeQuery("select", QUERY_PROJECTS)
    projects = [project[0] for project in project_records]
    
    if not projects:
        print("Warning: No projects found satisfying the criteria (coverage >= 365 sessions). Exiting.")
        return

    print(f"\n--- Starting to process {len(projects)} projects ---")
    
    # List to aggregate results from all projects
    all_results = []
    header = [
        'project', 'timecreated_i', 'modules_i', 'revisions_i', 
        'timecreated_i+1', 'modules_i+1', 'revisions_i+1', 
        'covered_line_i', 'total_line_i', 
        'covered_line_i+1', 'total_line_i+1', 
        'diff_total_line', 'diff_coverage'
    ]
    
    for project_name in tqdm(projects, desc="Processing projects"):
        
        # 1. Get successful build logs (timecreated, modules, revisions)
        raw_build_logs = db.executeQuery("select", GET_BUILD_LOGS(project_name))
        if not raw_build_logs:
            continue
            
        # 2. Get coverage data (date, covered_line, total_line)
        raw_coverage_data = db.executeQuery("select", GET_COVERAGE_DATA(project_name))
        if not raw_coverage_data:
            continue

        # Convert to DataFrame and prepare date as key
        # Coverage Data: date (datetime.date), covered_line (float), total_line (float)
        cov_df = pd.DataFrame(raw_coverage_data, columns=['date', 'covered_line', 'total_line'])
        cov_df['date'] = pd.to_datetime(cov_df['date']).dt.date  # Convert date to date object

        # Build Logs: timecreated (timestamp), modules, revisions
        build_df = pd.DataFrame(raw_build_logs, columns=['timecreated', 'modules', 'revisions'])
        # Convert timecreated to date object, create 'build_date' as join key
        build_df['build_date'] = pd.to_datetime(build_df['timecreated']).dt.date

        # 3. Filter build logs (keep only first and last of consecutive same combinations)
        
        # Identify groups with consecutive same modules/revisions combinations
        build_df['group_key'] = build_df['modules'].astype(str) + '_' + build_df['revisions'].astype(str)
        # Detect change points (shift)
        build_df['group_id'] = (build_df['group_key'] != build_df['group_key'].shift(1)).cumsum()
        
        # Get first (min timecreated) and last (max timecreated) rows for each group
        # Get the row with earliest timecreated (first log) and latest timecreated (last log)
        def get_start_end_logs(group):
            start_log = group.iloc[0]  # Log with earliest timecreated
            end_log = group.iloc[-1]   # Log with latest timecreated

            # Extract timecreated, modules, revisions from start and end logs for change point analysis
            return pd.Series({
                'timecreated_start': start_log['timecreated'],
                'modules': start_log['modules'],
                'revisions': start_log['revisions'],
                'timecreated_end': end_log['timecreated'],
                'build_date_end': end_log['build_date']
            })

        # Process by group
        filtered_logs_df = build_df.groupby('group_id').apply(get_start_end_logs, include_groups=False).reset_index()
        project_change_points = []
        
        for i in range(len(filtered_logs_df) - 1):
            log_i = filtered_logs_df.iloc[i]
            log_i_plus_1 = filtered_logs_df.iloc[i+1]
            
            # log_i (before change) uses end time of group i
            timecreated_i = log_i['timecreated_end']
            modules_i = log_i['modules']
            revisions_i = log_i['revisions']
            date_i = log_i['build_date_end']

            # log_i_plus_1 (after change) uses start time of group i+1
            timecreated_i_plus_1 = log_i_plus_1['timecreated_start']
            modules_i_plus_1 = log_i_plus_1['modules']
            revisions_i_plus_1 = log_i_plus_1['revisions']
            date_i_plus_1 = pd.to_datetime(timecreated_i_plus_1).date()

            # Get coverage data (join by date match)
            # Coverage data before change (log_i)
            cov_i_match = cov_df[cov_df['date'] == date_i]
            if not cov_i_match.empty:
                # If multiple matching data exist, use first record for convenience
                covered_line_i = cov_i_match['covered_line'].iloc[0]
                total_line_i = cov_i_match['total_line'].iloc[0]
            else:
                covered_line_i, total_line_i = np.nan, np.nan
            
            # Coverage data after change (log_i+1)
            cov_i_plus_1_match = cov_df[cov_df['date'] == date_i_plus_1]
            if not cov_i_plus_1_match.empty:
                covered_line_i_plus_1 = cov_i_plus_1_match['covered_line'].iloc[0]
                total_line_i_plus_1 = cov_i_plus_1_match['total_line'].iloc[0]
            else:
                covered_line_i_plus_1, total_line_i_plus_1 = np.nan, np.nan
            
            # 5. Calculate total_line change and coverage rate change
            
            # NaN check
            valid_i = not pd.isna(total_line_i) and total_line_i != 0
            valid_i_plus_1 = not pd.isna(total_line_i_plus_1) and total_line_i_plus_1 != 0
            
            coverage_i = (covered_line_i / total_line_i) * 100 if valid_i else np.nan
            coverage_i_plus_1 = (covered_line_i_plus_1 / total_line_i_plus_1) * 100 if valid_i_plus_1 else np.nan

            if valid_i and valid_i_plus_1:
                diff_total_line = total_line_i_plus_1 - total_line_i
                diff_coverage = coverage_i_plus_1 - coverage_i
            else:
                diff_total_line = np.nan
                diff_coverage = np.nan
                
            # 6. Save result as one row
            row = [
                project_name,
                timecreated_i,  # timecreated_i (last build time just before change)
                modules_i,
                revisions_i,
                timecreated_i_plus_1,  # timecreated_i+1 (first build time after change)
                modules_i_plus_1,
                revisions_i_plus_1,
                covered_line_i,
                total_line_i,
                covered_line_i_plus_1,
                total_line_i_plus_1,
                diff_total_line,
                diff_coverage
            ]
            project_change_points.append(row)
            all_results.append(row)  # Add to overall results

        # Save to CSV file for each project
        if project_change_points:
            output_csv_path = os.path.join(CSV_OUTPUT_DIR, f"{project_name}.csv")
            with open(output_csv_path, "w", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(project_change_points)
    
    print("\n--- Project processing finished ---\n")
    
    # Save combined CSV for all projects
    if all_results:
        all_csv_path = os.path.join(OUTPUT_DIR, "all_coverage_change_analysis.csv")
        with open(all_csv_path, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results)
        print(f"All project change analysis saved to: {all_csv_path}")


def main():
    """
    Main function: Execute DB connection and analysis processing
    """
    print("--- Main process started for RQ3 ---")

    # Add the module path to sys.path for custom module imports
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)
    
    try:
        from dbFile import DB
    except ImportError as e:
        print(f"Error: Custom module import failed. Please check MODULE_PATH and modules. Error: {e}")
        sys.exit(1)


    # --- 1. Database Connection ---
    config = ConfigParser()
    config.read(DB_CONFIG_FILE)
    
    if not config.has_section("POSTGRES"):
        print(f"Error: DB configuration file {DB_CONFIG_FILE} does not contain [POSTGRES] section.")
        sys.exit(1)
        
    db_config = config["POSTGRES"]

    db = DB(database=db_config["POSTGRES_DB"], user=db_config["POSTGRES_USER"],
            password=db_config["POSTGRES_PASSWORD"], host=db_config["POSTGRES_IP"],
            port=db_config["POSTGRES_PORT"])
    
    try:
        db.connect()
    except Exception as e:
        print(f"Error: Database connection failed. Details: {e}")
        sys.exit(1)
    
    # --- 2. Analysis Execution ---
    analyze_coverage_change(db)

    # --- 3. Close DB Connection ---
    
    print("\n--- Main process finished for RQ3 ---")

if __name__ == '__main__':
    main()