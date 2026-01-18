import sys
import os
import subprocess
import csv
import datetime
from tqdm import tqdm
import shutil
import pandas as pd
from configparser import ConfigParser
import logging
import requests 

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd() 
OSS_FUZZ_ROOT = os.path.join(CWD, 'data/collect_data/repos/oss-fuzz')

MODULE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../program/__module'))
DB_CONFIG_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../../program/envFile.ini'))

PROJECTS_DIR = os.path.join(OSS_FUZZ_ROOT, 'projects')
OUTPUT_CSV = os.path.join(CWD, 'data/processed_data/csv/project_corpus_analysis.csv')
TARGET_STRING = '_seed_corpus.zip'

# ★modification: GITHUB_TOKENはグローバル変数として保持し、mainでセット
GITHUB_OWNER = 'google'
GITHUB_REPO = 'oss-fuzz'
GITHUB_TOKEN = None 

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Helper Functions for Time Conversion ---

def parse_iso_time(time_str):
    """Convert ISO 8601 format string to datetime object"""
    if not time_str:
        return None
    
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    try:
        return datetime.datetime.fromisoformat(time_str)
    except ValueError as e:
        logger.error(f"Error parsing date string: {time_str} | Error: {e}")
        return None

# --- Git Analysis Functions (omitted) ---

def run_git_command(command, repo_path):
    """
    Specified repository path(repo_path)Execute Git command with this as CWD
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,
            cwd=repo_path
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None
    except FileNotFoundError:
        logger.error("Error: 'git' command not found. Make sure Git is installed.")
        sys.exit(1)

def get_project_creation_time(git_command_args, repo_path):
    """
    Execute Git command and get first commit datetime（SHA not required）
    """
    output = run_git_command(git_command_args, repo_path)
    if output:
        first_line = output.split('\n')[0].strip()
        return parse_iso_time(first_line)
    return None

def get_corpus_commit_details(git_command_args, repo_path):
    """
    Execute Git command and get corpus commit SHA and datetime
    """
    output = run_git_command(git_command_args, repo_path)
    if output:
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if len(lines) >= 2:
            sha = lines[0]
            time_str = lines[1]
            commit_time = parse_iso_time(time_str)
            return (sha, commit_time)
    return (None, None)

# --- GitHub API Function (Merge Time) ---

def get_merge_time_from_github_api(commit_sha):
    """
    GitHub APIを使って、指定されたコミットのマージ時間を取得する
    """
    global GITHUB_TOKEN
    
    # Skip API call if token is not set
    if not GITHUB_TOKEN:
        logger.warning("GitHub Token is not set. Skipping API call for merge time.")
        return None
    
    pulls_url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{commit_sha}/pulls'
    
    headers = {'Accept': 'application/vnd.github.v3+json'}
    headers['Authorization'] = f'token {GITHUB_TOKEN}' # ★Assuming token is set
    
    try:
        # Step 1: List Pull Requests containing the commit
        response = requests.get(pulls_url, headers=headers, params={'state': 'closed', 'per_page': 1})
        response.raise_for_status()
        
        pulls_data = response.json()
        
        if not pulls_data:
            return None 
            
        pr_number = pulls_data[0]['number']
        
        # Step 2: Get PR details and check merge time
        pr_url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/pulls/{pr_number}'
        pr_response = requests.get(pr_url, headers=headers)
        pr_response.raise_for_status()
        
        pr_data = pr_response.json()
        
        merge_time_str = pr_data.get('merged_at')
        
        if merge_time_str:
            return parse_iso_time(merge_time_str)
        
        return None
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"API Error (404 Not Found) for commit {commit_sha}: Check repo access/existence.")
        elif e.response.status_code == 403:
            logger.error(f"API Error (403 Forbidden/Rate Limit Exceeded). Check GITHUB_TOKEN validity or rate limits.")
        else:
            logger.error(f"HTTP Error during API call for {commit_sha}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error during API call for {commit_sha}: {e}")
        return None


def analyze_repository(project_names, repo_path):
    """
    Analyze repository and return results as list
    """
    logger.info(f"Starting Git and GitHub API analysis of {len(project_names)} projects...")
    results = []
    
    for project_name in tqdm(project_names, desc="Analyzing projects"):
        
        project_relative_path = os.path.join('projects', project_name)
        build_sh_relative_path = os.path.join(project_relative_path, 'build.sh')
        build_sh_abs_path = os.path.join(PROJECTS_DIR, project_name, 'build.sh')
        
        project_creation_time = None
        corpus_commit_time = None
        corpus_merged_time = None 
        is_corpus = False
        time_elapsed = None 
        merged_time_elapsed = None 

        # Get project creation datetime
        cmd_create = ['git', 'log', '--reverse', '--diff-filter=A', '--pretty=format:%cI', '--', project_relative_path]
        project_creation_time = get_project_creation_time(cmd_create, repo_path)
        
        if not project_creation_time:
            continue
            
        if not os.path.exists(build_sh_abs_path):
            results.append([project_name, is_corpus, None, None, project_creation_time, time_elapsed, merged_time_elapsed])
            continue
            
        # Get corpus commit SHA and datetime
        cmd_corpus = ['git', 'log', '--reverse', f'-S{TARGET_STRING}', '--pretty=format:%H%n%cI', '--', build_sh_relative_path]
        commit_sha, corpus_commit_time = get_corpus_commit_details(cmd_corpus, repo_path)
        
        if corpus_commit_time:
            is_corpus = True
            
            # --- Get merge time via API ---
            if commit_sha:
                # If GITHUB_TOKEN is not set, skipped within this function
                corpus_merged_time = get_merge_time_from_github_api(commit_sha)
            
            # --- Calculate duration ---
            delta_commit = corpus_commit_time - project_creation_time
            time_elapsed = delta_commit.total_seconds() 
            
            if corpus_merged_time:
                delta_merge = corpus_merged_time - project_creation_time
                merged_time_elapsed = delta_merge.total_seconds()
            
        results.append([
            project_name, 
            is_corpus, 
            corpus_commit_time, 
            corpus_merged_time, 
            project_creation_time, 
            time_elapsed,
            merged_time_elapsed
        ])
    return results

def save_results_to_csv(results):
    """
    Save analysis results to CSV (omitted)
    """
    logger.info(f"\nAnalysis complete. Saving results to '{OUTPUT_CSV}'...")
    try:
        header = [
            'project_name', 
            'is_Corpus', 
            'corpus_commit_time', 
            'corpus_merged_time', 
            'project_creation_time', 
            'time_elapsed_seconds',
            'merged_time_elapsed_seconds'
        ]
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
        logger.info(f"Successfully saved {len(results)} records to '{OUTPUT_CSV}'.")
    except IOError as e:
        logger.error(f"Error writing to CSV file: {e}")

# get_projects_from_db と categorize_projects は前述のコードと同一のためomitted

def get_projects_from_db(db):
    # (omitted: Database connection and project list retrieval logic)
    # ...
    query = """
        SELECT project
        FROM total_coverage
        WHERE coverage IS NOT NULL AND coverage > 0 AND date < '2025-01-08'
        GROUP BY project
        HAVING COUNT(*) >= 365
    """
    # (実際のDB処理はomitted)
    try:
        project_records = db.executeQuery("select", query)
        projects_set = {project[0] for project in project_records}
        logger.info(f"Found {len(projects_set)} target projects in database.")
        return projects_set
    except Exception:
        return None

def categorize_projects(db_projects_set):
    # (omitted: CSV reading and category classification logic)
    # ...
    try:
        df = pd.read_csv(OUTPUT_CSV)
        df['corpus_commit_time'] = pd.to_datetime(df['corpus_commit_time'], errors='coerce')
        df['corpus_merged_time'] = pd.to_datetime(df['corpus_merged_time'], errors='coerce') 
        df['project_creation_time'] = pd.to_datetime(df['project_creation_time'], errors='coerce')
        
        analysis_column = 'merged_time_elapsed_seconds' 
        
    except FileNotFoundError:
        logger.error(f"Error: CSV file '{OUTPUT_CSV}' not found. Cannot perform categorization.")
        return
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return
        
    filtered_df = df[df['project_name'].isin(db_projects_set)].copy()
    
    if filtered_df.empty:
        logger.warning("No matching projects found between CSV and Database.")
        return
        
    def classify_time(seconds):
        if pd.isna(seconds):
            return 'N/A (No Merge Time)' 
        elif seconds < 86400:
            return 'Under 1 Day'
        elif 86400 <= seconds < 604800:
            return '1-7 Days'
        else: 
            return '7+ Days'

    filtered_df['time_category'] = filtered_df[analysis_column].apply(classify_time)

    logger.info("\n--- 4. Project Categorization Results (Based on Merge Time) ---")
    
    total_count = len(filtered_df)
    category_counts = filtered_df['time_category'].value_counts()
    category_percentages = (category_counts / total_count) * 100
    
    results_df = pd.DataFrame({
        'Category': category_percentages.index,
        'Percentage (%)': category_percentages.values,
        'Count': category_counts.values
    })
    
    category_order = ['Under 1 Day', '1-7 Days', '7+ Days', 'N/A (No Merge Time)']
    ordered_categories = [c for c in category_order if c in results_df['Category'].values]
    
    results_df['Category'] = pd.Categorical(results_df['Category'], categories=ordered_categories, ordered=True)
    results_df = results_df.sort_values('Category')
        
    print(results_df.to_string(index=False, float_format="%.2f"))


def main():
    """
    Main script logic
    """
    global GITHUB_TOKEN
    logger.info("--- Project Corpus Analysis Started ---")

    # --- 0. Resolve module path ---
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)

    try:
        from dbFile import DB
    except ImportError as e:
        logger.error(f"Error: Could not import 'dbFile'. ImportError details: {e}")
        sys.exit(1)

    # --- 1. GitHub API Token Setup ---
    logger.info("\n--- 1. GitHub API Token Setup ---")
    
    # Try getting from environment variable first
    token = os.environ.get('GITHUB_TOKEN')
    
    if not token:
        print("Please enter your GitHub Personal Access Token (PAT, needs 'repo' scope).")
        print("You can also set the GITHUB_TOKEN environment variable to skip this step.")
        token = input("Token: ").strip()

    if not token:
        logger.error("\nERROR: GitHub Token cannot be empty. Aborting analysis.")
        exit(1)

    GITHUB_TOKEN = token
    logger.info("GitHub Token successfully set.")


    # --- 2. Check CSV existence and execute Git analysis (omitted) ---
    git_check_path = os.path.join(OSS_FUZZ_ROOT, '.git')
    
    if not os.path.exists(git_check_path):
        logger.error(f"Error: '.git' not found at expected location: '{git_check_path}'")
        return
        
    if not os.path.isdir(PROJECTS_DIR):
        logger.error(f"Error: 'projects' directory not found at: '{PROJECTS_DIR}'")
        return

    if os.path.exists(OUTPUT_CSV):
        logger.info(f"'{OUTPUT_CSV}' already exists. Skipping Git analysis.")
    else:
        logger.warning(f"'{OUTPUT_CSV}' not found. Starting Git/API repository analysis...")
        
        try:
            project_names = sorted([
                name for name in os.listdir(PROJECTS_DIR) 
                if os.path.isdir(os.path.join(PROJECTS_DIR, name))
            ])
        except FileNotFoundError:
            logger.error(f"Error: '{PROJECTS_DIR}' directory not found.")
            return

        results = analyze_repository(project_names, OSS_FUZZ_ROOT)
        
        if results:
            save_results_to_csv(results)
        else:
            logger.error("No results generated from Git/API analysis. Aborting.")
            return 

    # --- 3. DB接続 ---
    logger.info("\n--- 2. Database Connection ---")
    config = ConfigParser()
    if not os.path.exists(DB_CONFIG_FILE):
         logger.error(f"Error: Config file '{DB_CONFIG_FILE}' not found.")
         sys.exit(1)
    config.read(DB_CONFIG_FILE)
    
    try:
        db_config = config["POSTGRES"]
        # DB connection logic (as before)
        # ...
        db = DB(database=db_config.get("POSTGRES_DB"), 
                user=db_config.get("POSTGRES_USER"),
                password=db_config.get("POSTGRES_PASSWORD"), 
                host=db_config.get("POSTGRES_IP"),
                port=db_config.get("POSTGRES_PORT"))
        db.connect()
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)


    # --- 4. DB integration and category classification ---
    
    db_projects = get_projects_from_db(db)
    
    if db_projects:
        categorize_projects(db_projects)
    else:
        logger.error("Could not retrieve project list from DB. Skipping categorization.")
        
    logger.info("\n--- Project Corpus Analysis Finished ---")

if __name__ == '__main__':
    main()