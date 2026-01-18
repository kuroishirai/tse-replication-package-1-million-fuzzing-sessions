# 📦 Replication Package: Large-Scale Empirical Analysis of Continuous Fuzzing

This repository is the replication package for the paper **"Large-Scale Empirical Analysis of Continuous Fuzzing: Insights from 1 Million Fuzzing Sessions"**.

It contains all the necessary scripts and data schemas to replicate the analysis presented in the paper, including data collection, storage, and statistical analysis using Docker and PostgreSQL.

## 📁 Directory Structure

The repository is organized as follows:

```
.
├── data/
│   ├── processed_data/csv/       # Contains processed CSV files used for analysis (e.g., issues.csv, coverage_data.csv)
│   ├── database/backup_clean.sql # SQL dump file for restoring the PostgreSQL database
│   └── result_data/              # Output directory for analysis results (tables, figures)
├── program/
│   ├── preparation/              # Scripts used for data collection and preprocessing
│   ├── research_questions/       # Python scripts for answering RQ1-RQ4 (e.g., detection rate, coverage trends)
│   └── envFile.ini               # Configuration file
├── requirements.txt              # Python dependencies list
├── Dockerfile                    # Docker build configuration
├── docker-compose.yml            # Docker Compose configuration
├── run_all_analysis.sh           # Helper script to execute all analysis steps
└── README.md
```

## 🐳 Docker Setup

### Reset Docker (Optional)

If you need to start fresh or clean up existing volumes:

```bash
docker compose down --volumes
docker volume ls
docker volume rm fuzzingeffectiveness_pgdata
docker system prune -a --volumes -f
```

### Build and Launch

Build the Docker containers and start the services:

```bash
docker compose build --no-cache
docker compose up -d
```

## 🗃️ Restore Database

Restore the PostgreSQL database from the provided SQL dump:

```bash
docker compose exec -T db psql -U replication_user -d replication_db < data/database/backup_clean.sql
```

## ✅ Run Analysis Programs

You can run the analysis scripts using the Docker container.

To run all analysis scripts sequentially:

```bash
docker compose run --rm research bash run_all_analysis.sh
```

To run a specific analysis script (e.g., RQ1):

```bash
docker compose run --rm research python program/research_questions/rq1_detection_rate.py
```