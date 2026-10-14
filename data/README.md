# Benchmark Dataset Documentation

This directory contains the curated multi-domain requirements engineering corpus derived from the PROMISE Software Engineering Repository, PROMISE-NFR, and FNFC datasets.

## Dataset Structure

File: `promise_fnfc_benchmark.csv`

| Column | Type | Description |
|---|---|---|
| `requirement` | String | Natural language software requirement statement |
| `category` | String | Ground-truth taxonomy label |

## Target Classes & Distribution

The benchmark spans 5 canonical categories:
1. **Functional (FR)**: 48 requirements (Business logic, authentication, data processing, user actions)
2. **Security (SE)**: 48 requirements (Access control, cryptographic protocols, token validation)
3. **Performance (PE)**: 48 requirements (Throughput, SLA response latency, concurrency scalability)
4. **Usability (US)**: 48 requirements (Accessibility WCAG compliance, user experience, feedback)
5. **Maintainability (MN)**: 48 requirements (Cyclomatic complexity, modularity, OpenAPI contracts)

Total records: **240 requirements**.
Strict zero-leakage 10-fold stratified cross-validation is used across all benchmark experiments.
