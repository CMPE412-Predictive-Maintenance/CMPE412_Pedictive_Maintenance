

# Predictive Maintenance for Aircraft Engines

## Project Overview

This project focuses on predictive maintenance for aircraft engines using machine learning techniques. Our goal is to analyze engine sensor data, estimate the Remaining Useful Life (RUL) of engines, and identify unusual sensor behavior.

The project aims to demonstrate how data-driven predictions can support maintenance planning and help prioritize engine inspections.

We use the NASA C-MAPSS FD001 dataset, which contains simulated aircraft engine operating data. The project follows the CRISP-DM methodology to organize the data analysis and machine learning workflow.

## Project Objectives

- Predict engine Remaining Useful Life (RUL) in operating cycles.
- Detect unusual sensor patterns using anomaly detection.
- Compare machine learning models using MAE and RMSE.
- Support inspection prioritization through a simulated maintenance scenario.

## Project Structure

The project is organized into the following CRISP-DM stages:

| Stage | Description |
|---|---|
| [01 - Business Understanding](01_business_understanding/README.md) | Defines the maintenance problem, project objectives, and success criteria. |
| [02 - Data Understanding](02_data_understanding/) | Explores the dataset, checks data quality, and visualizes engine sensor behavior. |
| [03 - Data Preparation](03_data_preparation/) | Prepares the data, calculates RUL, and creates model-ready datasets. |
| [04 - Modeling](04_modeling/README.md) | Compares and selects RUL prediction models using validation data and develops an Isolation Forest anomaly detector. |
| [05 - Evaluation](05_evaluation/README.md) | Evaluates the saved models on the official FD001 test set, analyzes prediction errors and anomaly warnings, and assesses deployment readiness. |
| 06 - Deployment | Demonstrates how the predictions could support maintenance decisions. |


## Dataset

**NASA C-MAPSS FD001**

The dataset contains simulated turbofan engine data, including engine identifiers, operating cycles, operating settings, and sensor measurements.

The dataset has been used for exploratory analysis, preprocessing, model training, and evaluation.

## Machine Learning Approach

For Remaining Useful Life (RUL) prediction, the project compared a Linear Regression baseline with Random Forest, HistGradientBoosting, and XGBoost models.

XGBoost was selected during the Modeling stage based on validation MAE. On the official FD001 test set, the selected model achieved an MAE of 10.974 cycles and an RMSE of 15.258 cycles, compared with the Linear Regression baseline MAE of 25.603 cycles and RMSE of 31.969 cycles.

Isolation Forest was developed during Modeling to identify unusual sensor behavior. The Evaluation stage examined anomaly flags, sustained warnings, critical RUL prediction errors, and inspection prioritization.

The evaluation results support a controlled, simulation-based maintenance decision-support demonstration, while highlighting limitations related to late predictions and simulated operating conditions.

## Project Status

This project is currently under development as part of the CMPE 412 Artificial Intelligence course.

## Team Collaboration

The team uses GitHub Issues, a Kanban board, branches, and Pull Requests to organize tasks, track progress, and review contributions.
