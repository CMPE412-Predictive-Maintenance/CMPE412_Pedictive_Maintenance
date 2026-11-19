[README.md](https://github.com/user-attachments/files/35085887/README.md)

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
| 04 - Modeling | Trains machine learning models for RUL prediction and anomaly detection. |
| 05 - Evaluation | Evaluates model performance and compares prediction results. |
| 06 - Deployment | Demonstrates how the predictions could support maintenance decisions. |

## Dataset

**NASA C-MAPSS FD001**

The dataset contains simulated turbofan engine data, including engine identifiers, operating cycles, operating settings, and sensor measurements.

The data will be used for exploratory analysis, preprocessing, model training, and evaluation.

## Machine Learning Approach

For RUL prediction, we plan to compare a Linear Regression baseline with other regression models, including Random Forest and Gradient Boosting.

Anomaly detection will be explored separately using Isolation Forest to identify unusual sensor patterns.

Model performance will be evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Project Status

This project is currently under development as part of the CMPE 412 Artificial Intelligence course.

## Team Collaboration

The team uses GitHub Issues, a Kanban board, branches, and Pull Requests to organize tasks, track progress, and review contributions.
