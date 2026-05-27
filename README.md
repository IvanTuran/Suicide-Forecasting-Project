# Suicide Forecasting Project

A hybrid machine learning framework for population-level suicide risk forecasting using behavioral distress signals from social media and structural socioeconomic indicators across the United States.

## Project Overview

Most traditional suicide surveillance systems rely on retrospective mortality datasets and static socioeconomic indicators. While useful for long-term public health analysis, these systems are often delayed and may fail to capture rapidly evolving psychological distress occurring across communities in near real time.

This project explores whether behavioral distress patterns extracted from online language can improve population-level suicide risk forecasting when combined with structural socioeconomic data.

Using approximately 12,500 geolocated suicide-related Reddit posts alongside state-level socioeconomic indicators, this framework integrates:

- Natural Language Processing (NLP)
- Machine Learning
- Explainable AI (SHAP)
- Geospatial Analysis
- Public Health Forecasting

The project focuses on regional forecasting rather than individual prediction. The goal is to examine whether aggregated behavioral patterns may provide supplementary insight into broader public health vulnerability.

---

## Framework Pipeline

1. Suicide-related Reddit posts were collected and geolocated across all 50 U.S. states.

2. NLP techniques were used to extract linguistic distress features associated with:
   - hopelessness
   - suicidal ideation
   - emotional intensity
   - negative affect
   - distress-related language patterns

3. Linguistic distress scores were aggregated at the state level.

4. Behavioral distress metrics were integrated with socioeconomic indicators including:
   - unemployment rate
   - median household income
   - bachelor's degree attainment
   - income inequality (Gini Index)

5. A Gradient Boosting machine learning framework was used to forecast relative state-level suicide vulnerability.

---

## Key Results

The hybrid forecasting framework demonstrated stronger explanatory performance than both linguistic-only and socioeconomic-only approaches, highlighting the importance of integrating behavioral and structural indicators within population-level suicide forecasting systems.

Rather than treating socioeconomic conditions or online behavioral distress as isolated predictors, this project explored how combining these domains may provide a more complete representation of regional psychological vulnerability. Structural socioeconomic indicators capture long-term contextual vulnerability, while linguistic distress signals derived from social media provide more dynamic insight into rapidly evolving emotional and behavioral conditions occurring across populations.

The results suggest that hybrid behavioral-socioeconomic forecasting frameworks may represent a promising direction for future computational public health surveillance systems.

The linguistic distress classifier achieved:

- Accuracy: 0.90
- Precision: 0.90
- Recall: 0.90
- ROC-AUC: 0.97

Additional analyses included:
- Pearson correlation analysis
- SHAP explainability analysis
- Moran’s I spatial autocorrelation analysis
- state-level geospatial visualization using QGIS

SHAP analysis further demonstrated that behavioral distress features contributed substantially within the hybrid framework, while geospatial analyses revealed meaningful regional variation between structural socioeconomic conditions and behavioral distress patterns across states.

---

## Interactive Dashboard

This repository also includes an interactive dashboard for:
- socioeconomic monitoring
- state-level distress visualization
- hybrid risk forecasting
- comparative geographic analysis

The dashboard was developed to translate machine learning outputs into a more interpretable and actionable public health framework. In addition to visualizing predicted regional vulnerability, the system integrates behavioral distress metrics with socioeconomic indicators to support comparative state-level analysis and intervention-oriented interpretation.

The dashboard also serves as a demonstration of how hybrid forecasting systems could eventually support more responsive and data-informed public health monitoring infrastructure.

---

## Repository Structure

```bash
data/             #Processed datasets
linguistic_model/ # Scripts for the Linguistic Model
hybrid_model/     # Scripts for the Hybrid Model
dashboard/        # Interactive dashboard files
figures/          # Figures used in the manuscript
