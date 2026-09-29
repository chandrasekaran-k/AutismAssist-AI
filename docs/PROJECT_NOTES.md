# Project Notes

## Modeling choices

- Random Forest was selected as the main baseline because it handles mixed nonlinear relationships well and provides feature importance.
- A MultiOutputClassifier wrapper is used for the five binary support-area targets.
- The notebook folder contains the reproducible workflow.

## Suggested next improvements

1. Add cross-validation and confidence intervals.
2. Compare XGBoost/LightGBM where appropriate.
3. Calibrate screening probabilities.
4. Add SHAP or permutation explanations.
5. Test fairness across demographic groups.
6. Validate against an independently collected, clinically reviewed dataset.
7. Add model versioning and experiment tracking.
