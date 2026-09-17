from __future__ import annotations

from pathlib import Path
import argparse
import os
import traceback

import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def code(source):
    return nbf.v4.new_code_cell(source)


def markdown(source):
    return nbf.v4.new_markdown_cell(source)


SETUP = """from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
ART = ROOT / 'artifacts'
FIG = ROOT / 'reports' / 'figures'
pd.set_option('display.max_columns', 100)
"""


def notebook(title, intro, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Fraud Project", "language": "python", "name": "fraud-project"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [markdown(f"# {title}\n\n{intro}"), code(SETUP), *cells]
    return nb


def build_all():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_eda.ipynb": notebook(
            "01 - Data Quality and Exploratory Analysis",
            "Canonical schema validation, duplicate policy, rare-event prevalence and temporal profiles.",
            [
                code("audit = json.loads((ART / 'data_audit.json').read_text())\npd.Series(audit, name='value')"),
                code("splits = pd.read_csv(ART / 'temporal_split_summary.csv')\nsplits"),
                code("display(Image(filename=str(FIG / '01_class_imbalance.png')))"),
                code("display(Image(filename=str(FIG / '02_amount_and_time.png')))"),
                markdown("The raw data contain an extreme rare event. Exact duplicate rows are removed before every temporal boundary; the final holdout remains chronologically later than model, calibration and policy blocks."),
            ],
        ),
        "02_classical_and_imbalanced_models.ipynb": notebook(
            "02 - Classical and Imbalance-Aware Models",
            "Temporal CV compares weighting, leakage-safe SMOTE, boosting and balanced ensembles. The winner is selected on a later validation block.",
            [
                code("cv = pd.read_csv(ART / 'temporal_cv_results.csv')\ncv.sort_values('cv_pr_auc_mean', ascending=False).head(15)"),
                code("validation = pd.read_csv(ART / 'model_validation_comparison.csv')\nvalidation.sort_values('validation_pr_auc', ascending=False)"),
                code("display(Image(filename=str(FIG / '03_model_validation.png')))"),
                code("json.loads((ART / 'selected_model.json').read_text())"),
                markdown("The strategy comparison is connected to model selection: the best hyperparameter configuration within each family/strategy proceeds to validation, and the validation winner alone feeds calibration and the decision system."),
            ],
        ),
        "03_deep_learning.ipynb": notebook(
            "03 - Deep Tabular Challenger",
            "A PyTorch MLP compares weighted binary cross-entropy with focal loss under the same chronological train and validation blocks.",
            [
                code("pd.read_csv(ART / 'deep_validation_comparison.csv')"),
                code("json.loads((ART / 'deep_challenger.json').read_text())"),
                code("display(Image(filename=str(FIG / '04_deep_training.png')))"),
                markdown("The neural network is a controlled challenger. It is not assumed to outperform gradient-boosted trees on low-dimensional tabular data."),
            ],
        ),
        "04_calibration_and_governance.ipynb": notebook(
            "04 - Calibration and Model Governance",
            "A dedicated calibration block fits Platt scaling after model selection. The policy and final test blocks are not used to fit the calibrator.",
            [
                code("pd.read_csv(ART / 'calibration_fit_summary.csv')"),
                code("display(Image(filename=str(FIG / '05_final_pr_calibration.png')))"),
                code("pd.read_csv(ART / 'temporal_split_summary.csv')"),
                markdown("Production monitoring should track prevalence, calibration intercept/slope, PR-AUC at capacity, amount distribution and score drift. Recalibration must use labels available after the verification delay."),
            ],
        ),
        "05_decision_analysis.ipynb": notebook(
            "05 - Decision Economics",
            "A review policy is selected on the policy block and frozen before final test evaluation. Capacity is enforced within elapsed-hour queue windows.",
            [
                code("policy = pd.read_csv(ART / 'policy_selection_results.csv')\npolicy"),
                code("test = pd.read_csv(ART / 'final_test_policy_results.csv')\ntest"),
                code("business = json.loads((ART / 'business_case.json').read_text())\nbusiness"),
                code("display(Image(filename=str(FIG / '06_business_waterfall.png')))"),
                code("display(Image(filename=str(FIG / '07_sensitivity_heatmap.png')))"),
            ],
        ),
        "06_causal_policy.ipynb": notebook(
            "06 - Semi-Synthetic Causal Policy Evaluation",
            "Known potential outcomes audit naive, IPTW, matching, g-computation and cross-fitted AIPW. This is not a causal claim from the ULB data.",
            [
                code("pd.read_csv(ART / 'causal_estimates.csv')"),
                code("pd.read_csv(ART / 'causal_balance.csv')"),
                code("json.loads((ART / 'causal_diagnostics.json').read_text())"),
                code("display(Image(filename=str(FIG / '08_causal_overlap.png')))"),
                code("display(Image(filename=str(FIG / '09_causal_balance.png')))"),
            ],
        ),
        "07_business_dashboard.ipynb": notebook(
            "07 - Business Dashboard",
            "Executive view of the frozen model, operational policy, fraud capture and scenario economics.",
            [
                code("display(Image(filename=str(FIG / '10_executive_dashboard.png')))"),
                code("business = json.loads((ART / 'business_case.json').read_text())\npd.Series(business['test_metrics'])"),
                markdown("Savings are estimated under explicit scenario assumptions and are shown in currency units. The dashboard does not represent realized results from an identified bank."),
            ],
        ),
    }
    for filename, nb in notebooks.items():
        nbf.write(nb, NOTEBOOK_DIR / filename)
    return list(notebooks)


def execute_all(names):
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for name in names:
            path = NOTEBOOK_DIR / name
            nb = nbf.read(path, as_version=4)
            InteractiveShell.clear_instance()
            shell = InteractiveShell.instance()
            execution_count = 0
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                execution_count += 1
                cell.execution_count = execution_count
                cell.outputs = []
                with capture_output(display=True) as captured:
                    result = shell.run_cell(cell.source, store_history=False)
                if captured.stdout:
                    cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=captured.stdout))
                if captured.stderr:
                    cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=captured.stderr))
                for output in captured.outputs:
                    cell.outputs.append(nbf.v4.new_output(
                        "display_data", data=output.data, metadata=output.metadata or {}
                    ))
                error = result.error_before_exec or result.error_in_exec
                if error is not None:
                    cell.outputs.append(nbf.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback.format_exception(type(error), error, error.__traceback__),
                    ))
                    raise error
            nbf.write(nb, path)
            print("Executed", name)
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    names = build_all()
    if args.execute:
        execute_all(names)


if __name__ == "__main__":
    main()
