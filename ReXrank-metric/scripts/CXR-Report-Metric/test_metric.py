import config
import os 
from CXRMetric.run_eval import calc_metric
from CXRMetric.run_eval import CompositeMetric



if __name__ == "__main__":
    dataset_list = ['iu_xray','chexpert_plus','mimic-cxr', 'gradient_health']
    model_list =['MedGemma15']

    for dataset in dataset_list:
        for model in model_list:
            gt_reports = f"../reports/{dataset}/gt_reports_{model}.csv"
            predicted_reports = f"../reports/{dataset}/predicted_reports_{model}.csv"
            out_file = f"../results/{dataset}_report/report_scores_{model}.csv"
            os.makedirs(f"../results/{dataset}_report", exist_ok=True)
            if os.path.exists(out_file):
                print(f"Skipping {out_file} because it already exists")
            else:
                print("Calculating metrics for:", gt_reports, predicted_reports, out_file)
                use_idf = False
                calc_metric(gt_reports, predicted_reports, out_file, use_idf)

# python test_metric_findings.py
# python test_metric.py