import config
import os 
from CXRMetric.run_eval import calc_metric
from CXRMetric.run_eval import CompositeMetric
import pandas as pd

# gt_reports = config.GT_REPORTS
# predicted_reports = config.PREDICTED_REPORTS
# out_file = config.OUT_FILE
# use_idf = config.USE_IDF

if __name__ == "__main__":
    dataset_list = ['iu_xray','chexpert_plus','mimic-cxr', 'gradient_health']
    # model_list =['CX-Mind','zzy-RGv1-8b','RadPhi4VisionCXR']
    model_list =['MedGemma15']
    # dataset_list = ['gradient_health']
    # model_list =['0906_xiaoli']


    for dataset in dataset_list:
        for model in model_list:
            gt_reports = f"../findings/{dataset}/gt_reports_{model}.csv"
            predicted_reports = f"../findings/{dataset}/predicted_reports_{model}.csv"
            out_file = f"../results/{dataset}_findings/report_scores_{model}.csv"
            os.makedirs(f"../results/{dataset}_findings", exist_ok=True)
            # if os.path.exists(out_file) and len(pd.read_csv(out_file)) == 10000:
            #     print(f"Skipping {out_file} because it already exists")
            # else:
            print("Calculating metrics for:", gt_reports, predicted_reports, out_file)
            use_idf = False
            calc_metric(gt_reports, predicted_reports, out_file, use_idf)
