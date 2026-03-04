import pandas as pd
import os
import numpy as np

def calculate_average_metrics(findings_path, reports_path):
    """
    Calculate average metrics from findings and reports CSV files for ExampleModel model
    
    Args:
        findings_path: Path to the findings results directory
        reports_path: Path to the reports results directory
        
    Returns:
        Dictionary with average metrics
    """
    findings_metrics = {}
    reports_metrics = {}
    
    # Process findings metrics
    findings_dir = os.path.join(findings_path)
    if os.path.exists(findings_dir):
        # Read ratescore file
        ratescore_path = os.path.join(findings_dir, "ratescore_ExampleModel.csv")
        if os.path.exists(ratescore_path):
            ratescore_df = pd.read_csv(ratescore_path)
            if 'ratescore' in ratescore_df.columns:
                findings_metrics['ratescore'] = ratescore_df['ratescore'].mean()
        
        # Read green score file
        green_path = os.path.join(findings_dir, "results_green_ExampleModel.csv")
        if os.path.exists(green_path):
            green_df = pd.read_csv(green_path)
            if 'green_score' in green_df.columns:
                findings_metrics['green_score'] = green_df['green_score'].mean()
        
        # Read report scores file
        report_scores_path = os.path.join(findings_dir, "report_scores_ExampleModel.csv")
        if os.path.exists(report_scores_path):
            report_df = pd.read_csv(report_scores_path)
            for metric in ['bleu_score', 'bertscore', 'semb_score', 'radgraph_combined', 'RadCliQ-v0', 'RadCliQ-v1']:
                if metric in report_df.columns:
                    if metric == 'RadCliQ-v1':
                        findings_metrics['1/RadCliQ-v1'] = 1/report_df[metric].mean()
                    findings_metrics[metric] = report_df[metric].mean()
    
    # Process reports metrics
    reports_dir = os.path.join(reports_path)
    if os.path.exists(reports_dir):
        # Read ratescore file
        ratescore_path = os.path.join(reports_dir, "ratescore_ExampleModel.csv")
        if os.path.exists(ratescore_path):
            ratescore_df = pd.read_csv(ratescore_path)
            if 'ratescore' in ratescore_df.columns:
                reports_metrics['ratescore'] = ratescore_df['ratescore'].mean()
        
        # Read green score file
        green_path = os.path.join(reports_dir, "results_green_ExampleModel.csv")
        if os.path.exists(green_path):
            green_df = pd.read_csv(green_path)
            if 'green_score' in green_df.columns:
                reports_metrics['green_score'] = green_df['green_score'].mean()
        
        # Read report scores file
        report_scores_path = os.path.join(reports_dir, "report_scores_ExampleModel.csv")
        if os.path.exists(report_scores_path):
            report_df = pd.read_csv(report_scores_path)
            for metric in ['bleu_score', 'bertscore', 'semb_score', 'radgraph_combined', 'RadCliQ-v0', 'RadCliQ-v1']:
                if metric in report_df.columns:
                    if metric == 'RadCliQ-v1':
                        reports_metrics['1/RadCliQ-v1'] = 1/report_df[metric].mean()
                    reports_metrics[metric] = report_df[metric].mean()
    
    return {
        'findings': findings_metrics,
        'reports': reports_metrics
    }

def main():
    # Define paths
    # save as BLEU,BertScore,SembScore,RadGraph,1/RadCliQ-v1,RaTEScore,GREEN
    dataset_list = ['iu_xray','chexpert_plus','mimic-cxr','gradient_health']
    for dataset in dataset_list:
        findings_path = f"./{dataset}_findings"
        reports_path = f"./{dataset}_report"
    
        # Calculate average metrics
        avg_metrics = calculate_average_metrics(findings_path, reports_path)
        
        # Save results to CSV
        findings_df = pd.DataFrame([avg_metrics['findings']])
        reports_df = pd.DataFrame([avg_metrics['reports']])
        # only keep 3 decimal places
        findings_df = findings_df.round(3)
        reports_df = reports_df.round(3)
        
        # Reorder columns to match the desired order: BLEU,BertScore,SembScore,RadGraph,1/RadCliQ-v1,RaTEScore,GREEN
        column_order = ['bleu_score', 'bertscore', 'semb_score', 'radgraph_combined', '1/RadCliQ-v1', 'ratescore', 'green_score']
        
        # Filter to only include columns that exist in the dataframes
        findings_columns = [col for col in column_order if col in findings_df.columns]
        reports_columns = [col for col in column_order if col in reports_df.columns]
        
        findings_df = findings_df[findings_columns]
        reports_df = reports_df[reports_columns]
        
        os.makedirs(f"metric/{dataset}/findings", exist_ok=True)
        os.makedirs(f"metric/{dataset}/reports", exist_ok=True)
        findings_df.to_csv(f"metric/{dataset}/findings/ExampleModel.csv", index=False)
        reports_df.to_csv(f"metric/{dataset}/reports/ExampleModel.csv", index=False)
        
        print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()
