from utils import *
from torch import cuda

# Define example inputs
examples = {
    'chest_xray': {
        'images': ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
        'context': "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea. PICC line placement.\nComparison: None.",
        'prompt': "How would you characterize the findings from <img0>?",
        'modality': "cxr",
        'task': "report"
    },
    'chest_xray_respiratory': {
        'images': ["./demo_ex/79eee504-b1b60ab8-5e8dd843-b6ed87aa-670747b1.png"],
        'context': "Age:70-80.\nGender:F.\nIndication: Respiratory distress.\nComparison: None.",
        'prompt': "How would you characterize the findings from <img0>?",
        'modality': "cxr",
        'task': "report"
    },
    'chest_xray_dual': {
        'images': [
            "./demo_ex/f39b05b1-f544e51a-cfe317ca-b66a4aa6-1c1dc22d.png",
            "./demo_ex/f3fefc29-68544ac8-284b820d-858b5470-f579b982.png"
        ],
        'context': "Age:80-90.\nGender:F.\nIndication: ___-year-old female with history of chest pain.\nComparison: None.",
        'prompt': "How would you characterize the findings from <img0><img1>?",
        'modality': "cxr",
        'task': "report"
    },
    'chest_xray_sob': {
        'images': ["./demo_ex/1de015eb-891f1b02-f90be378-d6af1e86-df3270c2.png"],
        'context': "Age:40-50.\nGender:M.\nIndication: ___-year-old male with shortness of breath.\nComparison: None.",
        'prompt': "How would you characterize the findings from <img0>?",
        'modality': "cxr",
        'task': "report"
    },
    'chest_xray_edema': {
        'images': ["./demo_ex/1EyfW5o8U2kS.jpg"],
        'context': "",
        'prompt': "Does the patient have pulmonary edema?",
        'modality': "cxr",
        'task': "vqa"
    },
    'chest_xray_tachycardia': {
        'images': ["./demo_ex/bc25fa99-0d3766cc-7704edb7-5c7a4a63-dc65480a.png"],
        'context': "Age:40-50.\nGender:F.\nIndication: History: ___F with tachyacrdia cough doe  // infilatrate\nComparison: None.",
        'prompt': "How would you characterize the findings from <img0>?",
        'modality': "cxr",
        'task': "report"
    },
    'derm_classification': {
        'images': ["./demo_ex/ISIC_0032258.jpg"],
        'context': "Age:70.\nGender:female.\nLocation:back.",
        'prompt': "What is primary diagnosis?",
        'modality': "derm",
        'task': "classification"
    },
    'derm_segmentation': {
        'images': ["./demo_ex/ISIC_0032258.jpg"],
        'context': "Age:70.\nGender:female.\nLocation:back.",
        'prompt': "Segment the lesion.",
        'modality': "derm",
        'task': "segmentation"
    },
    'ct_liver_1': {
        'images': ["./demo_ex/Case_01013_0000.nii.gz"],
        'context': "",
        'prompt': "Segment the liver.",
        'modality': "ct volume",
        'task': "segmentation"
    },
    'ct_liver_2': {
        'images': ["./demo_ex/Case_00840_0000.nii.gz"],
        'context': "",
        'prompt': "Segment the liver.",
        'modality': "ct volume",
        'task': "segmentation"
    }
}

# Configure generation parameters
params = {
    'num_beams': 1,
    'do_sample': True,
    'min_length': 1,
    'top_p': 0.9,
    'repetition_penalty': 1,
    'length_penalty': 1,
    'temperature': 0.1
}

def process_example(model, example_key, device=None):
    """
    Process a single medical imaging example
    """
    example = examples[example_key]
    
    # Generate predictions
    seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
        model,
        example['images'],
        example['context'],
        example['prompt'],
        example['modality'],
        example['task'],
        **params,
        device=device
    )
    
    # Print results
    print(f"\nResults for {example_key}:")
    print(f"{output_text}")
    
    if seg_mask_2d is not None:
        print(f"2D segmentation mask shape: {seg_mask_2d[0].shape}")  # H, W
    
    if seg_mask_3d is not None:
        print(f"Number of 3D slices: {len(seg_mask_3d)}")
        print(f"3D segmentation mask shape per slice: {seg_mask_3d[0].shape}")  # H, W

# Example usage
if __name__ == "__main__":
    # ---  Launch Model ---
    device = 'cuda' if cuda.is_available() else 'cpu'
    model_cls = registry.get_model_class('medomni') # medomni is the architecture name :)
    model = model_cls.from_pretrained('hyzhou/MedVersa').to(device).eval()
    # Process chest X-ray example
    process_example(model, 'chest_xray', device)
    
    # Process dermatology segmentation
    process_example(model, 'derm_segmentation', device)
    
    # Process CT liver segmentation
    process_example(model, 'ct_liver_1', device)

