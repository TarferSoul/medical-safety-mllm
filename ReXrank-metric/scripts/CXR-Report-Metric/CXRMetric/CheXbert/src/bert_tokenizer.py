import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm
import argparse

def get_impressions_from_csv(path):	
        df = pd.read_csv(path)
        imp = df['report']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        return imp

def tokenize(impressions, tokenizer):
        new_impressions = []
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
        for i in tqdm(range(impressions.shape[0])):
                tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                if tokenized_imp: #not an empty report
                        # Compatible with both old and new transformers versions
                        if hasattr(tokenizer, 'encode_plus'):
                                res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                        else:
                                # For transformers >= 4.36, use encode() or __call__()
                                res = tokenizer.encode(
                                        tokenized_imp,
                                        add_special_tokens=True,
                                        is_split_into_words=True
                                )
                        if len(res) > 512: #length exceeds maximum size
                                #print("report length bigger than 512")
                                res = res[:511] + [tokenizer.sep_token_id]
                        new_impressions.append(res)
                else: #an empty report
                        new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
        return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
        parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                            help='path to csv containing reports. The reports should be \
                            under the \"Report Impression\" column')
        parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                            help='path to intended output file')
        args = parser.parse_args()
        csv_path = args.data
        out_path = args.output_path
        
        # Use local bert-base-uncased to avoid downloading from HuggingFace Hub
        local_bert_path = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--google-bert--bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(local_bert_path)

        impressions = get_impressions_from_csv(csv_path)
        new_impressions = tokenize(impressions, tokenizer)
        with open(out_path, 'w') as filehandle:
                json.dump(new_impressions, filehandle)
