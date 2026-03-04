cd /mnt/shared-storage-user/xieyuejin/MLLM-Safety/MedicalSafety/LLaMA-Factory
export CUDA_HOME=/mnt/shared-storage-user/xieyuejin/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# pip install -e ".[torch,metrics]" --no-build-isolation
# pip install transformers===4.57.1
# pip3 install beautifulsoup4
pip install deepspeed===0.16.9 -i  http://mirrors.h.pjlab.org.cn/pypi/simple --trusted-host mirrors.h.pjlab.org.cn 
# pip install 

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_vl_8b_full_sft_260222.yaml