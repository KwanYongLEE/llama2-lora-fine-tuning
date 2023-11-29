# Lora와 deepspeed를 사용하여 LLaMA2-Chat 미세 조정하기

두 개의 P100 (16G)에서 Llama-2-7b-chat 모델을 미세 조정합니다.

데이터 소스는 train과 validation 두 가지 데이터 소스로 구성된 alpaca 형식을 사용했습니다.

## 1. 그래픽 카드 요구 사항

16GB 이상의 메모리를 가진 그래픽 카드 (P100 또는 T4 이상), 하나 이상.

## 2. 소스 코드 클론하기

```bash
git clone https://github.com/KwanYongLEE/llama2-lora-fine-tuning
cd llama2-lora-fine-tuning
```

## 3. 종속성 환경 설치하기

```bash
# 가상 환경 생성하기
conda create -n llama2 python=3.9 -y
conda activate llama2
# github.com에서의 의존성 리소스 다운로드하기 (성공할 때까지 반복해야 하므로 따로 설치함)
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip install git+https://github.com/PanQiWei/AutoGPTQ.git -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip install git+https://github.com/huggingface/peft -i https://pypi.mirrors.ustc.edu.cn/simple
pip install git+https://github.com/huggingface/transformers -i https://pypi.mirrors.ustc.edu.cn/simple
# 다른 종속성 패키지 설치하기
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# bitsandbytes 검증하기
python -m bitsandbytes
```

## 4. 원래 모델 다운로드하기

```bash
python model_download.py --repo_id beomi/llama-2-koen-13b
```

## 5. 중국어 토큰 어휘 확장하기

```bash
# https://github.com/ymcui/Chinese-LLaMA-Alpaca.git 의 방법을 사용하여 중국어 토큰 어휘를 확장했습니다.
# 확장된 토큰 어휘는 merged_tokenizes_sp(전체 정밀도)와 merged_tokenizer_hf(반정밀도)에 있습니다.
# 미세 조정 시 --tokenizer_name ./merged_tokenizer_hf 매개변수를 사용합니다.
python merge_tokenizers.py \
  --llama_tokenizer_dir ./models/beomi/llama-2-koen-13b \
  --chinese_sp_model_file ./chinese_sp.model
```

## 6. 미세 조정 매개변수 설명

다음 매개변수를 조정할 수 있습니다:

| 매개변수                     | 설명                     | 값                                                           |
| ---------------------------- | ------------------------ | ------------------------------------------------------------ |
| load_in_bits                 | 모델 정밀도              | 4와 8, 메모리가 넘치지 않는다면 가능한 높은 정밀도 8을 선택 |
| block_size                   | token 최대 길이          | 우선 2048, 메모리 오버플로우 시 선택할 수 있음 1024, 512 등 |
| per_device_train_batch_size  | 훈련 시 각 장치당 배치 크기 | 메모리가 넘치지 않는다면 가능한 크게 선택                   |
| per_device_eval_batch_size   | 평가 시 각 장치당 배치 크기 | 메모리가 넘치지 않는다면 가능한 크게 선택                   |
| include                      | 사용된 그래픽 카드 시퀀스 | 두 개인 경우: localhost:1,2 (nvidia-smi에서 본 것과 동일하지 않을 수 있음) |
| num_train_epochs             | 훈련 에포크 횟수         | 최소 3회                                                     |

## 7. 미세 조정하기

```bash
chmod +x finetune-lora.sh
# 미세 조정하기
./finetune-lora.sh
# 백그라운드에서 미세 조정하기
pkill -9 -f finetune-lora
nohup ./finetune-lora.sh > train.log  2>&1 &
tail -f train.log
```

## 8. 테스트하기

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --base_model './models/daryl149/llama-2-7b-chat-hf' \
    --lora_weights 'output/checkpoint-2000' \
    --load_8bit # 이 매개변수를 추가하지 않으면 4bit를 사용합니다
```