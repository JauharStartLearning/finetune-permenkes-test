

# Qwen-8B LoRA - Permenkes (7 Epochs)

Model ini adalah hasil *Supervised Fine-Tuning* (SFT) menggunakan metode LoRA (Parameter-Efficient Fine-Tuning) pada model *base* Qwen 8B. Model ini dilatih menggunakan dataset Permenkes (Peraturan Menteri Kesehatan) untuk memahami dan menjawab konteks regulasi kesehatan. 

Pelatihan dioptimalkan menggunakan pustaka **Unsloth** untuk efisiensi VRAM dan kecepatan *training*.

## 📌 Informasi Model
* **Base Model:** Qwen 8B
* **Hugging Face Hub:** [Jauharul/qwen3-8b-lora-permenkes-7epoch](https://huggingface.co/Jauharul/qwen3-8b-lora-permenkes-7epoch)
* **Library/Framework:** Unsloth, TRL, Hugging Face `transformers`, `peft`

## ⚙️ Konfigurasi LoRA
Model ini menggunakan konfigurasi LoRA berikut untuk efisiensi:
* **Rank (r):** 32
* **LoRA Alpha:** 32
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Dropout:** 0 (Dioptimalkan)
* **Bias:** none

## 📊 Hyperparameter Pelatihan
* **Epochs:** 7
* **Learning Rate:** 2e-4 (Linear Scheduler)
* **Optimizer:** AdamW 8-bit
* **Train Batch Size:** 4 per device (Gradient Accumulation: 4) -> *Effective Batch Size: 16*
* **Eval Batch Size:** 2
* **Warmup Steps:** 20
* **Weight Decay:** 0.001
* **Gradient Checkpointing:** "unsloth" (Menghemat ~30% VRAM)
* **Tracking:** Weights & Biases (wandb)
---

## 📈 Hasil Pelatihan (Result)

<p align="center">
  <img width="49%" alt="TrainingLoss" src="https://github.com/user-attachments/assets/68eee0fa-5067-4276-b977-7b306b2d3c42" />
  <img width="49%" alt="EvalLoss" src="https://github.com/user-attachments/assets/4c26267e-caa6-47d6-af80-dd27ac159705" />
</p>

> 💡 **Pemilihan Checkpoint (Optimal Model)**  
> Terlihat bahwa model mulai menunjukkan tanda-tanda *overfitting* setelah **step ke-135** (sekitar epoch ke-4.2). Oleh karena itu, model dengan **checkpoint epoch ke-4** (Commit: `61d47d10cd6472db02f9913893877c266ecff3dd`) yang memiliki nilai *Eval Loss* terendah dipilih untuk menguji respons terhadap dataset *test*.

<p align="center">
  <img width="85%" alt="Checkpoint Details" src="https://github.com/user-attachments/assets/a0ce0d30-3d8b-47ba-b2bc-1f971ebd22b3" />
</p>

📌 *Catatan: Hasil inferensi dari model setelah tahap fine-tuning ini dapat dilihat selengkapnya pada file **`Test-Result-FineTune`**.*

---

## 🚀 Cara Menggunakan Model (Inference)

Anda bisa menjalankan model ini menggunakan **Unsloth** untuk kecepatan inference (2x lebih cepat):

```python
from unsloth import FastLanguageModel

# Load model dan tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Jauharul/qwen3-8b-lora-permenkes-7epoch",
    max_seq_length = 2048, # Sesuaikan jika perlu
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Contoh inference
inputs = tokenizer(
    [
        "Menurut Permenkes, apa saja standar pelayanan minimal rumah sakit?"
    ], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs))
