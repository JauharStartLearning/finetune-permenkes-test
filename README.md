

# Qwen-8B LoRA - Permenkes (7 Epochs)

Model ini adalah hasil *Supervised Fine-Tuning* (SFT) menggunakan metode QLoRA (Parameter-Efficient Fine-Tuning) pada model *base* Qwen 8B 4bit. Model ini dilatih menggunakan dataset Permenkes (Peraturan Menteri Kesehatan Nomor 10 Tahun 2024). 

## ⚙️ Proses Pembuatan Data Latih

Penyusunan dataset dilakukan melalui dua tahapan utama untuk memastikan kualitas, keakuratan substansi, dan variasi kalimat:

* **Tahap 1: Ekstraksi Q&A Basis (Unik)**
  Pasangan Pertanyaan dan Jawaban (Q&A) diekstraksi dari setiap pasal Permenkes. Fokus utama pada tahap ini adalah mengamankan substansi hukum; setiap pertanyaan dipastikan memiliki inti pembahasan yang unik dan sepenuhnya berbeda satu sama lain (bukan sekadar variasi kalimat).
  > **Hasil:** Diperoleh 70 pasangan Q&A basis yang 100% unik.

* **Tahap 2: Augmentasi Data (Parafrase)**
  Untuk memperkaya pemahaman model terhadap berbagai cara user bertanya, ke-70 pasangan Q&A basis tersebut diparafrase masing-masing sebanyak 10 variasi gaya bahasa.
  > **Hasil Akhir:** Terkumpul 700 pasangan Q&A komprehensif yang siap digunakan untuk proses *training* model.

  > Kode pembuatan data dapat dilihat di folder **Data-Generation-Code** 
---

## 📌 Informasi & Spesifikasi Model

| Kategori | Keterangan |
| :--- | :--- |
| **Base Model** | unsloth/Qwen3-8B-unsloth-bnb-4bit |
| **Repository** | [Jauharul/qwen3-8b-lora-permenkes-7epoch](https://huggingface.co/Jauharul/qwen3-8b-lora-permenkes-7epoch) |
| **Framework** | Unsloth, TRL, Hugging Face `transformers`, `peft` |

---

## ⚙️ Proses Fine-Tuning Model

Pelatihan model dilakukan menggunakan teknik **QLoRA (Quantized Low-Rank Adaptation)** untuk memaksimalkan efisiensi komputasi tanpa mengorbankan performa bahasa model. Berikut adalah tahapan utamanya:

* **Kuantisasi Model Dasar (4-bit):** Model Qwen 8B dimuat dalam presisi 4-bit menggunakan ekosistem Unsloth dan BitsAndBytes. Pendekatan ini menekan penggunaan VRAM secara drastis selama proses pelatihan.
* **Injeksi Adapter LoRA:** Alih-alih memperbarui seluruh 8 Miliar parameter, pelatihan hanya difokuskan pada *adapter* kecil yang disematkan pada lapisan atensi dan *feed-forward* krusial.
* **Akselerasi dengan Unsloth:** Proses *Supervised Fine-Tuning* (SFT) dieksekusi menggunakan pustaka TRL. Penggunaan fitur *Gradient Checkpointing* dari Unsloth memungkinkan pelatihan berjalan jauh lebih cepat sekaligus menghemat VRAM hingga 30%.
* **Penggabungan dan Ekspor (GGUF):** Setelah model mencapai nilai *loss* evaluasi terendah pada *checkpoint* epoch ke-4, bobot LoRA digabungkan (*merged*) secara permanen ke model dasar 16-bit. Model akhir ini kemudian dikuantisasi ulang dan diekspor ke format **GGUF (`q4_k_m`)** agar siap digunakan untuk inferensi ringan di CPU.

---

## 📊 Konfigurasi Teknis & Hyperparameter

Parameter teknis di bawah ini disetel untuk menyeimbangkan stabilitas pelatihan dan efisiensi memori.

**Konfigurasi LoRA:**
* **Rank (r) / Alpha:** 32 / 32
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Dropout / Bias:** 0 (Dioptimalkan) / None

**Hyperparameter Pelatihan:**

| Hyperparameter | Konfigurasi |
| :--- | :--- |
| **Epochs** | 7 |
| **Learning Rate** | `2e-4` (Linear Scheduler) |
| **Optimizer** | AdamW 8-bit |
| **Train Batch Size** | 4 per device (Gradient Accumulation: 4) |
| **Effective Batch Size** | 16 |
| **Eval Batch Size** | 2 |
| **Warmup Steps** | 20 |
| **Weight Decay** | `0.001` |
| **Gradient Checkpointing**| `unsloth` |
| **Tracking** | Weights & Biases (wandb) |
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
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/JauharStartLearning/finetune-permenkes-test/blob/main/finetune-Qwen(Qlora)-Code.ipynb)

## 🚀 Cara Menggunakan Model (Inference)

Anda bisa menjalankan model ini menggunakan **Unsloth** untuk kecepatan inference (2x lebih cepat):

```python
!pip install unsloth
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

import os
from huggingface_hub import hf_hub_download
from gpt4all import GPT4All

# 1. Tentukan model dari Hugging Face (Model Qwen ini sangat ringan untuk dicoba)
repo_id = "Jauharul/qwen3-8b-lora-permenkes-7epoch-GGUF"
nama_model = "qwen3-8b.Q4_K_M.gguf"
lokasi_folder = "." # Titik berarti folder saat ini (E:\TestModel)

# 2. Unduh otomatis jika file model belum ada di komputer
if not os.path.exists(nama_model):
    print("Mengunduh model... (Hanya dilakukan 1x)")
    hf_hub_download(repo_id=repo_id, filename=nama_model, local_dir=lokasi_folder)

# 3. Muat model (allow_download=False mencegah error 404 server GPT4All)
print("\nSedang memuat model ke memori...")
model = GPT4All(model_name=nama_model, model_path=lokasi_folder, allow_download=False)

# 4. Tes eksekusi
prompt = "Mohon sebutkan jenis Dokumen Hukum yang termasuk dalam sistem pengelolaan JDIH Kemenkes."
print(f"\nPertanyaan: {prompt}")
print("Jawaban: ", end="")

for token in model.generate(prompt, max_tokens=250, streaming=True):
    print(token, end="", flush=True)
print("\n")
