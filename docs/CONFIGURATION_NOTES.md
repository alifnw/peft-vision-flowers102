# Konfigurasi Checkpoint 1

Konfigurasi aktual disimpan pada result.json setiap metode. Keputusan awal menggunakan satu konfigurasi yang dapat dijalankan dan diaudit pada satu task. Eksperimen ini belum menjalankan hyperparameter search paper.

## Yang mengikuti paper

ViT-B/16 supervised ImageNet-21K, AdamW, effective batch 64, cosine schedule, dan 100 epoch mengikuti eksperimen VTAB pada Appendix A.1. Paper menonaktifkan augmentation untuk menjaga perbandingan pada 19 task. Drop path 0.1 mengikuti rekomendasi regularisasi low-shot pada bagian 3 dan Figure 11. Manfaat drop path pada run kita belum diisolasi melalui ablation.

Learning rate 0.001 dan weight decay 0.0001 dipilih dari grid [0.001, 0.01] dan [0.0001, 0.001]. LoRA rank 8 termasuk grid bottleneck [1, 8, 16, 32] pada Table 3. Nilai ini merupakan pilihan baseline awal, belum menjadi konfigurasi optimal untuk Flowers102.

## Yang mengikuti implementasi resmi

Default VTAB upstream menetapkan warmup 10 epoch, learning rate awal 1e-6, minimum learning rate 1e-5, seed 42, crop size 224 dan normalized=False. Dataset loader menggunakan resize bicubic lalu ToTensor. Mempertahankan default mengurangi perubahan pendekatan selama reproduksi awal, dengan konsekuensi efek normalization dan augmentation khusus dataset belum diuji.

Implementasi native LoRA memakai satu down-projection A berukuran 768 -> 8 dan up-projection B berukuran 8 -> 2304, lalu membagi delta menjadi Q, K dan V. Ada 294.912 parameter LoRA di 12 blok, serta 78.438 parameter classifier. SSF memiliki 205.824 parameter scale/shift di lokasi fitur native, ditambah classifier yang sama. Diagram merangkum struktur aktual ini. Jalur inisialisasi penuh dari constructor upstream dipertahankan, sehingga laporan tidak mengasumsikan bahwa inisialisasi lokal sebuah submodule adalah keadaan akhirnya setelah seluruh constructor selesai.

LoRA menggunakan dropout input 0.1 dan merge_factor 1 dari default upstream. Merge factor merupakan pengali delta pada forward pass. Implementasi ini tidak menerapkan faktor alpha/r terpisah. Formula LoRA dalam Appendix B paper mengilustrasikan pembaruan Q/V, sementara kode yang dipin memperbarui Q/K/V melalui proyeksi bersama. Laporan mengikuti implementasi yang benar-benar digunakan.

## Penyesuaian karena resource

Node memakai A100 80 GB bersama dengan batas host RAM cgroup 16 GiB. Beberapa startup terhenti ketika headroom RAM turun. Catatan ini menjelaskan perubahan operasional dan bukan skor kualitas metode.

LoRA dan SSF memakai microbatch 8 dengan effective batch 64. Loss microbatch dibobotkan berdasarkan banyak sampel, lalu optimizer melakukan satu update setelah satu effective batch. Ada 12 update per epoch dan 1.200 sepanjang 100 epoch. Linear menggunakan batch langsung 64. Perbedaan dropout draws dan urutan penjumlahan floating point membuat trajectory keduanya tidak dijamin identik.

Workers 0, pin_memory=False, satu thread, FP32 serta allocator trimming membatasi alokasi host pada run ini. Checkpoint menyimpan parameter yang dilatih, optimizer, scheduler dan RNG. Hash classifier awal, split, serta frozen backbone diperiksa. Kebijakan resource dan sumber ketika resume tersimpan di provenance. Peak memory dan runtime tidak digunakan sebagai ranking efisiensi karena kondisi node dan batching tidak terkontrol.

`drop_last=True` mengikuti loader upstream. Sebanyak 768 dari 800 gambar yang diacak berkontribusi pada setiap epoch. Tidak ada alasan empiris untuk menyatakan bahwa membuang batch terakhir meningkatkan kualitas model kita.

## Perbedaan lingkup terhadap hasil paper

Penulis melakukan search pada split 800/200, lalu melatih ulang pada 1.000 gambar dan melaporkan performa test. Kita melaporkan validation epoch 100 tanpa search dan tanpa tahap trainval-test. Mirror lolos pemeriksaan integritas, tetapi exact membership terhadap arsip penulis belum terverifikasi. Dengan demikian, selisih skor terhadap paper tidak mengisolasi perbedaan implementasi atau kualitas metode.

Persiapan data memeriksa metadata dan sampel file test untuk integritas. Training runner tidak memakai test untuk training, konfigurasi, atau pengukuran performa. Content-level train-test duplicate checking belum dilakukan.

## Portabilitas runner

Runner saat ini menerima --reference-result PATH untuk pemeriksaan hash inisialisasi opsional. Tanpa argumen itu, run tidak membutuhkan hasil eksperimen sebelumnya. Referensi yang diberikan harus tersedia dan cocok pada hash classifier awal serta frozen backbone. Perbaikan ini dilakukan setelah eksperimen tercatat dan tidak mengubah provenance historis atau hasil training.

## Sumber

- Paper, section 3, section 4, Appendix A.1, Table 3 dan Figure 11: https://arxiv.org/abs/2409.16434
- Pinned default YAML: https://github.com/OSU-MLB/ViT_PEFT_Vision/blob/cea7e21b2ddecef8172f5d177da5a29d531c4279/experiment/config/default_vtab_processed.yml
- Native LoRA: https://github.com/OSU-MLB/ViT_PEFT_Vision/blob/cea7e21b2ddecef8172f5d177da5a29d531c4279/model/lora.py
- Native dataset transforms: https://github.com/OSU-MLB/ViT_PEFT_Vision/blob/cea7e21b2ddecef8172f5d177da5a29d531c4279/data/dataset/vtab_processed.py
