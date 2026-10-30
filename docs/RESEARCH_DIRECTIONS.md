# Arah kontribusi setelah Checkpoint 1

Ketiga arah di bawah berangkat dari hasil validation saat ini dan analisis paper. Mekanisme yang diusulkan belum dilatih atau dievaluasi. Hasil ensemble yang sudah tersedia adalah averaging biasa pada validation, terpisah dari evaluasi usulan kebijakan.

## A. Fusion heterogen secara selektif berdasarkan confidence

### Temuan yang memotivasi

Linear dan LoRA sama-sama benar pada 198/200 gambar, tetapi tidak berbagi gambar error. Averaging probabilitas keduanya benar pada 200/200 gambar validation yang sudah dianalisis. NLL averaging sekitar 0.0734, lebih tinggi daripada LoRA sendiri sekitar 0.0410. Jadi tujuan yang menarik mencakup kualitas probabilitas dan biaya, selain akurasi.

### Mekanisme yang diusulkan

Jalankan Linear terlebih dahulu. Jika confidence yang telah dikalibrasi memenuhi aturan penerimaan, gunakan prediksinya. Untuk input lain, jalankan LoRA lalu gabungkan prediksi kedua metode. Bandingkan mean probabilitas dan mean logits sebagai ablation aturan fusion, karena NLL keduanya berbeda pada eksplorasi validation saat ini. Temperature tiap model dan aturan routing dipilih pada data kalibrasi yang terpisah dari evaluasi akhir. Dua forward pass pada kasus fallback dan biaya memuat kedua model harus masuk pengukuran.

Pertanyaan penelitian: berapa banyak pemanggilan LoRA yang diperlukan untuk mempertahankan manfaat komplementaritas tanpa memperburuk NLL secara berarti?

### Pembanding dan pengujian

Bandingkan Linear, LoRA, always-on equal averaging, routing acak pada anggaran sama, gate confidence tanpa kalibrasi, dan gate dengan kalibrasi. Ukur accuracy, NLL, Brier, risk-coverage, proporsi input yang memakai model kedua, latency end-to-end dan memory residency. Hasil akhir memakai test setelah aturan dikunci, dengan beberapa seed untuk menilai kestabilan. Validation 200 gambar dapat dipakai untuk kalibrasi ke depan, tetapi cakupan 90 kelasnya harus dicatat. Jangan memilih aturan memakai label test atau hanya lima contoh error yang ditampilkan.

### Posisi terhadap literatur

Mai et al. sudah memeriksa komplementaritas dan ensemble. Caption Figure 4 menyebut majority vote, sedangkan rincian Figure 4 pada Appendix C menjelaskan average logits. Notebook membandingkan average logits dan average probabilitas secara eksplisit. EulerLoRA menghasilkan keragaman prediktif melalui variasi stokastik komponen LoRA. Lotfi et al. membandingkan ensembling dan routing untuk expert PEFT pada LLM. Usulan ini menguji cascade heterogen Linear-LoRA untuk klasifikasi visual dengan batas pemanggilan expert serta evaluasi probabilitas dan latency aktual. Perbedaan ini menjadi fokus eksperimen, sementara keuntungan mekanismenya masih perlu dibuktikan.

## B. Pembobotan noise yang menjaga kontribusi kelas kecil

### Temuan yang memotivasi

Training hanya memiliki 5-10 gambar per kelas. Ini membuat pengurangan kontribusi beberapa sampel berpotensi berdampak besar pada kelas tertentu. Tidak ada bukti dari analisis sekarang bahwa label Flowers102 yang digunakan berisik, atau bahwa error saat ini disebabkan oleh imbalance.

### Mekanisme yang diusulkan

Gunakan loss atau indikator kesulitan yang berasal dari proses training. Ubah indikator tersebut menjadi bobot relatif di dalam kelas, lalu beri batas minimum total kontribusi setiap kelas. Utamakan pembobotan lunak agar kelas kecil tetap menyumbang gradien. Informasi label yang sengaja dikorupsi hanya dipakai untuk audit eksperimen, tidak diberikan sebagai fitur atau mask oracle pada metode.

Pertanyaan penelitian: apakah pembobotan yang menjaga kontribusi kelas memperbaiki recall kelas kecil ketika label noise dan ketimpangan kelas muncul bersamaan?

### Pembanding dan pengujian

Pilot menggunakan dua faktor: noise 0% versus 20% dan distribusi asli versus long-tail terkontrol dengan target rasio head/tail sekitar 5, dengan minimal satu sampel per kelas. Korupsi hanya dilakukan pada training. Bandingkan LoRA biasa, class weighting, pembobotan noise global, serta pembobotan per kelas dengan contribution floor. Sertakan kontrol SSF untuk memeriksa apakah manfaat berasal dari loss umum atau interaksi dengan LoRA. Ukur macro-F1, balanced accuracy, recall head/tail dan NLL pada evaluasi bersih. Jalankan beberapa seed setelah pilot integrasi.

### Posisi terhadap literatur

Ma, Pan dan Wei (ICMR 2026) membahas robustness PEFT pada noisy labels melalui gradient-loss decoupling. Usulan kita berfokus pada kontribusi kelas kecil ketika weighting menangani noise dan class imbalance secara bersama. Metadata dan abstract paper ini telah diperiksa, tetapi rincian full text masih perlu ditelaah sebelum mengunci perbedaan mekanisme terhadap seluruh intervensi mereka. Arah ini membutuhkan training lebih banyak daripada analisis routing atas model yang sudah selesai.

## C. Pemilihan resep PEFT berdasarkan resource dan kualitas probabilitas

### Temuan yang memotivasi

Ketiga metode memiliki akurasi sama dan NLL berbeda. SSF melatih parameter lebih sedikit daripada LoRA, tetapi peak GPU allocated yang tercatat pada run SSF lebih besar. Catatan resource tersebut berasal dari kondisi node bersama dan membutuhkan profiling terkontrol sebelum diinterpretasikan sebagai perbedaan intrinsik metode. Batas host RAM cgroup 16 GiB juga memengaruhi kelayakan startup, walaupun perangkat GPU adalah A100 80 GB.

### Mekanisme yang diusulkan

Bentuk kandidat resep Linear, LoRA dan SSF dengan effective batch tetap 64, precision FP32, serta pilihan microbatch yang dinyatakan sebelumnya. Profiling singkat menghasilkan estimasi peak host RAM, GPU allocated/reserved, serta waktu update. Saring resep yang memenuhi batas resource, lalu pilih dengan kualitas validation dan biaya. Aturan pemilihan meminimalkan NLL dengan kendala accuracy minimum dan budget resource yang ditentukan sebelum evaluasi.

### Pembanding dan pengujian

Bandingkan pemilihan berdasarkan accuracy saja, parameter count saja, resource saja dan aturan gabungan. Biaya profiling dan pencarian ikut dihitung. Profiling menggunakan warmup, jumlah update dan kondisi pengukuran yang sama. Resep baru membutuhkan training kualitas yang sesuai sebelum skornya dibandingkan. Uji pada lebih dari satu task serta beberapa seed dan laporkan kelayakan, NLL, accuracy, biaya total serta keberhasilan menyelesaikan run. Reproduksi resource tidak mengubah batas server atau menyentuh proses pengguna lain.

### Posisi terhadap literatur

Appendix A paper utama menyatakan bahwa studi tidak berfokus pada memory usage dan FLOPs. Khan dan Siddique mengevaluasi PET dengan parameter budget dan kalibrasi pada klasifikasi chest X-ray multimodal. Usulan ini menguji pemilihan resep visual PEFT berdasarkan kelayakan host/GPU aktual, biaya profiling dan kualitas probabilitas pada satu pipeline yang sama. Bentuk kontribusinya adalah evaluasi terkontrol beserta kebijakan pemilihan resep.

## Prioritas eksperimen

A paling dekat dengan bukti komplementaritas yang sudah tersedia. C menjadi alternatif praktis yang terkait blocker nyata dan batas studi paper. B menjadi arah robustness yang memerlukan lebih banyak training serta pendalaman full text pembanding.

Ketiga arah masih berupa usulan eksperimen dengan pertanyaan penelitian, mekanisme, baseline dan ukuran keberhasilan yang ditetapkan. Gate, noise weighting dan resource selector belum dievaluasi.

## Referensi pembanding

- Mai et al., CVPR 2025: https://arxiv.org/abs/2409.16434
- Anumasa dan Liu, EulerLoRA, arXiv 2026: https://arxiv.org/abs/2608.01142
- Lotfi et al., Trade-offs in Ensembling, Merging and Routing Among Parameter-Efficient Experts, arXiv 2026: https://www.microsoft.com/en-us/research/publication/trade-offs-in-ensembling-merging-and-routing-among-parameter-efficient-experts/
- Ma, Pan dan Wei, ICMR 2026: https://doi.org/10.1145/3805622.3810828
- Khan dan Siddique, ICCIT 2025 sebagaimana dinyatakan pada arXiv record: https://arxiv.org/abs/2512.21508
