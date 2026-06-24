# Plaka Tanıma — YOLO11m + BoT-SORT + fast-alpr

Videodaki araçların plakalarını tespit edip okuyan, sonucu her araç için
stabil şekilde ekrana basan bir bilgisayarla görü pipeline'ı.

Mantık basit: aracı tespit et, takip et, ekranın belirli bir bandına
(zon) girince plakasını okumaya başla, birkaç okumanın en güveniliriyle
plakayı **kilitle** ve araç kadrajda kaldığı sürece hep o plakayı göster.

---

## Neden bu yaklaşım?

ALPR tarafında en sık karşılaşılan problem, aynı plakanın kare kare farklı
okunması. Işık, açı, motion blur derken bir frame'de `34ABC123`, diğerinde
`34A8C123` çıkabiliyor. Tek bir frame'e güvenmek yerine:

- Araç zona girdiğinde birden fazla geçerli okuma topluyorum,
- Yeterli okuma birikince (`N_VALID_READS`) en yüksek OCR confidence'lı
  olanı seçip kilitliyorum,
- Kilitlendikten sonra araç zondan çıksa bile takip ettiğim sürece aynı
  plakayı gösteriyorum.

Bu sayede çıktı titremiyor, her araç için tek ve net bir plaka oluyor.

---

## Çalışma akışı

1. **Araç tespiti** — `YOLO11m` ile sadece `car / bus / truck` (COCO 2, 5, 7)
2. **Takip** — `BoT-SORT` (`persist=True`), her araca frame'ler boyunca sabit
   bir track id
3. **Zon kontrolü** — aracın merkez `cy`'si frame yüksekliğinin `7/10 – 9/10`
   bandındaysa plaka okuması tetiklenir
4. **Plaka okuma** — araç crop'u `fast-alpr`'a verilir
   (`yolo-v9-t` detektör + `cct-s-v2-global` OCR)
5. **Kilitleme** — geçerli okumalardan en yüksek confidence'lı olan seçilir
6. **Çizim** — plaka kutusu (zon içindeyken) + kilitlenmiş plaka metni

---

## Kurulum

```bash
pip install ultralytics opencv-python
pip install "fast-alpr[onnx-gpu]"   # CUDA'lı NVIDIA GPU için
```

CPU'da çalıştıracaksan `[onnx-gpu]` yerine `[onnx]` kullanabilirsin (yavaş olur).

> **Not:** GPU'da `onnxruntime`'ın CUDA/cuDNN'i bulabilmesi için kodda
> `ALPR()` çağrısından önce `ort.preload_dlls()` çalıştırıyorum. Bunu
> atlarsan ONNX sessizce CPU'ya düşebiliyor.

---

## Kullanım

`plaka_tanima.py` içindeki ayarları düzenle ve dosyayı çalıştır:

```python
VIDEO_PATH  = "video3.mp4"          # girdi videosu
OUTPUT_PATH = "output_plaka3.mp4"   # çıktı videosu
```

```bash
python plaka_tanima.py
```

İşlem bittiğinde işaretlenmiş video `OUTPUT_PATH`'e yazılır. Canlı izlemek
istersen `SHOW_WINDOW = True` yap.

---

## Ayarlar

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| `VEHICLE_CLASSES` | Tespit edilecek COCO sınıfları (car/bus/truck) | `[2, 5, 7]` |
| `VEHICLE_CONF` | Araç tespit eşiği | `0.40` |
| `ZONE_TOP_RATIO` | Zonun üst sınırı (frame yüksekliğinin oranı) | `0.70` |
| `ZONE_BOTTOM_RATIO` | Zonun alt sınırı | `0.90` |
| `N_VALID_READS` | Kilitlemeden önce gereken geçerli okuma sayısı | `5` |
| `PLATE_MIN_CONF` | Bir okumayı adaya almak için OCR eşiği | `0.50` |
| `BOX_THICK` / `FONT_SCALE` | Çizim ölçekleri (4K için büyütüldü) | `5` / `2.2` |

> Zon oranları kameranın açısına göre ayarlanmalı. Plakanın net göründüğü
> banda denk getirmek okuma kalitesini ciddi şekilde artırıyor.

---

## Plaka formatı — Avrupa / Türkiye

Bu önemli bir nokta, o yüzden ayrı başlık açtım.

**Kullandığım modeller (hem plaka detektörü hem OCR) ülkeden bağımsız,
global çalışıyor — Türk plakalarını da sorunsuz okur.** Yani modeli
değiştirmene gerek yok.

Kodda "Avrupa'ya ayarlı" olan tek şey **plaka doğrulama regex'i**. Test
videom Avrupa plakalı olduğu için doğrulamayı gevşek bir EU formatına göre
yaptım (`5–9 alfanumerik, en az 1 harf + 1 rakam`). Bu kontrol, OCR'dan
gelen saçma okumaları (örn. yarım okunmuş, sadece rakam vb.) eler.

Türkiye'de kullanacaksan **sadece doğrulama mantığını** Türk plaka formatına
çevirmen yeterli. Aşağıda hangi kısımların değişeceği adım adım var.

### Türk plaka formatı

```
İl kodu (01–81) + 1–3 harf + 2–5 rakam
Örnekler: 06 ABC 123 · 34 AB 1234 · 35 A 1234
```

### Değişecek kısımlar

**1) Türk plaka regex'ini ekle** (dosyanın üst kısmına, `re` import'undan sonra):

```python
# Türk plakası: il kodu (01-81) + 1-3 harf + 2-5 rakam
TR_PLATE_RE = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,5}$")
```

**2) Doğrulama fonksiyonunu Türkiye'ye göre yaz.** `is_valid_eu_plate`
fonksiyonunu olduğu gibi bırakıp yanına bir Türk versiyonu ekleyebilirsin:

```python
def is_valid_tr_plate(text):
    t = normalize_plate(text)          # normalize_plate aynen kalıyor
    return bool(TR_PLATE_RE.match(t))
```

**3) `en_iyi_plaka` içindeki kontrolü değiştir.** Tek satır:

```python
# ÖNCE (Avrupa):
if not is_valid_eu_plate(r.ocr.text):
    continue

# SONRA (Türkiye):
if not is_valid_tr_plate(r.ocr.text):
    continue
```

Bu kadar. `normalize_plate`, kilitleme mantığı, çizim, modeller — hepsi aynı
kalıyor.

> **İnce ayar:** Türk plakalarında `Q, W, X` ve Türkçe'ye özgü harfler
> kullanılmaz. OCR yine de bunları üretebilir. İstersen harf grubunu
> `[ABCDEFGHIJKLMNOPRSTUVYZ]{1,3}` gibi kısıtlayarak yanlış okumaları
> biraz daha eleyebilirsin — ama çoğu durumda yukarıdaki regex yeterli.

---

## Klasör yapısı

```
.
├── plaka_tanima.py     # ana pipeline
├── yolo11m.pt          # araç tespit modeli (ilk çalıştırmada iner)
└── README.md
```

`fast-alpr` modelleri ilk çalıştırmada otomatik indirilir.

---

## Notlar

- Araç tespiti `yolo11m` ile yapılıyor; daha hızlı istersen `yolo11n`,
  daha hassas istersen `yolo11l`/`yolo11x` denenebilir.
- `BoT-SORT` `model.track(persist=True)` ile decoupled değil entegre
  kullanılıyor; tek video işleme senaryosu için bu yeterli ve temiz.
- Çizim ölçekleri 4K (3840×2160) için büyütüldü. Daha küçük çözünürlükte
  `FONT_SCALE` ve `BOX_THICK` değerlerini düşür.
