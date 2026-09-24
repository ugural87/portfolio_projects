# TCMB PolicyRAG

TCMB'nin 2019'dan itibaren yayımladığı Türkçe basın duyuruları üzerinde çalışan, kaynak gösteren ve değerlendirilebilen bir Retrieval-Augmented Generation uygulaması.

Sistem yalnızca “PDF yükle, soru sor” demosu değildir. Kaynak toplama, belge doğrulama, token tabanlı chunking, embedding cache, FAISS retrieval, tarih ve politika aracı filtreleri, cevap üretimi, kaynak doğrulama, retrieval benchmark, CLI, Streamlit, Docker ve CI aynı repodadır.

## Mimari

```text
TCMB yıllık duyuru arşivleri
        │
        ▼
Crawler ──► documents.jsonl + ham HTML + crawl report
        │
        ▼
Token-aware chunking (450 token, 70 overlap)
        │
        ▼
OpenAI embeddings + yerel cache
        │
        ▼
FAISS cosine index + chunk metadata
        │
        ├──► CLI: search / ask / evaluate
        │
        └──► Streamlit UI
                  │
                  ▼
          GPT-5.6 Luna grounded answer
          + duyuru numarası + tarih + URL
```

CLI ve Streamlit aynı `RAGService` sınıfını kullanır. UI bağımsız bir retrieval ya da prompt implementasyonu içermez.

## Corpus

Repo, 2019-2026 yıllık TCMB arşivlerinin 24 Eylül 2026 tarihli görüntüsünü içerir. Ingestion başlığa göre filtrelenmez: normal HTML duyuru sayfası bulunan **478 basın duyurusu** ile üç güncel referans sayfası corpus'a alınır. Standart numaralı duyuruların yanında eski adlandırmalı başkan açıklamaları ve röportajlar da yakalanır. Doğrudan PDF'e bağlanan sekiz yıllık açık mektup ile dört sunum metin corpus'unun kapsamı dışında bırakılmıştır; crawler bunları sessizce kaybetmez, `excluded_documents` alanında sırasıyla `open_letter_out_of_scope` ve `presentation_out_of_scope` gerekçeleriyle tüm arşiv metadatasıyla raporlar. Bu tercih, “Faiz Oranlarına İlişkin” başlıklı olup metin içinde menkul kıymet tesisi gibi makroihtiyati kararlar barındıran duyuruların kaybolmasını engeller.

`data/raw/documents.jsonl` her belge için metin, tarih, duyuru numarası, URL, belge türü, politika aracı etiketleri, ek bağlantıları ve içerik hash'i taşır. Güncel referans sayfalarının `published_date` alanı bilerek boştur; bunlar tarihsel karar sürümü değildir.

## Hızlı başlangıç

Python 3.11 veya üstü gerekir.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c constraints/typecheck.txt -e ".[dev,crawl]"
cp .env.example .env
```

`.env` içine API anahtarını yazın. Model adları ortam değişkenleriyle değiştirilebilir:

```dotenv
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-5.6-luna
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Önce indeks oluşturulur. Corpus embedding sonuçları `artifacts/index/embedding_cache` altında tutulduğu için sonraki indekslemelerde değişmeyen parçalar için tekrar API çağrısı yapılmaz. Kullanıcı sorguları bu kalıcı cache'e yazılmaz.
İndeksleme sırasında tamamlanan chunk sayısı, toplam chunk sayısı, yüzde ve cache hit
sayısı terminalde canlı olarak gösterilir.

```bash
tcmb-rag build-index
```

Ardından CLI ile yalnızca retrieval kontrol edilebilir:

```bash
tcmb-rag search "Menkul kıymet tesisi ne zaman sonlandırıldı?" --top-k 5
```

Veya kaynak bağlı cevap üretilebilir:

```bash
tcmb-rag ask "2023-2024 döneminde KKM'den TL mevduata geçiş nasıl teşvik edildi?"
```

Tarih ve politika aracı filtresi örneği:

```bash
tcmb-rag ask "Zorunlu karşılıklar nasıl değişti?" \
  --date-from 2024-01-01 \
  --date-to 2024-12-31 \
  --instrument reserve_requirements
```

Streamlit arayüzü:

```bash
streamlit run app/streamlit_app.py
```

## Neden hem CLI hem Streamlit?

CLI production ve araştırma işlerinde gereklidir: indeksleme, batch evaluation, log alma ve retrieval teşhisi tarayıcı arayüzüne bağlı kalmaz. Streamlit ise portföy demosunu ve kaynakların görsel incelenmesini sağlar. İkisi ortak servis kullandığı için davranışları ayrışmaz.

## Evaluation

`benchmarks/retrieval.json` içinde tarihsel, olgusal ve politika aracı odaklı başlangıç soruları vardır. Her soru bir veya daha fazla beklenen duyuru numarası taşır.

```bash
tcmb-rag evaluate --k 5
```

Çıktı `artifacts/evaluation/retrieval.json` dosyasına yazılır ve şu metrikleri içerir:

- Recall@k: beklenen duyuruların ilk k sonuçta bulunma oranı
- MRR: ilk doğru duyurunun sırasına duyarlılık
- Her soru için getirilen duyuru numaraları

Generation yanıtında yalnızca retrieval bağlamı kullanılabilir. Modelden `[K1]` biçiminde kaynak kimlikleri istenir; bilinmeyen kimlikler sonuçtan atılır. Bağlam yoksa sistem cevap uydurmak yerine abstain eder. Bu V1 benchmark'ı retrieval'ı ölçer; insan etiketli correctness/faithfulness değerlendirmesi V2 kapsamıdır.

## Crawler

Corpus'u güncellemek için:

```bash
python scripts/tcmb_crawler.py \
  --start-year 2019 \
  --scope all \
  --include-reference \
  --output data/raw
```

PDF dosyalarını ve sayfa bazlı PDF metnini de almak için `--download-assets --extract-pdf` eklenebilir. Crawler yalnızca resmi TCMB alan adlarına gider, yeniden deneme ve istek aralığı uygular, ham HTML'i saklar ve kısmi hataları `crawl_report.json` içinde raporlar.

Yıllar ayrı shard'lar halinde çekildiyse aynı snapshot deterministik biçimde birleştirilebilir. Merge sırasında URL bazlı tekilleştirme yapılır, politika etiketleri yeniden hesaplanır ve crawler şemasıyla uyumlu birleşik rapor üretilir:

```bash
python scripts/merge_crawls.py \
  data/parts/a data/parts/b data/parts/c data/parts/d data/parts/v101_delta \
  --output data/raw
```

Release ZIP'i bu beş kaynak shard'ı da içerir; dolayısıyla `make merge` commit edilen
snapshot'ı yeniden üretir. Aynı URL birden fazla shard'da varsa `fetched_at_utc` değeri
en yeni belge ve ona ait ham HTML kazanır. Çakışan URL sayısı `crawl_report.json`
içindeki `merge_conflicts` alanına yazılır.

## Docker

İndeksi host üzerinde bir kez oluşturup Streamlit'i başlatmak için:

```bash
docker compose run --rm app tcmb-rag build-index
docker compose up --build
```

Arayüz `http://localhost:8501` adresinde açılır. `artifacts` volume olarak bağlanır; indeks container kapanınca kaybolmaz.

## CI/CD

`ci.yml`, Python 3.11 ve 3.12 üzerinde testleri, Ruff kontrollerini ve strict mypy'ı
çalıştırır. Ayrıca commit edilen corpus'un kaynak shard'lardan birebir yeniden
üretilebildiğini kontrol eder, Python dağıtım paketlerini oluşturur ve Docker image'ını
build eder. Docker image'ında Streamlit health check bulunur.

`v1.0.5` gibi bir Git tag'i push edildiğinde `release.yml` kalite kapılarını yeniden
çalıştırır; wheel, source distribution, tam repo ZIP'i ve SHA-256 manifesti üretip GitHub
Release olarak yayımlar. Uygulamanın gerçek bir sunucuya deployment'ı hedef platform ve
secret yönetimi seçilmediği için bilinçli olarak workflow'a bağlanmamıştır.

## Proje yapısı

```text
tcmb-policyrag/
├── app/streamlit_app.py
├── benchmarks/retrieval.json
├── data/raw/
│   ├── documents.jsonl
│   ├── archive_index.json
│   └── crawl_report.json
├── data/parts/                 # yeniden üretilebilir snapshot shard'ları
├── scripts/
│   ├── merge_crawls.py
│   └── tcmb_crawler.py
├── src/tcmb_policyrag/
│   ├── chunking.py
│   ├── cli.py
│   ├── config.py
│   ├── corpus.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── index.py
│   ├── models.py
│   ├── retrieval.py
│   └── service.py
├── tests/
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Bilinen sınırlar

- TCMB sayfalarındaki anahtar sözcük etiketleri hukuki sınıflandırma değildir.
- Tarihsel karşılaştırma, retrieval'a bağlı çoklu belge sentezidir; ayrı bir olay tablosu henüz yoktur.
- Corpus TCMB basın duyurularını kapsar. BDDK ve Resmî Gazete düzenlemeleri kapsam dışıdır.
- İlk indeks API anahtarı ve embedding maliyeti gerektirir.
- Kaynağın corpus'ta bulunması, model yanıtının otomatik olarak doğru olduğu anlamına gelmez; benchmark ve kaynak incelemesi bunun için repodadır.
