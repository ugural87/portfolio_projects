from __future__ import annotations

import html
import textwrap
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/NYC_Taxi_Duration_Production_ML_Engineering_Guide.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1769AA")
CYAN = colors.HexColor("#2CB1BC")
GOLD = colors.HexColor("#F6C85F")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF4FB")
GREEN = colors.HexColor("#2F855A")
RED = colors.HexColor("#C53030")
LIGHT = colors.HexColor("#F7FAFC")


def register_fonts() -> None:
    base = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", str(base / "DejaVuSansMono.ttf")))
    addMapping("DejaVu", 0, 0, "DejaVu")
    addMapping("DejaVu", 1, 0, "DejaVu-Bold")
    addMapping("DejaVu", 0, 1, "DejaVu-Oblique")


class GuideDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
            title="NYC Taxi Duration - Production ML Engineering Guide",
            author="Ugur Alkan",
            subject="End-to-end production machine learning educational reference",
            invariant=1,
        )
        frame = __import__("reportlab.platypus", fromlist=["Frame"]).Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self._decorate_page))

    def _decorate_page(self, canvas: object, doc: object) -> None:
        page = self.page
        if page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10.5 * mm, "NYC Taxi Duration | Production ML Engineering Guide")
        canvas.drawRightString(A4[0] - 18 * mm, 10.5 * mm, f"{page}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in {"H1", "H2", "H3"}:
                level = {"H1": 0, "H2": 1, "H3": 2}[style_name]
                text = flowable.getPlainText()
                key = f"h{level}-{self.seq.nextf('heading')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            fontName="DejaVu-Bold",
            fontSize=27,
            leading=32,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", fontName="DejaVu", fontSize=13, leading=19, textColor=PALE
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle", fontName="DejaVu", fontSize=12, leading=18, textColor=MUTED
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="DejaVu-Bold",
            fontSize=18,
            leading=23,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="DejaVu-Bold",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=sample["Heading3"],
            fontName="DejaVu-Bold",
            fontSize=10.5,
            leading=14,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            fontName="DejaVu",
            fontSize=9,
            leading=13.4,
            textColor=INK,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "Small": ParagraphStyle(
            "Small", fontName="DejaVu", fontSize=7.5, leading=10.5, textColor=MUTED
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            fontName="DejaVu",
            fontSize=8.7,
            leading=12.5,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-7,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "Code": ParagraphStyle(
            "Code",
            fontName="DejaVuMono",
            fontSize=5.7,
            leading=7.7,
            leftIndent=7,
            rightIndent=7,
            borderColor=colors.HexColor("#CBD5E0"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#F8FAFC"),
            textColor=colors.HexColor("#1A202C"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            fontName="DejaVu",
            fontSize=8.8,
            leading=13,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=8,
            borderColor=CYAN,
            borderWidth=0,
            borderLeft=4,
            borderPadding=8,
            backColor=PALE,
            spaceBefore=5,
            spaceAfter=8,
        ),
    }


def esc(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br/>")


def p(story: list[Flowable], text: str, style: str = "Body") -> None:
    story.append(Paragraph(text, STYLES[style]))


def h(story: list[Flowable], text: str, level: int = 1) -> None:
    story.append(Paragraph(text, STYLES[f"H{level}"]))


def bullets(story: list[Flowable], items: list[str]) -> None:
    for item in items:
        story.append(Paragraph(f"• {item}", STYLES["Bullet"]))


def code(story: list[Flowable], content: str) -> None:
    wrapped: list[str] = []
    for line in content.rstrip().splitlines():
        if len(line) <= 118:
            wrapped.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        segments = textwrap.wrap(
            line.strip(),
            width=max(40, 114 - indent),
            replace_whitespace=False,
            drop_whitespace=False,
        )
        for index, segment in enumerate(segments):
            prefix = " " * indent if index == 0 else " " * (indent + 4)
            wrapped.append(prefix + segment.rstrip() + ("  ↩" if index < len(segments) - 1 else ""))
    story.append(Preformatted("\n".join(wrapped), STYLES["Code"], maxLineLength=122))


def table(
    story: list[Flowable], rows: list[list[object]], widths: list[float] | None = None
) -> None:
    formatted = [[Paragraph(esc(cell), STYLES["Small"]) for cell in row] for row in rows]
    obj = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([obj, Spacer(1, 7)])


def pipeline_drawing() -> Drawing:
    drawing = Drawing(475, 245)
    nodes = [
        (10, 190, 96, 34, "TLC Parquet"),
        (134, 190, 96, 34, "Validate"),
        (258, 190, 96, 34, "Time split"),
        (372, 190, 96, 34, "Train + gate"),
        (372, 110, 96, 34, "Artifact image"),
        (258, 110, 96, 34, "Staging"),
        (134, 110, 96, 34, "Production"),
        (10, 110, 96, 34, "Monitoring"),
        (134, 30, 96, 34, "Retraining"),
    ]
    for x, y, w, ht, label in nodes:
        fill = PALE if label not in {"Train + gate", "Production"} else colors.HexColor("#FFF4D6")
        drawing.add(Rect(x, y, w, ht, rx=5, ry=5, fillColor=fill, strokeColor=BLUE, strokeWidth=1))
        drawing.add(
            String(
                x + w / 2,
                y + 13,
                label,
                fontName="DejaVu-Bold",
                fontSize=7.6,
                textAnchor="middle",
                fillColor=NAVY,
            )
        )
    arrows = [
        (106, 207, 134, 207),
        (230, 207, 258, 207),
        (354, 207, 372, 207),
        (420, 190, 420, 144),
        (372, 127, 354, 127),
        (258, 127, 230, 127),
        (134, 127, 106, 127),
        (58, 110, 58, 47),
        (106, 47, 134, 47),
        (230, 47, 420, 47),
        (420, 47, 420, 110),
    ]
    for x1, y1, x2, y2 in arrows:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=CYAN, strokeWidth=1.6))
        if x2 >= x1:
            drawing.add(
                Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=CYAN, strokeColor=CYAN)
            )
        else:
            drawing.add(
                Polygon([x2, y2, x2 + 5, y2 + 3, x2 + 5, y2 - 3], fillColor=CYAN, strokeColor=CYAN)
            )
    drawing.add(
        String(
            238,
            5,
            "A failed gate or smoke test never advances the candidate",
            fontName="DejaVu",
            fontSize=7.5,
            textAnchor="middle",
            fillColor=MUTED,
        )
    )
    return drawing


def request_drawing() -> Drawing:
    drawing = Drawing(475, 125)
    labels = ["JSON", "Pydantic", "Timezone", "Features", "Model", "Clip + reply"]
    for i, label in enumerate(labels):
        x = 4 + i * 79
        drawing.add(Rect(x, 55, 65, 31, rx=4, ry=4, fillColor=PALE, strokeColor=BLUE))
        drawing.add(
            String(
                x + 32.5,
                67,
                label,
                fontName="DejaVu-Bold",
                fontSize=7.2,
                textAnchor="middle",
                fillColor=NAVY,
            )
        )
        if i < len(labels) - 1:
            drawing.add(Line(x + 65, 70.5, x + 79, 70.5, strokeColor=CYAN, strokeWidth=1.4))
    drawing.add(
        String(
            237,
            102,
            "Online inference path",
            fontName="DejaVu-Bold",
            fontSize=11,
            textAnchor="middle",
            fillColor=NAVY,
        )
    )
    drawing.add(
        String(
            237,
            26,
            "The fitted Pipeline owns feature transformation and prediction",
            fontName="DejaVu",
            fontSize=8,
            textAnchor="middle",
            fillColor=MUTED,
        )
    )
    return drawing


def metric_formulae(story: list[Flowable]) -> None:
    table(
        story,
        [
            ["Metric", "Definition", "Interpretation"],
            ["MAE", "mean(|y - y_hat|)", "Typical absolute error in minutes; directly readable."],
            ["RMSE", "sqrt(mean((y - y_hat)^2))", "Penalizes large misses more strongly than MAE."],
            [
                "R2",
                "1 - SSE/SST",
                "Variance explained relative to a mean predictor; can be negative.",
            ],
            [
                "p95 AE",
                "95th percentile of |error|",
                "Tail reliability: 95% of errors fall below this value.",
            ],
            [
                "Bias",
                "mean(y_hat - y)",
                "Positive means systematic overprediction; negative underprediction.",
            ],
            [
                "PSI",
                "sum((a-e) ln(a/e))",
                "Distribution shift heuristic, not a causal failure diagnosis.",
            ],
        ],
        [30 * mm, 49 * mm, 92 * mm],
    )


def add_cover(story: list[Flowable]) -> None:
    background = Table(
        [
            [
                Paragraph("NYC TAXI DURATION", STYLES["Title"]),
            ]
        ],
        colWidths=[174 * mm],
        rowHeights=[48 * mm],
    )
    background.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10 * mm),
            ]
        )
    )
    story.extend([Spacer(1, 16 * mm), background, Spacer(1, 9 * mm)])
    p(story, "Production ML Engineering Guide", "H1")
    p(
        story,
        "Basit bir regression modelinden güvenilir, izlenebilir ve geri alınabilir bir production sistemine",
        "CoverSubtitle",
    )
    story.append(Spacer(1, 18 * mm))
    table(
        story,
        [
            ["Kapsam", "Uçtan uca"],
            ["Problem", "NYC yellow taxi yolculuk süresi tahmini"],
            ["Ana vurgu", "Data contract, CI/CD, deployment, monitoring, retraining, rollback"],
            ["Model", "Median baseline + HistGradientBoostingRegressor"],
            ["Yazar", "Ugur Alkan"],
            ["Sürüm", "1.2.3 | September 2026"],
        ],
        [38 * mm, 130 * mm],
    )
    story.append(Spacer(1, 16 * mm))
    p(
        story,
        "Bu doküman repodaki gerçek kodla birebir eşleşir. Sentetik doğrulama metrikleri yalnızca sistem smoke testi içindir; gerçek TLC model performansı değildir.",
        "Callout",
    )
    story.append(PageBreak())


def add_front_matter(story: list[Flowable]) -> None:
    h(story, "Bu materyal nasıl çalışılmalı", 1)
    p(
        story,
        "Belge dört katmanlı okunabilir. Önce problem ve leakage sınırını anlayın. Sonra offline training akışını ve release gate mantığını izleyin. Ardından online API, container ve deployment katmanlarına geçin. Son bölümdeki tam kaynak kod ekini açık bir editörle birlikte okuyun ve laboratuvar komutlarını çalıştırın.",
    )
    bullets(
        story,
        [
            "Birinci geçiş: Bölüm 1-5 ile sistemin neden böyle tasarlandığını öğrenin.",
            "İkinci geçiş: Bölüm 6-12 ile kodun bileşenlerini ve çalışma sırasını izleyin.",
            "Üçüncü geçiş: Laboratuvarları çalıştırın, bilerek gate kırın ve rollback senaryosunu inceleyin.",
            "Referans kullanımı: Dosya kataloğu ve kaynak kod eki üzerinden belirli bir modüle dönün.",
        ],
    )
    p(
        story,
        "Önemli ayrım: production-ready bir repository ile şirketinizin production platformuna deploy edilmiş sistem aynı şey değildir. Bu repo uygulama kodunu ve platform sözleşmesini sağlar; gerçek cluster kimliği, TLS, secret manager, kalıcı telemetry, SLO alert'leri ve yönetişim organizasyona bağlanmalıdır.",
        "Callout",
    )
    h(story, "İçindekiler", 1)
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC0", fontName="DejaVu-Bold", fontSize=9, leading=14, leftIndent=0, textColor=NAVY
        ),
        ParagraphStyle(
            "TOC1", fontName="DejaVu", fontSize=8, leading=12, leftIndent=12, textColor=INK
        ),
        ParagraphStyle(
            "TOC2", fontName="DejaVu", fontSize=7.5, leading=11, leftIndent=24, textColor=MUTED
        ),
    ]
    story.extend([toc, PageBreak()])


def add_core_chapters(story: list[Flowable]) -> None:
    h(story, "1. Problem tanımı ve başarı ölçütü")
    p(
        story,
        "İş problemi şudur: Bir yolcu pickup zamanını, pickup taxi zone'unu, gitmek istediği drop-off zone'unu ve yolcu sayısını verdiğinde, yolculuk başlamadan önce beklenen süreyi dakika cinsinden tahmin etmek. Tahmin bir point estimate'tir; kullanıcıya belirsizlik aralığı sunulmaz. Bu, sürüm 1'in bilinçli sınırıdır.",
    )
    p(
        story,
        "Matematiksel olarak model f(X) = y_hat kurar. X yalnızca karar anında bilinen değişkenlerden oluşur. Ground truth y, drop-off tamamlandıktan sonra pickup ve drop-off timestamp farkından elde edilir. Böylece target gelecekte oluşur fakat feature set geleceğe bakmaz.",
    )
    code(
        story,
        "duration_minutes = (tpep_dropoff_datetime - tpep_pickup_datetime).total_seconds() / 60",
    )
    h(story, "1.1 Neden bu bir production problemi?", 2)
    p(
        story,
        "TLC verisi aylık yayımlanır, şema değişebilir, gecikmeli label üretir ve mevsimsel/operasyonel drift taşır. Dolayısıyla ingestion, veri kalitesi, yeniden eğitim ve promotion doğal bir lifecycle oluşturur. NYC TLC veri sayfası trip kayıtlarının aylık ve tipik olarak yaklaşık iki aylık gecikmeyle yayımlandığını, Parquet formatının kullanıldığını ve küçük şema değişiklikleri olabileceğini açıkça belirtir.",
    )
    h(story, "1.2 Kabul kriterleri", 2)
    table(
        story,
        [
            ["Boyut", "Kabul koşulu"],
            ["Fonksiyonel", "Geçerli request için dakika tahmini ve model/contract sürümü döner."],
            ["Veri", "Eksik kolon veya tümü geçersiz batch deterministik biçimde hata verir."],
            [
                "ML",
                "Candidate, median baseline'ı ayarlı marjla geçer ve tüm absolute threshold'ları sağlar.",
            ],
            [
                "Operasyon",
                "Liveness, readiness, structured logs ve Prometheus metric'leri çalışır.",
            ],
            ["Release", "Aynı image digest staging smoke testinden production'a promote edilir."],
            ["Recovery", "Başarısız production rollout önceki ReplicaSet'e geri alınır."],
        ],
        [35 * mm, 135 * mm],
    )

    h(story, "2. Dataset, source contract ve leakage")
    p(
        story,
        "Kaynak resmi NYC TLC yellow taxi trip record Parquet dosyalarıdır. TLC; pickup/drop-off zamanları ve lokasyonları, trip distance, itemized fare, rate, payment type ve driver-reported passenger count gibi alanlar bulunduğunu, kaydın teknoloji sağlayıcılarından geldiğini ve doğruluğu garanti etmediğini söyler. Bu uyarı veri kalitesi tasarımının neden zorunlu olduğunu açıklar.",
    )
    h(story, "2.1 Feature availability matrix", 2)
    table(
        story,
        [
            ["Alan", "Feature?", "Gerekçe"],
            ["tpep_pickup_datetime", "Evet", "Request anında bilinir; saat ve weekday türetilir."],
            ["PULocationID", "Evet", "Pickup zone request anında bilinir."],
            ["DOLocationID", "Evet", "Kullanıcının hedefi olarak bilinir."],
            ["passenger_count", "Evet", "Pickup sırasında bilinir; 1-6 contract aralığı."],
            ["tpep_dropoff_datetime", "Hayır", "Sadece label üretir; gelecekte oluşur."],
            ["trip_distance", "Hayır", "Gerçekleşmiş taximeter mesafesi trip sonunda kesinleşir."],
            ["fare/tip/tolls/total", "Hayır", "Post-trip ve target-adjacent leakage."],
            ["payment_type", "Hayır", "Çoğu senaryoda trip sonunda kesinleşir."],
        ],
        [45 * mm, 25 * mm, 100 * mm],
    )
    p(
        story,
        "Bu tablo modelden önce gelir. Feature importance'a bakıp sızıntı keşfetmek geç bir kontroldür; asıl kontrol decision-time data contract'tır. `INFERENCE_COLUMNS` allowlist'i bu kararı kod içinde enforce eder.",
        "Callout",
    )
    h(story, "2.2 Quality filtreleri", 2)
    bullets(
        story,
        [
            "Beş zorunlu kaynak kolonu yoksa batch hemen reddedilir.",
            "Timestamp parse edilemeyen satırlar elenir.",
            "Duration 1-120 dakika dışında ise outlier/bozuk kayıt kabul edilmez.",
            "Passenger count 1-6 ve routable zone ID 1-263 arasında olmalıdır.",
            "Pickup source dosyasının ayı dışında ise satır elenir ve ayrıca sayılır.",
            "Beş kaynak kolonu tamamen aynı olan duplicate satırların ilki tutulur.",
            "Temiz veri pickup time'a göre sıralanır; boş kalan batch hata verir.",
            "Rejection, out-of-period, duplicate oranları ve valid volume gate edilir.",
        ],
    )

    h(story, "3. Sistem mimarisi")
    story.append(pipeline_drawing())
    p(
        story,
        "Sistem iki ana döngü taşır. Offline döngü source data'yı model artifact'ına dönüştürür. Online döngü artifact'ı request başına çalıştırır ve telemetry üretir. Monitoring bu iki döngüyü bağlar: labelsiz input/prediction drift erken sinyal, gecikmeli gerçek duration ise performans sinyalidir.",
    )
    h(story, "3.1 Boundary'ler ve sorumluluklar", 2)
    table(
        story,
        [
            ["Boundary", "Girdi", "Çıktı", "Sahip olduğu garanti"],
            [
                "Ingestion",
                "year/month",
                "immutable Parquet + SHA",
                "atomik download, tekrar kullanılabilir checksum",
            ],
            ["Validation", "raw DataFrame", "clean DataFrame + report", "schema/domain kuralları"],
            ["Training", "clean temporal data", "fitted Pipeline", "train/serve transform birliği"],
            ["Gate", "metrics + thresholds", "pass/fail + reasons", "kötü candidate ilerlemez"],
            ["Artifact", "model + lineage", "joblib + JSON", "kod/veri/config izlenebilirliği"],
            ["Serving", "strict JSON", "versioned prediction", "timezone, bounds, availability"],
            ["Deployment", "image digest", "running ReplicaSet", "same-bit promotion + rollback"],
        ],
        [29 * mm, 38 * mm, 40 * mm, 63 * mm],
    )

    h(story, "4. Repository yapısı ve dosya kataloğu")
    p(
        story,
        "Notebook merkezli tasarım yoktur. Eğitim, monitoring ve serving import edilebilir modüller ve CLI entrypoint'leri olarak tanımlıdır. Notebook ancak keşif için eklenebilir; release path notebook çalıştırmaz.",
    )
    table(
        story,
        [
            ["Konum", "Görev"],
            ["configs/", "Base değerler ve development/staging/production override'ları."],
            ["src/taxi_duration/data", "HTTP ingestion, checksum, schema ve domain validation."],
            ["src/taxi_duration/features", "Tek leakage-safe feature transformer."],
            [
                "src/taxi_duration/models",
                "Estimator factory, artifact metadata ve güvenilir load/save.",
            ],
            ["src/taxi_duration/training", "Chronological split, metrics, gates ve orchestration."],
            [
                "src/taxi_duration/serving",
                "Pydantic contract, FastAPI lifecycle, metrics ve endpoints.",
            ],
            ["src/taxi_duration/monitoring", "Reference profile ve PSI drift hesapları."],
            ["tests/", "Unit, contract ve component integration testleri."],
            [".github/workflows", "Code CI ve gated retrain-to-production workflow."],
            ["k8s/", "Deployment/service base ve environment overlay'leri."],
            ["monitoring/", "Prometheus scrape ve Grafana provisioning."],
            ["docs/", "Architecture decisions, model card ve runbook."],
        ],
        [52 * mm, 118 * mm],
    )

    h(story, "5. Configuration sistemi")
    p(
        story,
        "YAML config dosyaları `AppConfig` ve alt Pydantic modelleriyle `extra=forbid` şeklinde doğrulanır. Yanlış yazılmış bir key sessizce ignore edilmez. Development ve production dosyaları `extends: base.yaml` ile yalnızca farkları belirtir. `_deep_merge` nested dictionary'leri korur; scalar veya liste override'ı doğrudan değiştirir.",
    )
    table(
        story,
        [
            ["Config grubu", "Kontrol ettiği davranış"],
            ["data", "dizinler, URL, sınırlar, rolling window ve source lookback"],
            ["split", "train ve validation fraction; kalan otomatik test"],
            ["model", "iteration, learning rate, leaves, L2, leaf minimum"],
            ["gates", "quality, MAE, tail, baseline, stability ve champion confidence"],
            ["serving", "bind adresi, port ve ortak output clip bounds"],
            ["monitoring", "numeric/categorical/prediction PSI ve unseen-route sınırı"],
        ],
        [44 * mm, 126 * mm],
    )
    p(
        story,
        "Production threshold'ları veri gördükten sonra keyfi biçimde ayarlanmamalıdır. Önceden tanımlı risk toleransından türetilmeli, code review ile değişmeli ve historical backtest ile kalibre edilmelidir.",
        "Callout",
    )

    h(story, "6. Ingestion ve veri doğrulama")
    h(story, "6.1 Publication-aware dönem çözümleme", 2)
    p(
        story,
        "Takvimde iki ay geriye gitmek dosyanın yayımlandığını kanıtlamaz. `resolve-period` önce bu adayı HEAD ile yoklar, bulunmazsa `source_lookback_months` sınırında ay ay geriye gider. Bulduğu son ayı üç aylık rolling window'un sonu yapar. Hiçbir nesne doğrulanamazsa workflow fail closed davranır; yanlış aya boş/404 download denemez. Yalnızca 200/206 'yayımlanmış', yalnızca 403/404 'yayımlanmamış' sayılır; ağ hatası veya beklenmeyen bir status üç denemeden sonra hata verir, böylece geçici bir kesinti eğitim penceresini sessizce eski bir aya kaydıramaz.",
    )
    code(
        story,
        "taxi-ml resolve-period --config-path configs/staging.yaml\n# year=2026\n# month=07\n# period=2026-07\n# window=2026-05,2026-06,2026-07",
    )
    h(story, "6.2 Atomik download", 2)
    p(
        story,
        "`download_month` doğrudan final dosyaya yazmaz. Önce `.part` uzantılı geçici dosyaya stream eder, HTTP status'ü doğrular ve başarılı tamamlanınca `replace` ile atomik olarak final isme taşır. Exception olursa partial dosya silinir. Mevcut final dosya overwrite istenmedikçe yeniden indirilmez.",
    )
    code(
        story,
        "with httpx.stream('GET', url, follow_redirects=True, timeout=120.0) as response:\n    response.raise_for_status()\n    with temporary.open('wb') as output:\n        for chunk in response.iter_bytes(chunk_size=1024 * 1024):\n            output.write(chunk)\ntemporary.replace(destination)",
    )
    h(story, "6.3 SHA-256 lineage", 2)
    p(
        story,
        "Her rolling-window dosyasının checksum'u metrics manifestine yazılır; sıralı manifestin hash'i model metadata'ya bağlanır. Böylece üç aylık training input tek bir lineage kimliğiyle izlenirken dosya bazında kanıt da korunur.",
    )
    h(story, "6.4 Validation ve rolling merge sırası", 2)
    bullets(
        story,
        [
            "Önce kolon varlığı doğrulanır; aksi durumda downstream KeyError yerine açıklayıcı contract error üretilir.",
            "Sonra timestamp coercion ve target construction yapılır.",
            "Her dosya kendi source ayına göre doğrulanır; cross-month stray satırlar elenir.",
            "Exact duplicate mask ve bütün domain koşulları birlikte uygulanır.",
            "Geçerli aylık frame'ler birleştirilir, pickup time'a göre stable sort edilir.",
            "Rapor oranı batch tamamen boş olsa dahi division-by-zero üretmez.",
        ],
    )

    h(story, "7. Feature engineering ve training-serving skew")
    p(
        story,
        "`TripFeatureBuilder` scikit-learn estimator contract'ını uygular. `fit` öğrenilecek istatistik olmadığı için yalnızca input feature isimlerini kaydeder. `transform` allowlist kolonlarını doğrular, timestamp'i parse eder ve DataFrame üretir. Transformer fitted model Pipeline'ının içindedir; API bağımsız bir feature script çalıştırmaz.",
    )
    table(
        story,
        [
            ["Feature", "Formül/mantık", "Neden"],
            ["route_target", "TargetEncoder(PU*1000+DO)", "Route log-süre ortalaması (simetrikse ≈ median)"],
            ["pickup_zone_target", "TargetEncoder(PULocationID)", "Sparse route fallback"],
            ["dropoff_zone_target", "TargetEncoder(DOLocationID)", "Hedef zone etkisi"],
            ["passenger_count", "numeric cast", "Basit kapasite/operasyon etkisi"],
            ["hour_sin/cos", "sin/cos(2*pi*hour/24)", "23:59 ile 00:01 yakınlığını korur"],
            ["weekday_sin/cos", "sin/cos(2*pi*weekday/7)", "Haftanın çevrimsel yapısı"],
            ["is_weekend", "weekday >= 5", "Rejim ayrımı"],
            ["is_rush_hour", "07-10 veya 16-20", "Basit trafik rejimi proxy'si"],
        ],
        [40 * mm, 62 * mm, 68 * mm],
    )
    p(
        story,
        "TLC zone ID sırası coğrafi değildir. v1.2 route, pickup ve drop-off kategorilerini cross-fitted TargetEncoder ile kodlar: training satırı kendi label'ını görmez, rare/unseen route global/zone ortalamasına shrink olur. `ordinal` yalnızca karşılaştırma koludur.",
        "Callout",
    )

    h(story, "8. Model, target dönüşümü ve baselines")
    h(story, "8.1 Median baseline", 2)
    p(
        story,
        "DummyRegressor(strategy='median') zorunlu referanstır. Candidate'ın kabul edilmesi yalnızca absolute MAE threshold'una bağlı değildir; median tahmine göre yeterli relative improvement da gerekir. Baseline olmadan 8 dakikalık MAE'nin anlamlı mı zayıf mı olduğu anlaşılamaz.",
    )
    h(story, "8.2 HistGradientBoostingRegressor", 2)
    p(
        story,
        "Histogram-based gradient boosting, tabular nonlinear ilişkileri düşük operasyonel karmaşıklıkla öğrenir. Loss `absolute_error` modelin conditional median'a yaklaşmasını sağlar. Built-in random early stopping kapalıdır; staged prediction eğrisi kronolojik validation parçasında değerlendirilir ve en iyi iteration yeniden fit edilir.",
    )
    h(story, "8.3 Log target", 2)
    p(
        story,
        "`TransformedTargetRegressor` training sırasında y' = log(1+y) uygular, prediction'ı exp(y')-1 ile dakika ölçeğine döndürür. Sağ kuyruk sıkıştırılır. `check_inverse=False`, sayısal inverse kontrol overhead'ini kaldırır; iki fonksiyon analitik olarak eşleştirilmiştir.",
    )
    code(
        story,
        "Pipeline([\n    ('features', TripFeatureBuilder()),\n    ('regressor', TransformedTargetRegressor(\n        regressor=HistGradientBoostingRegressor(loss='absolute_error', ...),\n        func=np.log1p, inverse_func=np.expm1, check_inverse=False,\n    )),\n])",
    )

    h(story, "9. Temporal split, evaluation ve release gates")
    p(
        story,
        "Üç ayrı aylık dosya önce kendi ay contract'ıyla temizlenir ve birleşir. Pickup zamanına göre ilk %70 train, sonraki %15 validation, son %15 test yapılır. Böylece final tail daha sonraki zamanı temsil eder. `min_valid_rows` küçük veya eksik batch'in metrik şansıyla promote edilmesini engeller.",
    )
    metric_formulae(story)
    h(story, "9.1 Gate mantığı", 2)
    bullets(
        story,
        [
            "Validation MAE ayarlı maximum'u aşmamalı.",
            "Final test MAE ayrı maximum'u aşmamalı.",
            "Test p95 absolute error tail-risk sınırında kalmalı.",
            "Validation MAE / baseline MAE - 1, gerekli relative improvement'tan kötü olmamalı.",
            "Validation ve test MAE farkı stability sınırını aşmamalı.",
            "Rejection, out-of-period ve duplicate oranları sınırı aşmamalı; valid rows minimum'u sağlamalı.",
            "Candidate champion'a göre non-inferior olmalı: paired absolute-error farkının saat-cluster'lı tek taraflı upper bound'u pozitif marjı (champion MAE'nin %1'i) aşmamalı.",
        ],
    )
    p(
        story,
        "Gate fail olduğunda metrics JSON yine yazılır, fakat model artifact'ı paketlenmez ve command non-zero exit verir. Bu sıra tanılama kanıtını korur, zayıf candidate'ın deploy edilmesini engeller.",
    )
    h(story, "9.2 Champion confidence karşılaştırması", 2)
    p(
        story,
        "Candidate ve production champion aynı test satırlarında skorlanır. Her satır için d = |y-candidate| - |y-champion| alınır (pozitif = candidate daha kötü). Karar kuralı non-inferiority'dir: d ortalaması + z·SE, m = champion MAE × max_regression_vs_champion marjını aşarsa promotion bloklanır. SE, aynı saatteki yolculuklar aynı trafiği paylaştığı için pickup saati üzerinden cluster-robust hesaplanır. Marj sıfır olsaydı kural bir üstünlük testine dönüşürdü: champion kadar iyi bir retrain yaklaşık %95 olasılıkla bloklanırdı. Bu yüzden config marjın pozitif olmasını zorunlu tutar (ADR 0007).",
    )
    h(story, "9.3 Doğrulanmış sentetik smoke sonucu", 2)
    table(
        story,
        [
            ["Kanıt", "Sonuç"],
            ["Rolling input", "3 x 1,200 = 3,600 geçerli sentetik satır"],
            ["Split", "2,520 train / 540 validation / 540 test"],
            ["Validation MAE", "8.07 dakika"],
            ["Test MAE", "7.38 dakika; tüm development gate'leri geçti"],
            ["Sonraki ay MAE", "9.03 dakika (1,200 satır)"],
            ["Drift", "Yeni ayda unseen route %26.67; critical alarm üretildi"],
        ],
        [50 * mm, 120 * mm],
    )
    p(
        story,
        "Bu sayılar plumbing, gate ve alarm davranışını doğrular; gerçek TLC performansı değildir. Gerçek threshold kalibrasyonu `docs/benchmark-protocol.md` prosedürüyle ayrıca yapılmalıdır.",
        "Callout",
    )

    h(story, "10. Artifact, metadata ve reproducibility")
    table(
        story,
        [
            ["Artifact", "İçerik", "Amaç"],
            [
                "model.joblib",
                "Feature transformer + fitted regressor",
                "Tek train/serve execution object",
            ],
            [
                "metadata.json",
                "version, UTC, Git SHA, hashes, metrics, contract",
                "Lineage ve API görünürlüğü",
            ],
            [
                "metrics.json",
                "quality, rows, periods, baseline, candidate, gate",
                "Review ve audit evidence",
            ],
            [
                "reference_profile.json",
                "numeric/categorical/prediction dağılımları",
                "Drift reference",
            ],
        ],
        [36 * mm, 65 * mm, 69 * mm],
    )
    p(
        story,
        "Joblib pickle temellidir ve untrusted input için güvenli değildir. Model yalnızca CI/retraining tarafından üretilen trusted artifact'tan yüklenmelidir. Daha sıkı ortamlarda imza/attestation, artifact registry access policy ve compatibility manifest eklenmelidir.",
        "Callout",
    )
    p(
        story,
        "Config SHA, `extends` sonrası tamamen çözülmüş canonical JSON'dan üretilir; base.yaml değişikliği override dosyası aynı kalsa da hash'i değiştirir. Runtime/dev lock'ları transitive package ve wheel hash'lerini taşır. Loader Python minor, scikit-learn, NumPy, pandas ve joblib uyumsuzluğunu reddeder.",
    )

    h(story, "11. FastAPI serving katmanı")
    story.append(request_drawing())
    h(story, "11.1 Lifespan ve readiness", 2)
    p(
        story,
        "App startup'ta config, model ve metadata'yı bir kez yükler. Başarırsa `ready=True`; artifact yoksa process yine yaşayabilir fakat readiness 503 döner. Bu ayrım Kubernetes'in broken pod'a trafik göndermemesini, buna rağmen process diagnostic endpoint'inin erişilebilir kalmasını sağlar.",
    )
    h(story, "11.2 Request contract", 2)
    bullets(
        story,
        [
            "Unknown fields forbidden: istemeden fare_amount gibi leakage input'u kabul edilmez.",
            "Routable zone 1-263, passenger 1-6 aralığında olmalıdır.",
            "Pickup datetime UTC offset içermelidir; naive datetime reddedilir.",
            "Instant America/New_York local time'a çevrilir ve source training semantiğiyle eşlenir.",
            "Response prediction ile model_version ve feature_contract_version döndürür.",
            "Prediction, training/evaluation ile aynı `predict_duration` bounded helper'ından geçer.",
        ],
    )
    h(story, "11.3 Endpoint'ler", 2)
    table(
        story,
        [
            ["Endpoint", "Amaç", "Failure semantics"],
            ["GET /health/live", "Process yaşıyor mu?", "Process/server failure"],
            [
                "GET /health/ready",
                "Model servis vermeye hazır mı?",
                "Artifact/config load sorunu -> 503",
            ],
            [
                "POST /v1/predict",
                "Tek trip duration tahmini",
                "422 contract, 503 unavailable, 500 unexpected",
            ],
            ["GET /metrics", "Prometheus exposition", "Platform scrape alarmı"],
            ["GET /docs", "OpenAPI UI", "Production policy ile kapatılabilir"],
        ],
        [42 * mm, 65 * mm, 63 * mm],
    )
    h(story, "11.4 Observability", 2)
    p(
        story,
        "Middleware her request'e caller-supplied veya UUID request ID bağlar. JSON logs path, latency ve prediction context içerir. Prometheus Counter status bazlı throughput/error, Histogram latency ve prediction distribution sağlar. Ham request persistence yoktur; gerçek sistemde prediction-label join için privacy-reviewed event ID gereklidir.",
    )

    h(story, "12. Container ve local observability stack")
    p(
        story,
        "Multi-stage Docker build önce wheel üretir, sonra minimal runtime image'a yalnızca package, configs ve gate'i geçmiş artifacts kopyalar. Runtime non-root `appuser` kullanır. Compose API root filesystem'ini read-only, `/tmp`'yi tmpfs ve artifact mount'u read-only yapar; privilege escalation kapalıdır.",
    )
    table(
        story,
        [
            ["Servis", "Port", "Rol"],
            ["api", "8000", "Prediction, health ve Prometheus exposition"],
            ["prometheus", "9090", "15 saniyede API metric scrape"],
            ["grafana", "3000", "Provisioned request-rate ve p95 latency dashboard"],
        ],
        [45 * mm, 25 * mm, 100 * mm],
    )
    p(
        story,
        "Docker executable bu çalışma ortamında bulunmadığı için image runtime build'i burada çalıştırılamadı. Dockerfile, Compose ve config syntax statik olarak doğrulandı; CI workflow gerçek Docker build ve Trivy scan'i zorunlu kılar.",
        "Callout",
    )

    h(story, "13. CI: code quality ve supply-chain kontrolleri")
    p(
        story,
        "Pull request ve main push iki job çalıştırır. Quality job format, lint, strict type check, tests/coverage, Bandit ve dependency audit uygular. Runtime, dev ve docs lock'ları ayrı, transitively pinned ve `--require-hashes` ile kurulur. Container job ancak quality sonrası sentetik smoke artifact üretir, digest-pinned Python tabanından image build eder ve high/critical vulnerability için Trivy çalıştırır. Bütün third-party Actions full commit SHA'ya pinlidir. Bu çalışma rehberi ayrıca `make pdf` ile üretilir; PDF build veya byte comparison application CI gate'inin parçası değildir.",
    )
    table(
        story,
        [
            ["Kontrol", "Yakaladığı sınıf", "Neyi kanıtlamaz"],
            ["Ruff", "style, import, common bug patterns", "Business correctness"],
            ["mypy strict", "type contract violations", "Runtime data validity"],
            ["pytest + coverage", "asserted behavior regressions", "Test edilmemiş behavior"],
            ["Bandit", "yaygın Python security smells", "Tam threat model"],
            ["pip-audit", "bilinen dependency CVE'leri", "Zero-day veya app logic"],
            ["Docker build", "image'ın üretilebilirliği", "Production load capacity"],
            ["Trivy", "image package vulnerabilities", "Runtime authorization"],
        ],
        [34 * mm, 65 * mm, 71 * mm],
    )

    h(story, "14. CD: gated retraining, staging, promotion, rollback")
    p(
        story,
        "Workflow manual veya monthly schedule ile başlar. Manual run yıl/ay alır; schedule conservative lag adayını gerçek source availability ile doğrular ve gerekirse geriye yürür. Üç aylık window indirilir, her ay doğrulanır, birleşik temporal split train/gate edilir. Artifacts evidence olarak saklanır ve yalnızca başarıdan sonra model dahil image oluşturulur. Image GHCR'a provenance ve SBOM ile push edilir.",
    )
    bullets(
        story,
        [
            "Staging protected environment exact image digest'i deploy eder.",
            "Rollout tamamlanana kadar beklenir; readiness başarısızsa ilerlemez.",
            "Smoke test önce readiness sonra gerçek prediction contract'ını çağırır.",
            "Production environment approval/policy staging başarılı olmadan açılmaz.",
            "Production aynı digest'i kullanır; rebuild yapılmaz.",
            "Champion pull/extraction hatası release'i durdurur; karşılaştırmayı atlamak explicit `skip_champion` ister.",
            "Kustomize 5.6.0 commit-pinned setup action ile kurulur; runner state'ine güvenilmez.",
            "Rollback yalnız production apply başarılı olduktan sonraki bir step hata verirse çalışır.",
        ],
    )
    p(
        story,
        "Repo cluster credential saklamaz. OIDC veya platformun scoped deployment integration'ı ayrıca bağlanmalıdır. Environment name kullanmak tek başına approval sağlamaz; GitHub repository settings içinde reviewers ve protection rules tanımlanmalıdır.",
        "Callout",
    )

    h(story, "15. Kubernetes deployment modeli")
    p(
        story,
        "Base Deployment iki replica, non-root pod security context, CPU/memory request-limit, readiness ve liveness probe, no privilege escalation, read-only root filesystem ve dropped capabilities içerir. Service port 80'i container 8000'e yönlendirir. Kustomize overlay staging ve production namespace/tag farkını taşır; production replica sayısını üçe çıkarır.",
    )
    h(story, "15.1 Eksik platform bileşenleri", 2)
    bullets(
        story,
        [
            "Ingress, TLS certificate ve authentication/authorization",
            "NetworkPolicy ve egress allowlist",
            "HorizontalPodAutoscaler ve disruption budget",
            "Secret manager ve workload identity",
            "Central log/metric retention, alert routing ve SLO dashboards",
            "Canary/blue-green controller ve automated analysis",
            "Model registry approval UI ve artifact signature verification",
        ],
    )
    p(
        story,
        "Bunların repo dışı olduğu gizlenmemiştir. Gerçek platform seçilmeden vendor-specific YAML uydurmak yerine açık integration contract bırakılmıştır.",
    )

    h(story, "16. Monitoring: operasyon, drift ve delayed labels")
    h(story, "16.1 Üç monitoring katmanı", 2)
    table(
        story,
        [
            ["Katman", "Ne zaman?", "Sinyaller", "Aksiyon"],
            [
                "Operational",
                "Anında",
                "availability, 5xx, p95 latency, saturation",
                "traffic/infra incident",
            ],
            [
                "Data/prediction",
                "Labelsız erken",
                "numeric/categorical PSI, unseen route, prediction shift",
                "investigate, block auto-promotion",
            ],
            [
                "Performance",
                "Drop-off label sonrası",
                "MAE, p95 AE, bias, slice metrics",
                "retrain/challenger/rollback",
            ],
        ],
        [31 * mm, 30 * mm, 66 * mm, 43 * mm],
    )
    h(story, "16.2 PSI", 2)
    p(
        story,
        "Reference profile training split'ten numeric histogramlar, pickup/drop-off/route category dağılımları, known-route seti ve bounded prediction histogramı üretir. Current batch aynı edge/category bucket'larıyla karşılaştırılır. PSI 0.10/0.25 bandları ve unseen-route limiti config'dedir; bunlar evrensel gerçek değil, kalibrasyon başlangıcıdır.",
    )
    h(story, "16.3 Neden prediction drift yetmez?", 2)
    p(
        story,
        "Prediction dağılımı input mix nedeniyle değişebilir ve model hala doğru olabilir. Tersine dağılım sabit görünürken conditional error artabilir. Bu nedenle gecikmeli labels geldiğinde overall ve hour/weekday/route-frequency/zone slice bazında performans hesaplanmalıdır.",
    )

    h(story, "17. Test stratejisi")
    table(
        story,
        [
            ["Test", "Seviye", "Koruduğu davranış"],
            ["test_config", "Unit", "Inheritance ve validated settings"],
            ["test_validation", "Unit", "Duration, schema ve exact duplicate policy"],
            [
                "test_features",
                "Unit",
                "Finite outputs, column order, leakage exclusion, cyclical encoding",
            ],
            ["test_gates", "Unit", "Good candidate pass ve açıklayıcı multi-failure"],
            ["test_ingest/period", "Unit", "Availability fallback ve rolling window"],
            ["test_monitoring/drift", "Unit", "PSI, invalid rate ve category behavior"],
            ["test_api_schema", "Contract", "Timezone kabulü ve unknown field rejection"],
            [
                "test_training_pipeline",
                "Integration",
                "Single/rolling Parquet'ten artifact, lineage ve metrics",
            ],
            ["test_api", "Integration", "Fitted model startup, prediction ve readiness"],
        ],
        [47 * mm, 25 * mm, 98 * mm],
    )
    p(
        story,
        "Doğrulama komutları: `make lint`, `make type`, `make test`, `make security`. "
        "Güncel test sayısı ve coverage bu komutların çıktısından okunur; rehber bu değerleri sabit yazmaz.",
    )

    h(story, "18. Local çalışma laboratuvarı")
    h(story, "18.1 Kurulum", 2)
    code(
        story,
        "python -m venv .venv\n. .venv/bin/activate\npython -m pip install --upgrade pip\npip install --require-hashes -r requirements-dev.lock\npip install --no-deps -e .",
    )
    h(story, "18.2 Sentetik smoke data", 2)
    code(
        story,
        "python scripts/generate_sample_data.py --rows 5000 --start 2025-01-01 --output data/raw/yellow_tripdata_2025-01.parquet\npython scripts/generate_sample_data.py --rows 5000 --start 2025-02-01 --output data/raw/yellow_tripdata_2025-02.parquet\npython scripts/generate_sample_data.py --rows 5000 --start 2025-03-01 --output data/raw/yellow_tripdata_2025-03.parquet\ntaxi-ml train data/raw/yellow_tripdata_2025-0{1,2,3}.parquet --config-path configs/development.yaml --model-version local-v1.2",
    )
    p(
        story,
        "Sentetik generator gerçek performans değerlendirmesi değildir. Ama schema, label construction, split, transformer, model, gates ve artifact I/O'nun tamamını hızlı ve deterministik çalıştırır.",
    )
    h(story, "18.3 API", 2)
    code(
        story,
        'uvicorn taxi_duration.serving.app:app --reload --port 8000\n\ncurl -s http://localhost:8000/health/ready\ncurl -s -X POST http://localhost:8000/v1/predict \\\n  -H \'Content-Type: application/json\' \\\n  -d \'{"pickup_datetime":"2025-01-15T08:30:00-05:00","pickup_location_id":132,"dropoff_location_id":230,"passenger_count":1}\'',
    )
    h(story, "18.4 Resmi data", 2)
    code(
        story,
        "taxi-ml download 2025 1 && taxi-ml download 2025 2 && taxi-ml download 2025 3\ntaxi-ml train data/raw/yellow_tripdata_2025-01.parquet \\\n  data/raw/yellow_tripdata_2025-02.parquet \\\n  data/raw/yellow_tripdata_2025-03.parquet \\\n  --config-path configs/development.yaml --model-version 2025-q1-v1.2",
    )
    h(story, "18.5 Drift", 2)
    code(
        story,
        "taxi-ml drift data/raw/current_month.parquet \\\n  --model-path artifacts/model.joblib \\\n  --metadata-path artifacts/metadata.json \\\n  --reference-path artifacts/reference_profile.json \\\n  --output-path reports/monitoring/drift.json --fail-on-critical",
    )

    h(story, "19. Failure injection çalışmaları")
    table(
        story,
        [
            ["Deney", "Nasıl", "Beklenen"],
            ["Leakage guard", "API JSON'a fare_amount ekle", "422: extra field forbidden"],
            ["Timezone guard", "Offset'siz datetime gönder", "422: timezone required"],
            ["Readiness", "Artifact path'i yanlış yap", "live 200, ready 503"],
            ["Data schema", "PULocationID kolonunu kaldır", "Training başlamadan ValueError"],
            [
                "Gate failure",
                "max_test_mae_minutes değerini çok düşür",
                "metrics yazılır, artifact release edilmez",
            ],
            ["Drift", "Yeni PU-DO route'ları ekle", "route PSI/unseen rate artar"],
            ["Staging smoke", "Endpoint path'ini boz", "Production job başlamaz"],
            ["Production rollout", "Readiness fail image deploy et", "rollout fail ve undo"],
        ],
        [38 * mm, 65 * mm, 67 * mm],
    )

    h(story, "20. Security ve threat model")
    table(
        story,
        [
            ["Tehdit", "Mevcut kontrol", "Kalan iş"],
            [
                "Malicious input",
                "Pydantic strict schema/ranges",
                "rate limit, auth, payload limits",
            ],
            [
                "Artifact tampering",
                "trusted CI path + hashes",
                "signature verification, registry policy",
            ],
            ["Dependency CVE", "pip-audit + Trivy", "patch SLA, internal mirror"],
            ["Container escape", "non-root, read-only, drop caps", "seccomp/AppArmor, node policy"],
            ["Secret leak", "repo secret yok", "secret manager + rotation"],
            [
                "Supply-chain rebuild",
                "same digest promotion, provenance/SBOM",
                "attestation enforcement",
            ],
            ["Data privacy", "raw request persistence yok", "retention, minimization, DPIA"],
            ["Abusive automation", "protected production environment", "segregation of duties"],
        ],
        [38 * mm, 68 * mm, 64 * mm],
    )

    h(story, "21. Production'a geçmeden önce checklist")
    checklist = [
        "TLC schema ve sample month üzerinde gerçek training tamamlandı.",
        "Rejection/out-of-period/duplicate/volume gate threshold'ları gerçek TLC ile kalibre edildi.",
        "Hash-locked dependencies ve base image digest sabitlendi.",
        "Cluster authentication OIDC ile bağlandı; least privilege test edildi.",
        "Production environment reviewers/protection rules tanımlandı.",
        "Ingress, TLS, API auth, rate limiting ve network policies çalışıyor.",
        "Load test sonucu replica/resource/HPA değerleri ayarlandı.",
        "SLO, alert thresholds, on-call route ve runbook ownership belirlendi.",
        "Prediction-label event join privacy review'dan geçti.",
        "Model artifact signature/attestation doğrulaması enforce edildi.",
        "Canary veya blue-green promotion stratejisi test edildi.",
        "Rollback tatbikatı ve post-incident evidence retention doğrulandı.",
    ]
    bullets(story, [f"□ {item}" for item in checklist])

    h(story, "22. v1.2 sonrası teknik yol haritası")
    table(
        story,
        [
            ["Öncelik", "Geliştirme", "Hipotez"],
            ["P0", "Gerçek TLC batch + schema regression suite", "Source drift erken yakalanır"],
            [
                "P0",
                "Gerçek TLC benchmark + threshold table",
                "Sentetik kanıt production iddiasına dönüşmez",
            ],
            ["P1", "Taxi-zone geometry/centroid route distance", "Unseen route genellemesi artar"],
            ["P1", "Traffic/weather/events challenger", "Peak regime error düşer"],
            ["P1", "Quantile models", "ETA uncertainty kullanıcıya taşınır"],
            ["P1", "Rare-route and zone slice gates", "Aggregate metric körlüğü azalır"],
            ["P2", "Canary analysis + automatic rollback policy", "Blast radius düşer"],
            ["P2", "Feature/model registry", "Governance ve concurrent experiments gelişir"],
        ],
        [22 * mm, 75 * mm, 73 * mm],
    )

    h(story, "23. Kaynaklar")
    sources = [
        "NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
        "NYC TLC Yellow Trip Data Dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf",
        "scikit-learn HistGradientBoostingRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
        "GitHub deployment environments: https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments",
        "Kubernetes probes and pod lifecycle: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
        "Repository source code and ADRs included with this guide.",
    ]
    bullets(story, sources)


def add_source_appendix(story: list[Flowable]) -> None:
    story.append(PageBreak())
    h(story, "Ek A. Tam kaynak kod referansı")
    p(
        story,
        "Aşağıdaki listing'ler teslim edilen repository'den build sırasında doğrudan okunmuştur. Bu sayede açıklama ile kod arasında kopyalama kaynaklı sürüm farkı oluşmaz. Binary artifacts, cache dosyaları ve üretilmiş coverage çıktıları eklenmemiştir.",
    )
    patterns = [
        "pyproject.toml",
        ".env.example",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "configs/*.yaml",
        "src/**/*.py",
        "scripts/*.py",
        "scripts/*.sh",
        "tests/**/*.py",
        ".github/workflows/*.yml",
        "monitoring/**/*.yml",
        "monitoring/**/*.yaml",
        "k8s/**/*.yaml",
        "docs/adr/*.md",
        "docs/model-card.md",
        "docs/runbook.md",
        "DATA_CARD.md",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    for index, path in enumerate(unique, start=1):
        relative = path.relative_to(ROOT)
        h(story, f"A.{index} {relative}", 2)
        p(
            story,
            FILE_EXPLANATIONS.get(
                str(relative),
                "Repository infrastructure or reference file; inline comments and surrounding chapters define its execution role.",
            ),
            "Small",
        )
        code(story, path.read_text(encoding="utf-8"))


FILE_EXPLANATIONS = {
    "src/taxi_duration/config.py": "Pydantic tabanlı strict configuration schema, inheritance merge ve YAML loader.",
    "src/taxi_duration/data/ingest.py": "Streaming, atomik TLC month download ve SHA-256 identity.",
    "src/taxi_duration/data/validation.py": "Source schema allowlist, label construction, domain filtering ve quality report.",
    "src/taxi_duration/features/builder.py": "Train ve serve için tek leakage-safe scikit-learn transformer.",
    "src/taxi_duration/models/factory.py": "HistGradientBoosting ve log target wrapper içeren fitted Pipeline factory.",
    "src/taxi_duration/models/artifact.py": "Model/metadata save-load ve lineage record.",
    "src/taxi_duration/training/split.py": "Chronological train/validation/test boundary.",
    "src/taxi_duration/training/evaluate.py": "Regression metrics ve tail/bias ölçümü.",
    "src/taxi_duration/training/gates.py": "Promotion kararını açıklayıcı failure listesiyle üretir.",
    "src/taxi_duration/training/pipeline.py": "Offline lifecycle orchestrator: validate, split, fit, evaluate, gate, package, profile.",
    "src/taxi_duration/serving/schemas.py": "Version 1 HTTP input-output contract.",
    "src/taxi_duration/serving/app.py": "Startup loading, endpoints, timezone conversion, prediction ve error semantics.",
    "src/taxi_duration/serving/metrics.py": "Prometheus metric definitions.",
    "src/taxi_duration/monitoring/profile.py": "Training reference histogramları ve PSI primitive'i.",
    "src/taxi_duration/monitoring/drift.py": "Current batch'i reference ile karşılaştıran drift report.",
    ".github/workflows/ci.yml": "Her code change için software/supply-chain gate zinciri.",
    ".github/workflows/retrain.yml": "Data ingestion'dan exact-digest production promotion'a gated ML CD.",
    "Dockerfile": "Gate'i geçmiş artifact içeren non-root, multi-stage serving image.",
    "docker-compose.yml": "Local API, Prometheus ve Grafana observability stack.",
}


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story: list[Flowable] = []
    add_cover(story)
    add_front_matter(story)
    add_core_chapters(story)
    add_source_appendix(story)
    document = GuideDocTemplate(str(OUTPUT))
    document.multiBuild(story)
    print(OUTPUT)


register_fonts()
STYLES = styles()

if __name__ == "__main__":
    build()
