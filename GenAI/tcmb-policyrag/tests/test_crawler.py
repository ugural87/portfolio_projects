from scripts.tcmb_crawler import archive_entries, instrument_tags


def test_turkish_suffixes_are_tagged() -> None:
    tags = instrument_tags("Mevduata geçiş, kredilerinin seyri ve repoya erişim")
    assert "deposit_and_kkm" in tags
    assert "credit" in tags
    assert "liquidity" in tags


def test_pdf_only_open_letter_is_explicitly_excluded() -> None:
    raw = """
    <div class="block-collection-box"><div class="collection-content">
      <a class="collection-title" href="/asset/DUY2024-18.pdf"
         title="Bankamız Kanunu Uyarınca Hükûmete Gönderilen Açık Mektup (2024-18)">
         Bankamız Kanunu Uyarınca Hükûmete Gönderilen Açık Mektup (2024-18)
      </a>
      <div class="collection-tag">05/04/2024</div>
    </div></div>
    """
    excluded: list[dict] = []
    unparsed: list[dict] = []
    entries = archive_entries(
        raw,
        "https://www.tcmb.gov.tr/archive/2024/",
        2024,
        excluded_documents=excluded,
        unparsed_boxes=unparsed,
    )
    assert entries == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "open_letter_out_of_scope"
    assert unparsed == []


def test_descriptive_announcement_slug_is_included() -> None:
    raw = """
    <div class="block-collection-box"><div class="collection-content">
      <a href="/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2020/DUY2020-Baskan">
        Başkanın Yazılı Basın Açıklaması
      </a>
      <div class="collection-tag">09/11/2020</div>
    </div></div>
    """
    entries = archive_entries(raw, "https://www.tcmb.gov.tr/archive/2020/", 2020)
    assert len(entries) == 1
    assert entries[0]["url"].endswith("/DUY2020-Baskan")


def test_pdf_only_presentation_is_explicitly_excluded() -> None:
    raw = """
    <div class="block-collection-box"><div class="collection-content">
      <a href="/asset/SunumBY.pdf">Başkan Yardımcısının Sunumu (İngilizce)</a>
      <div class="collection-tag">11/07/2024</div>
    </div></div>
    """
    excluded: list[dict] = []
    unparsed: list[dict] = []
    assert (
        archive_entries(
            raw,
            "https://www.tcmb.gov.tr/archive/2024/",
            2024,
            excluded_documents=excluded,
            unparsed_boxes=unparsed,
        )
        == []
    )
    assert excluded[0]["reason"] == "presentation_out_of_scope"
    assert unparsed == []
