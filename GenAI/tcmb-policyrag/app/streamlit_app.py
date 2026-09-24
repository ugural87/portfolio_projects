from __future__ import annotations

from datetime import date

import streamlit as st

from tcmb_policyrag.config import Settings
from tcmb_policyrag.models import SearchFilters
from tcmb_policyrag.service import RAGService

st.set_page_config(page_title="TCMB PolicyRAG", page_icon="🏦", layout="wide")
st.title("TCMB PolicyRAG")
st.caption("2019'dan itibaren TCMB duyuruları üzerinde kaynak bağlı soru-cevap")


@st.cache_resource
def service() -> RAGService:
    return RAGService(Settings.from_env())


with st.sidebar:
    st.header("Arama ayarları")
    top_k = st.slider("Getirilecek parça", 3, 15, 8)
    date_from = st.date_input("Başlangıç tarihi", value=None, min_value=date(2019, 1, 1))
    date_to = st.date_input("Bitiş tarihi", value=None, min_value=date(2019, 1, 1))
    instruments = st.multiselect(
        "Politika aracı",
        [
            "reserve_requirements",
            "securities_maintenance",
            "deposit_and_kkm",
            "credit",
            "liquidity",
            "macroprudential",
        ],
    )
    show_context = st.toggle("Getirilen parçaları göster", value=False)

question = st.text_area(
    "Sorunuz",
    placeholder="2023-2024 döneminde KKM'den TL mevduata geçiş hangi araçlarla teşvik edildi?",
    height=100,
)
if st.button("Yanıtla", type="primary", disabled=not question.strip()):
    filters = SearchFilters(date_from=date_from, date_to=date_to, instruments=instruments)
    try:
        with st.spinner("TCMB kaynakları aranıyor..."):
            result = service().ask(question, filters=filters, top_k=top_k)
    except Exception as exc:
        st.error(f"Sorgu çalıştırılamadı: {exc}")
    else:
        if result.abstained:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)
        st.subheader("Kaynaklar")
        if not result.citations:
            st.info("Yanıtta kullanılabilir bir kaynak gösterilmedi.")
        for citation in result.citations:
            label = " · ".join(
                part
                for part in [citation.announcement_number, citation.published_date, citation.title]
                if part
            )
            with st.expander(f"[{citation.citation_id}] {label}"):
                st.write(citation.quote)
                st.link_button("TCMB kaynağını aç", citation.source_url)
        if show_context:
            st.subheader("Retrieval sonuçları")
            for hit in result.retrieved:
                hit_label = hit.chunk.announcement_number or hit.chunk.title
                with st.expander(f"#{hit.rank} · {hit.score:.3f} · {hit_label}"):
                    st.write(hit.chunk.text)
                    st.caption(hit.chunk.source_url)
