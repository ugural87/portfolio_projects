from __future__ import annotations

import json
import re

from openai import OpenAI

from .models import Answer, Citation, SearchHit

SYSTEM = """Sen TCMB politika duyuruları için kaynak bağlı bir araştırma asistanısın.
Yalnızca sağlanan BAĞLAM içindeki bilgileri kullan. Bilgi yetersizse açıkça söyle.
Her olgusal iddiayı [K1], [K2] biçiminde bağlam kimliğiyle destekle.
Tarihleri ve oranları aynen koru; farklı tarihlerdeki kuralları birleştirme.
Yanıtı Türkçe ver. Sonuç yalnızca şu JSON olsun:
{"answer":"...", "citation_ids":["K1"], "abstained":false}
"""


class AnswerGenerator:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(
                question=question,
                answer="Bu soruyu yanıtlamak için yeterli TCMB kaynağı bulunamadı.",
                citations=[],
                retrieved=[],
                model=self.model,
                abstained=True,
            )
        context_parts: list[str] = []
        citation_map: dict[str, SearchHit] = {}
        for index, hit in enumerate(hits, 1):
            cid = f"K{index}"
            citation_map[cid] = hit
            chunk = hit.chunk
            context_parts.append(
                f"[{cid}] {chunk.title}\nTarih: {chunk.published_date or 'yok'}\n"
                f"Duyuru: {chunk.announcement_number or 'yok'}\n"
                f"URL: {chunk.source_url}\n{chunk.text}"
            )
        prompt = f"SORU:\n{question}\n\nBAĞLAM:\n\n" + "\n\n".join(context_parts)
        response = self.client.responses.create(model=self.model, instructions=SYSTEM, input=prompt)
        raw = response.output_text.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                raise ValueError("Model did not return the required JSON object") from None
            payload = json.loads(match.group())
        answer_text = str(payload.get("answer", "Yanıt üretilemedi."))
        requested_ids = [str(cid) for cid in payload.get("citation_ids", [])]
        cited_inline = re.findall(r"\[(K\d+)\]", answer_text)
        all_requested = list(dict.fromkeys(requested_ids + cited_inline))
        ids = [cid for cid in all_requested if cid in citation_map]
        unknown_ids = [cid for cid in all_requested if cid not in citation_map]
        if unknown_ids:
            unknown = set(unknown_ids)
            answer_text = re.sub(
                r"\[(K\d+)\]",
                lambda match: "" if match.group(1) in unknown else match.group(0),
                answer_text,
            )
            answer_text = re.sub(r"\s{2,}", " ", answer_text).strip()
        citations = [
            Citation(
                citation_id=cid,
                announcement_number=citation_map[cid].chunk.announcement_number,
                title=citation_map[cid].chunk.title,
                published_date=citation_map[cid].chunk.published_date,
                source_url=citation_map[cid].chunk.source_url,
                quote=citation_map[cid].chunk.text[:500],
            )
            for cid in ids
        ]
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved=hits,
            model=self.model,
            abstained=bool(payload.get("abstained", False)),
            diagnostics={
                "retrieved_count": len(hits),
                "cited_count": len(citations),
                "unknown_citation_ids": unknown_ids,
            },
        )
