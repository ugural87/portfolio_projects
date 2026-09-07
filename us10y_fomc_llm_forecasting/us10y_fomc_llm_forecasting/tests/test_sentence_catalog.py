from us10y_fomc.llm.sentence_catalog import (
    build_sentence_catalog,
    normalise_fed_editorial_ellipsis,
)


def test_official_fed_ellipsis_is_normalised_inside_catalog() -> None:
    official_excerpt = (
        "The Committee will maintain the target range for the federal funds rate at "
        "0 to 1/4 percent. Economic conditions . . . are likely to warrant exceptionally "
        "low levels of the federal funds rate for an extended period."
    )
    catalog = build_sentence_catalog(official_excerpt, "C")
    assert list(catalog) == ["C00001", "C00002"]
    assert normalise_fed_editorial_ellipsis(catalog["C00002"]) == (
        "economic conditions are likely to warrant exceptionally low levels of the "
        "federal funds rate for an extended period."
    )
