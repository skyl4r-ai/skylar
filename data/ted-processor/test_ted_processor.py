#!/usr/bin/env python3
# filepath: test_ted_processor.py
"""Tests for TED processor — based on real TED data format."""

import tempfile
from pathlib import Path

from ted_processor import (
    ExtractedDoc,
    FormatEra,
    clean_text,
    detect_format,
    is_italian_file,
    parse_legacy_txt,
    parse_xml_ted,
    write_pretrain_batch,
)

IT_NOTICE_FILE = """\
  **********************************************
  ***  T E D   D A I L Y - D E L I V E R Y   ***
  ***  ( ITALIAN   - VERSION)                ***
  **********************************************

1.00/067191
TI: I-Napoli: pasti
PD: 19930102
ND: 54411-1992
CY: IT
AU: UNITA SANITARIA LOCALE N. 43
AB: Merce: Confezionamento di pasti freddi per il personale e pasti
    caldi per i degenti.
    Valore base: Lit 700 000 000, IVA inclusa.
TX: 1. Ente appaltante: Unita sanitaria locale n. 43, via Valente (rione
    Miano) I-80145 Napoli.
    Tel. 754 06 05.
    2. Procedura: Gara ristretta.

1.00/067190
TI: I-Roma: lavori stradali
PD: 19930102
ND: 54412-1992
CY: IT
AU: COMUNE DI ROMA
AB: Lavori di manutenzione straordinaria.
TX: 1. Ente appaltante: Comune di Roma.
    2. Importo: Lit 2 500 000 000.
"""

XML_NOTICE_IT = """\
<?xml version="1.0" encoding="UTF-8"?>
<TED_EXPORT xmlns="http://publications.europa.eu/resource/schema/ted/R2.0.9/publication" DOC_ID="123456-2019">
  <TRANSLATION_SECTION>
    <ML_TITLES>
      <ML_TI_DOC LG="IT">
        <TI_TEXT><P>Appalto per servizi di consulenza informatica</P></TI_TEXT>
      </ML_TI_DOC>
    </ML_TITLES>
  </TRANSLATION_SECTION>
  <FORM_SECTION>
    <F02_2014 LG="IT">
      <OBJECT_CONTRACT>
        <TITLE><P>Appalto per servizi di consulenza informatica</P></TITLE>
        <SHORT_DESCR><P>Il Ministero cerca un fornitore IT.</P></SHORT_DESCR>
      </OBJECT_CONTRACT>
    </F02_2014>
  </FORM_SECTION>
</TED_EXPORT>
"""


def test_legacy_parser():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(IT_NOTICE_FILE)
        path = Path(f.name)

    docs = parse_legacy_txt(path)
    path.unlink()

    assert len(docs) == 2, f"Expected 2 docs, got {len(docs)}"
    assert "Napoli" in docs[0].title
    assert "pasti freddi" in docs[0].abstract
    assert "Unita sanitaria" in docs[0].body
    assert "Roma" in docs[1].title
    print("✓ Legacy TXT parser: OK")


def test_xml_parser():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(XML_NOTICE_IT)
        path = Path(f.name)

    docs = parse_xml_ted(path)
    path.unlink()

    assert len(docs) == 1
    assert "consulenza informatica" in docs[0].title
    assert "Ministero" in docs[0].abstract
    print("✓ XML parser: OK")


def test_format_detection():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(IT_NOTICE_FILE)
        path = Path(f.name)
    assert detect_format(path) == FormatEra.LEGACY_TXT
    path.unlink()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(XML_NOTICE_IT)
        path = Path(f.name)
    assert detect_format(path) == FormatEra.XML_TED
    path.unlink()
    print("✓ Format detection: OK")


def test_italian_file_detection():
    assert is_italian_file(Path("IT_19930102_1993001_ISO_ORG"))
    assert is_italian_file(Path("it_20100102_001_utf8_org"))
    assert not is_italian_file(Path("EN_19930102_1993001_ISO_ORG"))
    assert not is_italian_file(Path("FR_19950103_001_ISO_ORG"))
    assert not is_italian_file(Path("000005-2015.xml"))
    print("✓ Italian file detection: OK")


def test_pretrain_output():
    docs = [
        ExtractedDoc(doc_id="t1", title="Titolo", abstract="Riassunto", body="Corpo"),
        ExtractedDoc(doc_id="t2", title="Secondo", body="Altro testo"),
    ]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = Path(f.name)

    write_pretrain_batch(docs, path, append=False)
    content = path.read_text(encoding="utf-8")
    path.unlink()

    assert content.count("<bos>") == 2
    assert content.count("<eos>") == 2
    assert "Titolo" in content
    assert "Riassunto" in content
    print("✓ Pretrain output: OK")


def test_clean_text():
    dirty = '  <P>Hello   \x00  world</P>\n\n\n\nfoo  '
    clean = clean_text(dirty)
    assert "\x00" not in clean
    assert "<P>" not in clean
    assert "Hello world" in clean
    print("✓ Text cleaning: OK")


def test_real_file():
    """Test against the real uploaded EN file (if available)."""
    real = Path("/mnt/user-data/uploads/1772385631691_EN_19930102_1993001_ISO_ORG")
    if not real.exists():
        print("⊘ Real file test: skipped (file not available)")
        return

    docs = parse_legacy_txt(real)
    assert len(docs) == 199, f"Expected 199 notices, got {len(docs)}"

    # Verify no garbage in titles
    for doc in docs[:10]:
        assert not doc.title.startswith("1.0"), f"Separator leaked into title: {doc.title[:50]}"
        assert len(doc.title) > 5, f"Title too short: {doc.title}"

    print(f"✓ Real file test: {len(docs)} notices parsed correctly")


if __name__ == "__main__":
    test_format_detection()
    test_italian_file_detection()
    test_legacy_parser()
    test_xml_parser()
    test_pretrain_output()
    test_clean_text()
    test_real_file()
    print("\n🎉 All tests passed!")