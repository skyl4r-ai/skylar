#!/usr/bin/env python3
# filepath: test_ted_processor.py
"""Verify TED processor parsers with synthetic data."""

import tempfile
from pathlib import Path

from ted_processor import (
    ExtractedDoc,
    FormatEra,
    clean_text,
    detect_format,
    parse_legacy_txt,
    parse_xml_eforms,
    parse_xml_old,
    write_pretrain_batch,
)

LEGACY_NOTICE_IT = """\
ND: 1993001-0042
CY: IT
TI: Lavori di ristrutturazione del palazzo comunale di Roma
AB: Il Comune di Roma indice gara pubblica per la ristrutturazione integrale del palazzo comunale sito in Via del Corso.
TX: Importo complessivo dei lavori: 2.500.000 ECU. Durata prevista: 18 mesi. Le offerte devono pervenire entro il 15 marzo 1993.
"""

LEGACY_NOTICE_DE = """\
ND: 1993001-0043
CY: DE
TI: Bauarbeiten in Berlin
AB: Renovierung eines Bürogebäudes
TX: Gesamtbetrag: 1.000.000 ECU
"""

XML_OLD_NOTICE = """\
<?xml version="1.0" encoding="UTF-8"?>
<TED_EXPORT xmlns="http://publications.europa.eu/resource/schema/ted/R2.0.9/publication" DOC_ID="123456-2019" EDITION="2019001">
  <CODED_DATA_SECTION>
    <NOTICE_DATA>
      <ISO_COUNTRY VALUE="IT"/>
    </NOTICE_DATA>
  </CODED_DATA_SECTION>
  <TRANSLATION_SECTION>
    <ML_TITLES>
      <ML_TI_DOC LG="IT">
        <TI_CY>Italia</TI_CY>
        <TI_TOWN>Roma</TI_TOWN>
        <TI_TEXT><P>Appalto per servizi di consulenza informatica</P></TI_TEXT>
      </ML_TI_DOC>
    </ML_TITLES>
  </TRANSLATION_SECTION>
  <FORM_SECTION>
    <F02_2014 CATEGORY="TRANSLATION" FORM="F02" LG="IT">
      <OBJECT_CONTRACT>
        <TITLE><P>Appalto per servizi di consulenza informatica</P></TITLE>
        <SHORT_DESCR><P>Il Ministero delle Finanze cerca un fornitore per servizi di consulenza informatica per la modernizzazione dei sistemi IT.</P></SHORT_DESCR>
      </OBJECT_CONTRACT>
      <COMPLEMENTARY_INFO>
        <INFO_ADD><P>Le offerte devono essere presentate entro 60 giorni dalla pubblicazione.</P></INFO_ADD>
      </COMPLEMENTARY_INFO>
    </F02_2014>
    <F02_2014 CATEGORY="ORIGINAL" FORM="F02" LG="EN">
      <OBJECT_CONTRACT>
        <TITLE><P>IT consulting services contract</P></TITLE>
        <SHORT_DESCR><P>The Ministry of Finance seeks an IT consulting provider.</P></SHORT_DESCR>
      </OBJECT_CONTRACT>
    </F02_2014>
  </FORM_SECTION>
</TED_EXPORT>
"""

XML_OLD_NOTICE_NO_IT = """\
<?xml version="1.0" encoding="UTF-8"?>
<TED_EXPORT xmlns="http://publications.europa.eu/resource/schema/ted/R2.0.9/publication" DOC_ID="789012-2019">
  <CODED_DATA_SECTION>
    <NOTICE_DATA>
      <ISO_COUNTRY VALUE="DE"/>
    </NOTICE_DATA>
  </CODED_DATA_SECTION>
  <FORM_SECTION>
    <F02_2014 CATEGORY="ORIGINAL" FORM="F02" LG="DE">
      <OBJECT_CONTRACT>
        <TITLE><P>Beratungsvertrag</P></TITLE>
      </OBJECT_CONTRACT>
    </F02_2014>
  </FORM_SECTION>
</TED_EXPORT>
"""


def test_legacy_parser():
    """Test legacy TXT notice parser."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(LEGACY_NOTICE_IT)
        f.write("\n\n")
        f.write(LEGACY_NOTICE_DE)
        path = Path(f.name)

    docs = parse_legacy_txt(path)
    path.unlink()

    assert len(docs) == 1, f"Expected 1 IT doc, got {len(docs)}"
    doc = docs[0]
    assert "ristrutturazione" in doc.title, f"Title missing expected text: {doc.title}"
    assert "Comune di Roma" in doc.abstract, f"Abstract missing: {doc.abstract}"
    assert "2.500.000" in doc.body, f"Body missing: {doc.body}"
    print("✓ Legacy TXT parser: OK")


def test_xml_old_parser():
    """Test old XML format parser."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(XML_OLD_NOTICE)
        path = Path(f.name)

    docs = parse_xml_old(path)
    path.unlink()

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    doc = docs[0]
    assert "consulenza informatica" in doc.title, f"Title: {doc.title}"
    assert "Ministero delle Finanze" in doc.abstract, f"Abstract: {doc.abstract}"
    print("✓ XML old parser: OK")


def test_xml_old_no_italian():
    """Test that non-Italian XML notices are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(XML_OLD_NOTICE_NO_IT)
        path = Path(f.name)

    docs = parse_xml_old(path)
    path.unlink()

    assert len(docs) == 0, f"Expected 0 docs (non-IT), got {len(docs)}"
    print("✓ XML old parser (non-IT skip): OK")


def test_format_detection():
    """Test format detection."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(LEGACY_NOTICE_IT)
        path = Path(f.name)
    assert detect_format(path) == FormatEra.LEGACY_TXT
    path.unlink()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(XML_OLD_NOTICE)
        path = Path(f.name)
    assert detect_format(path) == FormatEra.XML_OLD
    path.unlink()
    print("✓ Format detection: OK")


def test_pretrain_output():
    """Test pretrain.txt output format."""
    docs = [
        ExtractedDoc(doc_id="test-001", title="Titolo Test", abstract="Abstract test", body="Corpo del documento"),
        ExtractedDoc(doc_id="test-002", title="Secondo Doc", body="Altro testo"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = Path(f.name)

    write_pretrain_batch(docs, path, append=False)
    content = path.read_text(encoding="utf-8")
    path.unlink()

    assert content.count("<bos>") == 2, f"Expected 2 <bos> markers"
    assert content.count("<eos>") == 2, f"Expected 2 <eos> markers"
    assert "Titolo Test" in content
    assert "Secondo Doc" in content
    print("✓ Pretrain output format: OK")


def test_clean_text():
    """Test text cleaning."""
    dirty = '  <P>Hello   \x00  world</P>\n\n\n\nfoo  '
    clean = clean_text(dirty)
    assert "\x00" not in clean
    assert "<P>" not in clean
    assert "Hello world" in clean
    print("✓ Text cleaning: OK")


if __name__ == "__main__":
    test_format_detection()
    test_legacy_parser()
    test_xml_old_parser()
    test_xml_old_no_italian()
    test_pretrain_output()
    test_clean_text()
    print("\n🎉 All tests passed!")
