# PDF fixtures (R1)

Place your **3 grant PDFs** here (tables, images, text).

```text
data/fixtures/pdfs/
  doc1.pdf
  doc2.pdf
  doc3.pdf
```

Then run:

```bash
source .venv/bin/activate
python -m govgrant.rag.cli ingest
python -m govgrant.rag.cli query "What eligibility requirements are mentioned?"
```

PDFs are ignored by git (binary fixtures). Keep originals outside the repo if sensitive.
