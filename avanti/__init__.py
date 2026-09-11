"""AVANTI -- research paper analyzer.

Layered so each stage can be tested on its own:

  extract    PDF bytes  -> layout-correct text (columns, tables, glyphs)
  sections   text       -> Abstract / Method / Results / ... segmentation
  textutil   text       -> sentences, citation stripping, prose filtering
  analyze    sections   -> structured findings, metrics, datasets, phrases
  summarize  sections   -> section-aware extractive summary
  qa         document   -> searchable index, answers cited to a page
  sources    query      -> arXiv search and PDF download
  llm        text       -> optional Gemini report (only if a key is set)
"""

__version__ = "2.0.0"
