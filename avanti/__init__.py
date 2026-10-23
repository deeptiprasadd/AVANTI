"""AVANTI -- research paper analyzer.

Layered so each stage can be tested on its own:

  extract    PDF bytes  -> layout-correct text (columns, tables, glyphs)
  sections   text       -> Abstract / Method / Results / ... segmentation
  textutil   text       -> sentences, citation stripping, prose filtering
  analyze    sections   -> structured findings, metrics, datasets, phrases
  summarize  sections   -> section-aware extractive summary
  qa         document   -> searchable index, answers cited to a page
  references document   -> parsed bibliography with in-text citation counts
  pipeline   PDF bytes  -> all of the above in one call (both apps use it)
  compare    2 analyses -> side-by-side comparison of two papers
  sources    query      -> arXiv / OpenAlex / Europe PMC search, safe download
  llm        text       -> optional Gemini report (only if a key is set)
"""

__version__ = "2.1.0"
