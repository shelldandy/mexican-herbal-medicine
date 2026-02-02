# Herbolaria - UNAM Traditional Mexican Medicine

A Python scraper and RAG (Retrieval-Augmented Generation) application for the [UNAM Digital Library of Traditional Mexican Medicine](http://www.medicinatradicionalmexicana.unam.mx/), which contains encyclopedic information about traditional medicine practices, medicinal plants, and indigenous healing knowledge from Mexico.

## RAG Application

Query the scraped data using natural language with your own API keys.

### Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Build the search index (requires `OPENAI_API_KEY` environment variable):

   ```bash
   export OPENAI_API_KEY=your-key-here
   python -m app.indexer --build
   ```

3. Run the application:

   ```bash
   streamlit run app/main.py
   ```

4. In your browser:
   - Select your LLM provider (OpenAI or Anthropic)
   - Enter your API key
   - Ask questions in Spanish about traditional Mexican medicine

### Supported Models

**OpenAI:**

- GPT-4o (most capable)
- GPT-4o Mini (faster)

**Anthropic:**

- Claude Sonnet 4 (balanced)
- Claude 3.5 Haiku (faster)

Note: When using Anthropic, you also need an OpenAI API key for generating embeddings.

### Example Questions

- ¿Qué plantas se usan para tratar la diabetes?
- ¿Cómo usan los mayas las plantas medicinales?
- ¿Qué es el temascal y para qué se usa?
- ¿Cuáles son los usos medicinales del nopal?

---

## Scraper

- Scrapes all 4 main sections of the library:
  - **DEMTM** - Diccionario Enciclopédico de la Medicina Tradicional Mexicana (Encyclopedia Dictionary)
  - **APMTM** - Atlas de las Plantas de la Medicina Tradicional Mexicana (Medicinal Plants Atlas)
  - **MTPIM** - La Medicina Tradicional de los Pueblos Indígenas de México (Indigenous Peoples' Medicine)
  - **FMIM** - Flora Medicinal Indígena de México (Indigenous Medicinal Flora)
- Converts internal cross-references to relative markdown links
- Downloads images locally with proper attribution
- Generates structured markdown files with YAML frontmatter
- Supports incremental scraping with progress tracking
- Configurable rate limiting for respectful scraping

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/herbolaria.git
   cd herbolaria
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   python -m playwright install chromium
   ```

## Usage

### Basic Usage

Scrape all sections:

```bash
python -m scraper.main
```

### Command Line Options

```
usage: main.py [-h] [--section {dictionary,plants,peoples,flora,all}]
               [--output OUTPUT] [--letter LETTER] [--delay DELAY]
               [--show-browser] [--verbose]

options:
  -h, --help            show this help message and exit
  --section {dictionary,plants,peoples,flora,all}
                        Section to scrape (default: all)
  -o, --output OUTPUT   Output directory (default: data)
  -l, --letter LETTER   Only scrape entries starting with this letter (for testing)
  -d, --delay DELAY     Delay between requests in seconds (default: 1.5)
  --show-browser        Show browser window (for debugging)
  -v, --verbose         Enable verbose logging
```

### Examples

Scrape only the plants section:

```bash
python -m scraper.main --section plants
```

Test with a single letter (useful for testing):

```bash
python -m scraper.main --section dictionary --letter a
```

Scrape with visible browser (for debugging):

```bash
python -m scraper.main --show-browser
```

Adjust request delay:

```bash
python -m scraper.main --delay 2.0
```

## Output Structure

```
data/
├── diccionario/           # Dictionary entries organized A-Z
│   ├── a/
│   │   ├── abeja.md
│   │   └── ...
│   ├── b/
│   └── ...
├── plantas/               # Plant monographs
│   ├── por-nombre-botanico/
│   ├── por-nombre-popular/
│   └── hongos/
├── pueblos-indigenas/     # Indigenous peoples
│   ├── maya.md
│   ├── nahua.md
│   └── ...
├── flora-medicinal/       # Indigenous medicinal flora
└── images/                # Downloaded images
    ├── plantas/
    └── pueblos/
```

## Output Format

Each entry is saved as a markdown file with YAML frontmatter:

```markdown
---
title: "Sábila"
botanical_name: "Aloe vera"
family: "Liliaceae"
source: "http://www.medicinatradicionalmexicana.unam.mx/apmtm/termino.php?l=3&t=aloe-vera"
scraped_at: "2026-01-29"
---

# Sábila

**Nombre botánico:** Aloe vera

![Sábila](../../images/plantas/sabila.jpg)
_La imagen fué proporcionada por: José Rangel Sánchez_

## Sinonimia botánica

Aloe barbadensis Miller, Aloe vulgaris Lam.

## Sinonimia popular

Posacmetl ([náhuatl](../../pueblos-indigenas/nahua.md)): "maguey morado"...

## Etnobotánica y antropología

En Puebla, es común su uso en problemas de la piel como [granos](../../diccionario/g/granos.md)...
```

Cross-references between sections are automatically converted to relative markdown links.

## Progress Tracking

The scraper saves progress to `data/.progress/` and can resume from where it left off if interrupted. To start fresh, delete the progress files:

```bash
rm -rf data/.progress/
```

## Rate Limiting

The scraper includes a configurable delay between requests (default: 1.5 seconds) to be respectful to the server. Please do not reduce this significantly.

## Data Quality Scripts

The `scripts/` directory contains utilities for auditing and cleaning scraped data:

### Audit Script

Check for data quality issues:

```bash
python scripts/audit_data_gaps.py
```

This detects:

- Missing `family` fields in plant files
- Files in accented letter folders (á, é, í, etc.)
- Navigation artifacts in content
- Invalid files (empty filenames, whitespace-only titles)
- Broken cross-reference links
- Flora section completeness

Use `--json` for machine-readable output.

### Cleanup Script

Fix common data issues:

```bash
# Preview changes (dry run)
python scripts/cleanup_data.py --all --dry-run

# Apply all fixes
python scripts/cleanup_data.py --all

# Individual operations
python scripts/cleanup_data.py --delete-invalid      # Remove invalid files
python scripts/cleanup_data.py --normalize-names     # Fix filenames with spaces
python scripts/cleanup_data.py --clean-navigation    # Remove navigation artifacts
```

## License

This scraper is provided for research and educational purposes. The content from the UNAM Digital Library is subject to their terms of use. As stated on their website:

> "Los conocimientos y la información original de esta publicación son de origen y creación colectiva, sus poseedores y recreadores son los pueblos indígenas de México, por lo que deben seguir siendo colectivos y, en consecuencia, está prohibida toda apropiación privada."

(The knowledge and original information in this publication are of collective origin and creation, their holders and recreators are the indigenous peoples of Mexico, so they must remain collective and, consequently, private appropriation is prohibited.)
