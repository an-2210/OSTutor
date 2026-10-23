# OSTutor: AI Tutor for Operating Systems

OSTutor is a study companion for Operating Systems students. It answers questions with short, structured explanations, offers quizzes and flashcards, simulates CPU scheduling with Gantt charts, and links to the textbooks behind its answers.

## Features

- **Tutor chat**: ask any OS question and get a markdown answer with key points and an example.
- **Quiz generator and flashcards**: practice by topic.
- **CPU scheduler simulator**: Gantt charts for FCFS, SJF, and Round Robin.
- **Library**: links to the textbooks and references the tutor draws on, at `/library`.

## How the tutor answers

Each question is answered by the first model that responds:

1. **Qwen 2.5 14B** running locally through [Ollama](https://ollama.com). Free, offline, no API key.
2. **Gemini** (`gemini-3.8-flash`), when Qwen fails or the key is set. Needs `GEMINI_API_KEY`.

If neither answers, the tutor says it has no reliable answer and points to the textbook. It does not guess.

## Retrieval pipeline (RAG)

The retrieval pipeline lives in [ostutor/rag/](ostutor/rag/). It turns course materials into a searchable index so answers can cite their sources.

1. **Ingest** (`ingest.py`): find course files in `ostutor/data/rag/raw/` (PDF, PPTX, TXT, or DOCX) and extract text page by page or slide by slide.
2. **Clean** (`clean.py`): rejoin hyphenated words, strip headers and footers, normalize whitespace.
3. **Chunk** (`chunk.py`): split into about 700-token chunks with 100 tokens of overlap. Each chunk keeps its metadata (unit, topic, page, source, license) and is saved to `chunks.jsonl`.
4. **Embed** (`embed.py`): encode each chunk with `all-MiniLM-L6-v2` (384 dimensions), L2-normalized so the inner product equals cosine similarity.
5. **Index** (`retrieve.py`): build a FAISS `IndexFlatIP` index and save it to `ostutor/data/rag/index/`.
6. **Retrieve**: embed the student's question, find the top-k most similar chunks, and return them with their source and page for citation.

`build_index.py` runs steps 1 to 5. `test_retrieval.py` runs step 6 from the command line:

```bash
cd ostutor
python3 rag/build_index.py
python3 rag/test_retrieval.py "What is the difference between preemptive and non-preemptive scheduling?"
```

**Status:** the pipeline is built, but it has no course materials to index yet, and the tutor doesn't use it. See [Current limitations](#current-limitations).

## Setup

### 1. Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pydantic pyyaml
```

### 2. Local model (Ollama)

```bash
brew install ollama
ollama pull qwen2.5:14b
```

Qwen 2.5 14B needs about 9 GB of disk space and 16 GB of RAM for comfortable use. For a smaller machine, set `OLLAMA_MODEL` to a smaller model such as `llama3.2:3b`.

### 3. Optional: Gemini fallback

Get a free key from Google AI Studio and export it in the terminal you run the server from:

```bash
export GEMINI_API_KEY=your_key_here
```

Do not commit the key or paste it into files.

### 4. Run the app

```bash
uvicorn app.main:app --port 8000
```

Then open http://127.0.0.1:8000.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where the Ollama server runs |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Local model for answers |
| `GEMINI_API_KEY` | unset | Enables the Gemini fallback |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model name |
| `TUTOR_TEMPERATURE` | `0.2` | Lower values give steadier answers |

## Current limitations

- **Textbook retrieval is not connected yet.** The tutor passes five hand-written reference paragraphs to the model. The retrieval pipeline in `ostutor/rag/` is built but has no course materials to index yet.
- **Answers can still be wrong.** Check important facts against the textbooks in the library.

## Repository layout

```text
app/
├── main.py            FastAPI server: API routes and tutor logic
└── frontend/          index.html, app.js, styles.css, library.html
src/
├── utils/             Config loading and logging
└── models/            LLM fine-tuning module
ostutor/
├── rag/               Retrieval pipeline (ingest, chunk, embed, FAISS index)
├── docs/              Design notes for the pipeline and the instruction dataset
└── data/              Taxonomy, instruction dataset, RAG metadata
configs/config.yaml    App settings (loaded at startup)
```

`src/utils/` provides config loading and logging for the app, and `src/models/fine_tune_llm.py` backs the fine-tuning endpoint. `configs/config.yaml` is loaded at startup and still contains settings from an earlier cyber-threat project.

## License

See [LICENSE](LICENSE).
