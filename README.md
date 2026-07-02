# screenscribe

**Extract *typed* data from videos — and synthesize across many.**

The best how-to knowledge — live demos, conference talks, tutorials, cooking videos — is trapped in video: you can watch it, but you can't *use* it. screenscribe turns it into structured, validated data an agent can act on. Hand it a schema and it fills it from what's on screen *and* said; point it at a whole channel and it synthesizes one coherent artifact across every video.

**Powered by Gemini (it watches the video); your agent does the reasoning.** Standalone CLI + MCP server (Claude Code, Cursor, Windsurf, any MCP client). One key — and transcript-only mode needs none. Works regardless of the video's language and needs no captions.

---

## What it does — three layers

1. **Watch & extract.** Gemini watches the video. Pull the transcript (free), a cheap whole-video analysis, or the key frames as PNG images your agent opens and reads.
2. **Typed extraction** ⭐ — hand it a **JSON Schema** (or a bundled preset) and get back **validated JSON**: every CLI command shown, the final config file, a recipe with quantities, the code at each step. Prose is for humans to read; this is what an agent automates on.
3. **Cross-video synthesis** ⭐ — point at a **channel / playlist / list of URLs** → fan the typed extraction out over every video → **compound** the results into one artifact conforming to an *aggregate* schema (a cookbook, a technique grammar, a comparison). It scales past a single context window, one capped batch at a time.

The raw extraction is a commodity (it's Gemini under the hood). The durable value is the **structure** and the **cross-video synthesis** — and both only get better as the models do.

---

## Why it matters

- **Watch → automate.** *"Extract every command shown in this setup tutorial"* → a typed list your tooling runs. *"Turn mom's whole cooking channel into one cookbook"* → a synthesized artifact, not 300 summaries.
- **Typed, not prose.** Output validates against your schema; a mismatch retries once, then returns a structured error — never malformed data masquerading as success.
- **Any language, no captions.** Gemini transcribes the spoken audio *and* reads on-screen text — proven on Bengali cooking videos with zero YouTube captions, and on live-coded music with on-screen code.
- **Agent-native.** An MCP server, so video drops straight into your coding loop.
- **Cached + resumable.** Per-`(video, schema)` caching makes re-runs free; synthesis aggregates are persisted and resumable.

---

## Levels — pick the depth you need

| Layer | CLI / API | Cost | What you get |
| ----- | --------- | ---- | ------------ |
| Transcript | `extract --transcript-only` / `extract_transcript` | free, no key | the words |
| Whole-video analysis | `analyze` / `analyze_video` | ~$0.03 | summary, timestamped sections, key moments, on-screen text |
| Frames | `extract` / `extract_frames` | ~$0.03 | key frames as PNGs your agent reads |
| **Typed extraction** ⭐ | `extract-structured` / `extract_structured` | ~$0.03 | **validated JSON conforming to your schema/preset** |
| **Cross-video synthesis** ⭐ | `synthesize` / `synthesize_pass` | ~$0.03 × N | **one compounding artifact across many videos** |

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (or plain `pip`). **No `ffmpeg` install needed** — a static binary is fetched on first use (your system `ffmpeg` is used if present).

**Try it in 10 seconds, no key:**

```bash
uvx screenscribe extract "https://youtu.be/dQw4w9WgXcQ" --transcript-only
```

**Add a Gemini key for everything visual** (read from your shell env or a `.env`):

```bash
export GEMINI_API_KEY=...   # the only key screenscribe needs

# typed extraction — a bundled preset, a schema file, or inline JSON Schema
uvx screenscribe extract-structured "https://youtu.be/dQw4w9WgXcQ" --schema cli_commands
```

([Get a Gemini key](https://aistudio.google.com/apikey).) Everything visual needs it; transcript mode and explicit `--timestamps` need no key.

**Add to your agent as an MCP server (one line):**

```bash
claude mcp add screenscribe -- uvx screenscribe-mcp
```

**Local video files** — anywhere a YouTube URL is accepted (the `analyze`, `extract`, `slides`, and `extract-structured` commands, and the `analyze_video` / `extract_frames` / `extract_structured` MCP tools), you can pass a local video instead — a path or a `file://` URI. Gemini uploads the file and watches it the same way.

```bash
screenscribe analyze /path/to/clip.mp4
screenscribe extract-structured ./cooking.mov --schema recipe
```

Local files have no transcript (so `--transcript-only` is YouTube-only), and are subject to the Gemini Files API limits (~2 GB/file). A session id is synthesized from the filename.

<details>
<summary><b>Run from source (development)</b></summary>

```bash
git clone git@github.com:ashmitb95/video-analyzer-llm.git
cd video-analyzer-llm
python3 -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env   # add your Gemini key, or export it in your shell
pytest
```

</details>

---

## Typed extraction

Hand screenscribe a shape, get validated JSON back. `--schema` accepts a **preset name**, a **path** to a `.json` schema, or an **inline JSON Schema** string.

```bash
# bundled preset
screenscribe extract-structured "https://youtu.be/<id>" --schema cli_commands

# your own schema file (data → stdout, metadata → stderr, so it pipes)
screenscribe extract-structured "https://youtu.be/<id>" --schema ./shape.json | jq .
```

**Bundled presets:** `cli_commands`, `code_blocks`, `final_config`, `step_sequence`, `resources_mentioned`, `chapters`, `recipe`.

- Output is validated against your schema (`jsonschema`); on a mismatch it retries once, then returns `{"status": "invalid", ...}` — never malformed data.
- Cached per `(video, schema)` — re-running the same extraction is free.
- Timestamps come back as numeric `seconds`. Schema field descriptions *are* the prompt — make them explicit about the precision you want.

**MCP:** `extract_structured(url, schema, focus="", time_range="")` — `schema` is a preset name or an inline JSON Schema string. Returns the validated `data` (or a structured error).

---

## Cross-video synthesis

Point at a **channel / playlist / list of URLs** and synthesize one artifact across all of it — *schema-driven*, like per-video extraction but for the aggregate. Two steps: **categorize** (cheap, read-only — discover the buckets and confirm before spending), then **pass** (extract a capped batch and fold it into a compounding aggregate).

**CLI:**

```bash
# 1. discover categories from the channel's titles (cheap, cached) — the confirm view
screenscribe synthesize categorize "https://www.youtube.com/@SomeChannel"

# 2. fold the top-20 of a category into a compounding cookbook (repeat per category)
screenscribe synthesize pass "https://www.youtube.com/@SomeChannel" \
    --category vegetarian --item-schema recipe --aggregate-schema cookbook --top 20
#   --media-res medium   # reads tiny on-screen detail (e.g. code) better
#   the artifact prints to stdout (pipeable); a summary line goes to stderr
```

**MCP:** `synthesize_categorize(url)` then `synthesize_pass(url, item_schema, aggregate_schema, category="", top_n=20)` — the agent shows you the categories, you pick, it runs the passes. (`category=""`/omitted synthesizes the whole resolved set in one pass.)

**Python** (the engine the above sit on):

```python
from dotenv import load_dotenv; load_dotenv()          # load GEMINI_API_KEY
from screenscribe.synthesis import categorize, synthesize_pass

# 1. discover categories from titles (cheap, cached) — the confirm view before any extraction
cats = categorize("https://www.youtube.com/@SomeChannel")
print([(c["name"], c["count"]) for c in cats["categories"]])

# 2. fold the top-N of a category into a compounding, persisted aggregate
out = synthesize_pass(
    "https://www.youtube.com/@SomeChannel", "vegetarian",
    item_schema="recipe", aggregate_schema="cookbook", top_n=20,
)
print(out["aggregate"])        # grows with each pass; resumable

# whole bounded set in one cross-cutting pass (category=None) — e.g. a list of URLs:
synthesize_pass([url1, url2, url3], None, item_schema=my_item, aggregate_schema=my_aggregate)
```

**How it works:** `resolve_videos` (channel/playlist/list → video IDs) → `extract_structured` per video (cached) → a text→structured **aggregation** that folds the new results into a persisted aggregate conforming to your *aggregate schema*. Each pass is bounded (a cap per category), so it scales to a whole channel without a giant prompt; the aggregate compounds and is resumable. One video's failure is isolated (it lands in `failed`, the pass continues) — never a silent drop.

Ships a growable **`cookbook`** aggregate preset (`schemas/aggregate/`); free-form aggregate schemas work too. The building blocks are also usable directly: `resolve_videos(source)` and `extract_structured_batch(source, schema)`.

---

## MCP server

Exposes screenscribe to any MCP-compatible client; the server handles the video, **your agent does the reasoning** (including viewing the extracted frames).

| Tool | Description |
| ---- | ----------- |
| `extract_transcript(url)` | Transcript only — fast, free, no key. |
| `analyze_video(url)` | Gemini watches the whole video → structured analysis (summary, sections, key moments). |
| `extract_frames(url, style)` | Gemini picks moments, ffmpeg extracts PNGs the agent reads. `style="keyframes"` (default) or `"slides"`. |
| `extract_structured(url, schema)` | Typed JSON conforming to a preset or your JSON Schema. The automation primitive. |
| `synthesize_categorize(url)` | Discover categories from a channel/playlist's titles (cheap, read-only) — the confirm view. |
| `synthesize_pass(url, item_schema, aggregate_schema, …)` | Fold the top-N of a category into a compounding cross-video aggregate. |
| `get_video_analysis(id)` | Read Gemini's whole-video analysis for a session. |
| `get_session(session_id)` | Session content (transcript, analysis, frame paths) with `analysis_source` metadata. |
| `list_sessions()` | List all processed videos. |

### Register the MCP server

No paths, no venv, no JSON surgery — `uvx` runs the published package on demand.

**Claude Code:**

```bash
claude mcp add screenscribe -- uvx screenscribe-mcp
```

**Cursor / Windsurf / Continue** (`.cursor/mcp.json` or your client's MCP config):

```json
{
  "mcpServers": {
    "screenscribe": {
      "command": "uvx",
      "args": ["screenscribe-mcp"],
      "env": { "GEMINI_API_KEY": "..." }
    }
  }
}
```

(The `env` block is only needed if your client doesn't inherit your shell environment.) Restart the client after updating its config.

---

## CLI usage

```bash
screenscribe extract <url> [--transcript-only]   # transcript, or download + Gemini-selected key frames
screenscribe slides  <url>                        # standalone "slide" frames
screenscribe analyze <url>                         # whole-video structured analysis (no frames)
screenscribe extract-structured <url> --schema <preset|path|inline>   # typed JSON
screenscribe synthesize categorize <channel-url>   # discover categories (cheap, no extraction)
screenscribe synthesize pass <url> --category <name> --item-schema <s> --aggregate-schema <s>
screenscribe sessions                              # list processed videos
```

Common options on the extract/slides/structured commands: `--focus "…"`, `--time-range 5:00-15:00`, `--timestamps 5:30,10:00` (bypasses AI selection; no key), `--force`. Prefix any command with `uvx ` to run without installing.

---

## Architecture

```
screenscribe/
├── pyproject.toml         Package metadata + console scripts (screenscribe, screenscribe-mcp)
├── src/screenscribe/
│   ├── main.py            CLI — extract / analyze / slides / extract-structured / synthesize / sessions
│   ├── server.py          MCP server — 9 tools
│   ├── resolver.py        (channel | playlist | list | video URL) → normalized video IDs
│   ├── structured_extractor.py  Schema-driven typed extraction + batch fan-out (+ presets)
│   ├── synthesis.py       Cross-video: categorize + compounding synthesize_pass
│   ├── gemini_selector.py Gemini watches the video to pick frames; text→structured calls
│   ├── gemini_analyzer.py Gemini whole-video structured analysis (analyze tier)
│   ├── frame_extractor.py ffmpeg extraction + snap-to-stable + perceptual dedup
│   ├── ffmpeg_paths.py    Resolves ffmpeg/ffprobe — system binary, else bundled static-ffmpeg
│   ├── downloader.py      yt-dlp download + youtube-transcript-api fetch
│   ├── transcript_selector.py  Selection helpers (timestamp parsing, validate/filter)
│   ├── session.py         Session persistence at ~/.video-analyzer/
│   ├── config.py          Gemini model + thresholds
│   └── schemas/           Bundled JSON Schemas: per-video presets + aggregate/ (cookbook)
├── tests/
└── .env.example
```

**Storage** (global, outside the repo, at `~/.video-analyzer/`): per-video sessions (`session.json`, `frames/`, `slides/`), typed-extraction cache (`{id}/structured/{schema}.json`), and synthesis state (`synthesis/{key}/`).

---

## Configuration (`src/screenscribe/config.py`)

| Setting | Default | Description |
| ------- | ------- | ----------- |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model that watches the video. |
| `GEMINI_MEDIA_RESOLUTION_LOW` | `True` | Low-res sampling (cheaper); set `False` for finer on-screen detail (e.g. reading code). |
| `FRAME_SELECTION_MAX` | `25` | Max frames Gemini selects. |
| `SLIDE_SELECTION_MAX` | `15` | Max slides to extract. |
| `IMAGE_MAX_WIDTH` | `1280px` | Frames resized to this width before saving. |
| `MAX_INLINE_TRANSCRIPT_CHARS` | `100000` | Max transcript chars `get_session` returns inline before pointing to the file. |

---

## API keys

One key: **`GEMINI_API_KEY`** (read from the shell environment or a `.env` in the working directory). [Get one from Google AI Studio](https://aistudio.google.com/apikey). No key needed for `extract --transcript-only`, `extract … --timestamps`, or reading existing sessions. Reasoning over the results — answering, generating — is your agent's job, so no separate model key is required.

---

## Notes

- **Any kind of video** — selection and extraction are content-agnostic. Use `--focus` (or a tool's `focus`) to steer toward a subject.
- **Never a silent cut.** Transcripts that exceed the inline cap point to the on-disk file with a `transcript_truncated` flag; resolver/synthesis surface skipped/failed counts; structured extraction returns an explicit `invalid` status rather than malformed data.
- **Machine-local** sessions (`~/.video-analyzer/`); installing on a new machine re-runs extraction once per video.
- `youtube-transcript-api` v1.2.4+ uses `YouTubeTranscriptApi().fetch(video_id)`. If you see `AttributeError: get_transcript`, run `pip install -U youtube-transcript-api`.
