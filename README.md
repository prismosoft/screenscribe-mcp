# Video Analyzer MCP

A minimal remote MCP server for **Gemini-powered video analysis**. This fork intentionally reduces the original ScreenScribe MCP surface to one agent-neutral primitive: `analyze_video`.

The calling client can be ChatGPT, Codex, Claude Code, Cursor, or any Streamable HTTP MCP client. The server does not depend on Anthropic or OpenAI models; Gemini performs video understanding and the connected agent decides what to do with the result.

## MCP tool

### `analyze_video(url, focus="", time_range="")`

Gemini watches the video (visual frames + audio/narration) and returns structured JSON with:

- summary
- timestamped sections
- key visual moments
- on-screen text
- visual style
- important people, objects, and locations

`focus` is optional. `time_range` accepts `START-END` using seconds, `MM:SS`, or `H:MM:SS`.

## Gemini configuration

All model configuration is controlled from environment variables, so changing models does not require a code deploy:

| Variable | Default |
|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` |
| `GEMINI_THINKING_LEVEL` | `medium` (`low`, `medium`, `high`) |
| `GEMINI_MEDIA_RESOLUTION` | `low` (`low`, `medium`, `high`) |
| `GEMINI_API_KEY` | required |

## Remote MCP + OAuth

The production server uses Streamable HTTP at:

```text
https://YOUR_HOST/mcp
```

Authentication follows the same pattern used by the Excalimate deployment:

- OAuth protected-resource metadata
- OAuth authorization-server metadata
- dynamic client registration
- Authorization Code flow with PKCE (`S256`)
- refresh tokens + revocation
- password-gated browser authorization page
- persistent OAuth state in SQLite

OAuth endpoints:

```text
/.well-known/oauth-protected-resource
/.well-known/oauth-protected-resource/mcp
/.well-known/oauth-authorization-server
/oauth/register
/oauth/authorize
/oauth/token
/oauth/revoke
```

Required OAuth environment variables:

```text
PUBLIC_BASE_URL=https://YOUR_HOST
OAUTH_LOGIN_PASSWORD=your-password
OAUTH_SESSION_SECRET=a-long-random-secret
OAUTH_DB_PATH=/data/oauth.sqlite3
```

Mount persistent storage at `/data` in Railway. This keeps registered clients and refresh tokens across deploys.

For MCP clients that cannot perform OAuth, an optional `MCP_API_KEY` can be configured and supplied as `Authorization: Bearer ...`.

## Railway

This repo includes a `Dockerfile` and `railway.toml`, deliberately bypassing Railpack's Python/mise builder path.

Recommended Railway setup:

1. Deploy `prismosoft/screenscribe-mcp` from `main`.
2. Mount a persistent volume at `/data`.
3. Set the environment variables from `.env.example`.
4. Generate a public HTTPS domain.
5. Set `PUBLIC_BASE_URL` to that exact domain (without `/mcp`).
6. Add `https://YOUR_HOST/mcp` to ChatGPT as the MCP URL.

Health endpoint:

```text
GET /health
```

It reports whether Gemini is configured, the active model, thinking level, and media resolution without exposing secrets.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export GEMINI_API_KEY=...
export PUBLIC_BASE_URL=https://example.com
export OAUTH_LOGIN_PASSWORD=...
export OAUTH_SESSION_SECRET=replace-with-at-least-32-random-characters
export OAUTH_DB_PATH=/tmp/video-analyzer-oauth.sqlite3
screenscribe-mcp
```

## License

MIT, preserving the upstream project's license.
