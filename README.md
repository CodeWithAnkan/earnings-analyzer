---
title: Veritas AI API
emoji: 📊
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# Veritas AI — Backend API

FastAPI backend for Veritas AI, an earnings call narrative drift detection engine.

## Endpoints

- `GET /drift/summary` — drift signal counts for all 21 companies
- `GET /drift/alerts/{ticker}` — active alerts for a ticker
- `GET /drift/timeline/{ticker}` — full drift timeline
- `GET /drift/quotes/{ticker}` — evidence quotes for an alert
- `GET /intelligence/similar` — semantic search via FinBERT
- `GET /search` — keyword and speaker search