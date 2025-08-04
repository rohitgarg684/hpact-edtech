# OpenAI Tagging & Qdrant Container

## Overview

A Python microservice that fetches content from a URL, tags it using OpenAI GPT-3.5, generates an embedding using OpenAI's `text-embedding-3-large` model, and stores the result in a Qdrant vector database.

## Prerequisites

- Docker
- OpenAI API Key
- Running Qdrant instance (can use [docker-compose](https://qdrant.tech/documentation/quick-start/))

## Environment Variables

Create a `.env` file or set these variables:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Usage

1. **Build & run the container:**

   ```
   docker build -t openai-tagging-qdrant .
   docker run --env-file .env -p 8000:8000 openai-tagging-qdrant
   ```

2. **Send a request:**

   ```bash
   curl -X POST "http://localhost:8000/tag-and-embed/" -H  "accept: application/json" -H  "Content-Type: application/json" -d '{"url":"https://example.com"}'
   ```

## Endpoints

- `POST /tag-and-embed/` - Takes a JSON body with `url` and returns tags, embedding dimension.

## Notes

- The service only stores the first 4000 characters of content for tagging and embedding.
- Qdrant vector size is set for `text-embedding-3-large` (3072).
