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

---

## Running as an AWS Lambda Container

You can deploy this service as a container image on AWS Lambda for serverless execution.

### 1. Build the Lambda-Compatible Image

Ensure your Dockerfile (or Dockerfile.lambda) is compatible with AWS Lambda's runtime. Typically, you should use the AWS Lambda Python base image, for example:

```Dockerfile
FROM public.ecr.aws/lambda/python:3.11
# (add your copy/install commands here)
CMD ["app.handler"]  # Example entrypoint; adjust as needed for your handler
```

Build the image:

```bash
docker build -t openai-tagging-qdrant-lambda -f Dockerfile.lambda .
```

### 2. Test Locally (Optional)

You can test your Lambda container image locally before deployment:

```bash
docker run -p 9000:8080 openai-tagging-qdrant-lambda
```

Then invoke it with:

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"url":"https://example.com"}'
```

### 3. Deploy to AWS Lambda

1. Push your image to Amazon ECR (Elastic Container Registry).
2. In AWS Lambda, create a function using the "Container image" option and provide your image URI from ECR.

### 4. Passing Inputs

- The Lambda expects a JSON payload like:
  ```json
  {
    "url": "https://example.com"
  }
  ```
- Input should be provided as the event payload for the Lambda function.
- The response will return the tags and embedding dimension.

---

If you need further customization (such as a different handler or multi-parameter input), make sure your code in the container is compatible with AWS Lambda's Python handler interface.
