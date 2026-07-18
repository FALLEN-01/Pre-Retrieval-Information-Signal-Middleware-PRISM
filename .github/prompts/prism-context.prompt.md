---
mode: agent
description: Use when you need to inspect, explain, or modify PRISM pipeline components with repository-aware context.
---

You are working in the PRISM repository. Follow these instructions carefully.

## Project context
- PRISM is a retrieval pipeline with an Information Optimization Layer (IOL) between retrieval and a downstream consumer.
- Core runtime flow: source ingestion in sources/ -> indexing in retrieval/vectorstore.py -> retrieval and optimization in pipeline.py and prism/ -> optional answer generation in consumer/llm_consumer.py.
- Configuration lives in config.py and is often the first place to inspect for behavior changes.

## When responding
- Be repository-aware and specific to the code in this workspace.
- Prefer citing the relevant files and modules rather than giving generic advice.
- If the task involves code changes, explain the intended change, the files affected, and the likely impact.
- If the task involves debugging, trace the issue from the entry point through the relevant module before proposing a fix.
- Use the project's conventions: Python, Windows-friendly commands, and the existing PRISM architecture.

## Required response style
- Start with a concise summary of the task.
- Provide actionable next steps or implementation guidance.
- Keep explanations practical and grounded in the repository structure.
- If code is modified, include a brief note on verification steps.

## Helpful defaults
- Prefer examining config.py before changing behavior.
- For source-related changes, inspect the relevant adapter in sources/.
- For retrieval issues, inspect retrieval/vectorstore.py.
- For optimization behavior, inspect prism/.
- For consumer integration, inspect consumer/llm_consumer.py.

## Optional task inputs
- If the user provides a file, symbol, or error message, use that as the primary anchor for the investigation.
- If no specific target is provided, infer the most relevant module from the request and explain the likely entry points.
