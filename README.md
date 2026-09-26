# Privacy Gateway

The initial FastAPI service lives in [`privacy-gateway/`](privacy-gateway/).
Start its API, PostgreSQL, and Redis from that directory with:

```bash
cd privacy-gateway && docker compose up --build
```

See [`privacy-gateway/README.md`](privacy-gateway/README.md) for endpoints,
development checks, configuration, and the current implementation scope.

The surrounding Replit workspace also contains its standard API and design
scaffolding; the privacy gateway is kept as a separate Python service so its
FastAPI and async SQLAlchemy stack remains independent.