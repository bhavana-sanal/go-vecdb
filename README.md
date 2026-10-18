# go-vecdb

[![CI](https://github.com/bhavana-sanal/go-vecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavana-sanal/go-vecdb/actions) ![License](https://img.shields.io/badge/license-MIT-blue)

A small in-memory **vector database** in Go with no dependencies: cosine-similarity search over embeddings, either exact or approximate.

- **Exact search** — brute force with a bounded min-heap for top-k.
- **Approximate search** — random-hyperplane **LSH** (locality-sensitive hashing) with multiple hash tables; only vectors that collide with the query are ranked.

```go
ix := vecdb.New(384, 8, 12, 42) // dim, LSH tables, bits per table, seed
ix.Add("doc-1", embedding)
hits, _ := ix.ApproxSearch(queryEmbedding, 5)
```

```bash
go test ./...
go test -bench . -benchmem   # compare exact vs approximate speed
```
