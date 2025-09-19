# Span (a.k.a. Block Attention) Examples

## Span Queries

This directory contains a [span query](https://github.com/IBM/spnl#readme). To send a query, first prepare the query shape:

```bash
curl -s -XPOST http://localhost:8000/v1/query/prepare --data @./query-ab.json -o /dev/null -w "%{time_total}\n"
1.504452
```

And then you can execute the query in either order, and you should see millisecond-level TTFT:

```bash
curl -s -XPOST http://localhost:8000/v1/query/execute --data @./query-ba.json -o /dev/null -w "%{time_total}\n"
0.077699
```

```bash
curl -s -XPOST http://localhost:8000/v1/query/execute --data @./query-ab.json -o /dev/null -w "%{time_total}\n"
0.078419
```
