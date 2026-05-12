# Context-Aware Retrieval Engine — Benchmark Report

> **Generated:** 2026-05-12 16:00:56  
> **Embedding Model:** `all-MiniLM-L6-v2` (384d)  
> **Vector Index:** FAISS `IndexFlatIP` (Cosine Similarity on L2-normalised vectors)  
> **Dataset:** technical_context.json (5 queries · 100 GCP architecture chunks)

---

## Summary

| # | Query | Avg Score A | Avg Score B | Delta | Winner |
|---|-------|:-----------:|:-----------:|:-----:|:------:|
| 1 | How does the system handle peak load? | 0.4073 | 0.7037 | +0.2964 | **B ✓** |
| 2 | What happens when the database goes down? | 0.3709 | 0.5857 | +0.2148 | **B ✓** |
| 3 | How do you keep users logged in? | 0.2614 | 0.4821 | +0.2207 | **B ✓** |
| 4 | How do you know when something breaks? | 0.2735 | 0.5984 | +0.3249 | **B ✓** |
| 5 | What if an entire region fails? | 0.3226 | 0.5515 | +0.2289 | **B ✓** |

**Strategy B outperformed Strategy A in 5/5 queries.**  
**Overall average score improvement: +0.2571**

---

## Query 1: "How does the system handle peak load?"

**Strategy B Expanded Query:**
> *How does the system handle peak load? horizontal scaling auto-scaling HPA load balancing traffic spike capacity planning throughput rate limiting pod replicas node pool manage orchestrate*

| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |
|:----:|------------------------|:-------:|--------------------------|:-------:|
| #1 | Traffic shaping controls are applied at multiple layers to protect backend services from overload. C… | 0.4395 | Workloads run on Google Kubernetes Engine (GKE) with Horizontal Pod Autoscaler (HPA) configured to s… | 0.7953 |
| #2 | Workloads run on Google Kubernetes Engine (GKE) with Horizontal Pod Autoscaler (HPA) configured to s… | 0.3988 | Kubernetes ResourceQuotas and LimitRanges govern compute resource consumption across namespaces. Eac… | 0.6668 |
| #3 | Kubernetes ResourceQuotas and LimitRanges govern compute resource consumption across namespaces. Eac… | 0.3836 | Event-driven workloads use KEDA (Kubernetes Event-Driven Autoscaling) to scale consumer deployments … | 0.6491 |

**Avg Score A:** `0.4073` | **Avg Score B:** `0.7037` | **Delta:** `+0.2964 ↑`

---

## Query 2: "What happens when the database goes down?"

**Strategy B Expanded Query:**
> *What happens when the database goes down? Cloud SQL Spanner PostgreSQL read replica primary connection pooling sharding failover outage unavailable circuit breaker replica promotion*

| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |
|:----:|------------------------|:-------:|--------------------------|:-------:|
| #1 | Cloud SQL for PostgreSQL is deployed in a high-availability configuration with a synchronous standby… | 0.3962 | Cloud SQL for PostgreSQL is deployed in a high-availability configuration with a synchronous standby… | 0.7393 |
| #2 | Graceful shutdown ensures in-flight requests complete cleanly when pods are terminated during deploy… | 0.3610 | Applications connect to Cloud SQL exclusively through the Cloud SQL Auth Proxy, which handles IAM-ba… | 0.5301 |
| #3 | The system targets a Recovery Time Objective (RTO) of 15 minutes and a Recovery Point Objective (RPO… | 0.3554 | Database traffic is split between primary write instances and read replicas to distribute load and p… | 0.4877 |

**Avg Score A:** `0.3709` | **Avg Score B:** `0.5857` | **Delta:** `+0.2148 ↑`

---

## Query 3: "How do you keep users logged in?"

**Strategy B Expanded Query:**
> *How do you keep users logged in? authentication JWT session token Redis Memorystore stateless refresh token*

| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |
|:----:|------------------------|:-------:|--------------------------|:-------:|
| #1 | User authentication relies on JSON Web Tokens (JWT) signed with RSA-256. The system is fully statele… | 0.3008 | User authentication relies on JSON Web Tokens (JWT) signed with RSA-256. The system is fully statele… | 0.6499 |
| #2 | User authentication follows the OAuth 2.0 Authorization Code flow with PKCE (Proof Key for Code Exch… | 0.2534 | User authentication follows the OAuth 2.0 Authorization Code flow with PKCE (Proof Key for Code Exch… | 0.4003 |
| #3 | Centralised log management aggregates logs from all GKE workloads, Cloud Run services, and GCP manag… | 0.2300 | API rate limiting is enforced at two layers: Apigee Gateway for external clients and an in-process R… | 0.3960 |

**Avg Score A:** `0.2614` | **Avg Score B:** `0.4821` | **Delta:** `+0.2207 ↑`

---

## Query 4: "How do you know when something breaks?"

**Strategy B Expanded Query:**
> *How do you know when something breaks? monitoring alerting SLO Cloud Monitoring Cloud Trace error budget p99 latency incident PagerDuty*

| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |
|:----:|------------------------|:-------:|--------------------------|:-------:|
| #1 | Production changes follow a structured change management process to minimise unplanned outages. Chan… | 0.2793 | Operational visibility is delivered through Cloud Monitoring with custom dashboards tracking p50, p9… | 0.6229 |
| #2 | Kubernetes liveness, readiness, and startup probes are configured separately to handle distinct fail… | 0.2753 | Service Level Objectives are defined collaboratively between engineering and product using the four … | 0.5911 |
| #3 | Incidents are managed through a structured process: detection via Cloud Monitoring alerts, triage by… | 0.2660 | Incidents are managed through a structured process: detection via Cloud Monitoring alerts, triage by… | 0.5812 |

**Avg Score A:** `0.2735` | **Avg Score B:** `0.5984` | **Delta:** `+0.3249 ↑`

---

## Query 5: "What if an entire region fails?"

**Strategy B Expanded Query:**
> *What if an entire region fails? disaster recovery multi-region RTO RPO failover Cloud Storage cross-region replica promotion runbook*

| Rank | Strategy A — Raw Search | Score A | Strategy B — AI-Enhanced | Score B |
|:----:|------------------------|:-------:|--------------------------|:-------:|
| #1 | The system targets a Recovery Time Objective (RTO) of 15 minutes and a Recovery Point Objective (RPO… | 0.3579 | The system targets a Recovery Time Objective (RTO) of 15 minutes and a Recovery Point Objective (RPO… | 0.7129 |
| #2 | Multi-region availability is achieved through GKE Fleet, which groups clusters across us-central1, e… | 0.3371 | Multi-region availability is achieved through GKE Fleet, which groups clusters across us-central1, e… | 0.4865 |
| #3 | Site Reliability Engineering practices govern production changes through error budget policies. Each… | 0.2728 | Resilience is validated through structured chaos engineering experiments using LitmusChaos deployed … | 0.4551 |

**Avg Score A:** `0.3226` | **Avg Score B:** `0.5515` | **Delta:** `+0.2289 ↑`

---
