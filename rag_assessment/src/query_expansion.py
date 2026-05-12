from __future__ import annotations

from dataclasses import dataclass

from .interfaces import BaseQueryExpander


# ---------------------------------------------------------------------------
# Vertex AI API-shape response class (mirrors the real SDK's contract)
# ---------------------------------------------------------------------------

@dataclass
class GenerateContentResponse:
    """Mirrors vertexai.generativeai.types.GenerateContentResponse."""
    text: str

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Domain-aware expansion rules (GCP cloud-native architecture vocabulary)
# ---------------------------------------------------------------------------

_EXPANSION_RULES: dict[str, list[str]] = {
    "peak load": [
        "horizontal scaling", "auto-scaling", "HPA", "load balancing",
        "traffic spike", "capacity planning", "throughput", "rate limiting",
        "pod replicas", "node pool",
    ],
    "handle": ["manage", "orchestrate", "coordinate", "process"],
    "database failure": [
        "failover", "replica promotion", "standby", "high availability",
        "connection pooling", "circuit breaker", "data replication",
        "disaster recovery", "RTO", "RPO",
    ],
    "database": [
        "Cloud SQL", "Spanner", "PostgreSQL", "read replica",
        "primary", "connection pooling", "sharding",
    ],
    "goes down": [
        "failover", "outage", "unavailable", "circuit breaker",
        "replica promotion", "automated recovery",
    ],
    "logged in": [
        "authentication", "JWT", "session", "token", "Redis",
        "Memorystore", "stateless", "refresh token",
    ],
    "user session": [
        "JWT", "stateless authentication", "Redis", "Memorystore",
        "token revocation", "refresh token", "distributed cache",
        "horizontal scaling", "no sticky sessions",
    ],
    "session": [
        "JWT", "stateless", "Redis", "Memorystore", "token", "authentication state",
    ],
    "something breaks": [
        "monitoring", "alerting", "SLO", "Cloud Monitoring", "Cloud Trace",
        "error budget", "p99 latency", "incident", "PagerDuty",
    ],
    "monitoring": [
        "Cloud Monitoring", "SLO", "SLA", "error budget", "Cloud Trace",
        "p95 latency", "alerting policy", "synthetic monitoring",
    ],
    "region fails": [
        "disaster recovery", "multi-region", "RTO", "RPO", "failover",
        "Cloud Storage cross-region", "replica promotion", "runbook",
    ],
    "high availability": [
        "failover", "redundancy", "multi-zone", "health check",
        "load balancing", "auto-scaling", "circuit breaker", "RTO", "RPO",
    ],
    "scaling": [
        "horizontal scaling", "HPA", "Cluster Autoscaler", "VPA",
        "auto-scaling", "pod replicas", "node pool", "throughput",
    ],
    "cache": [
        "Memorystore", "Redis", "Cloud CDN", "cache-aside", "TTL",
        "cache invalidation", "distributed cache", "stampede prevention",
    ],
    "message": [
        "Pub/Sub", "topic", "subscription", "asynchronous", "at-least-once",
        "dead-letter", "queue", "event-driven",
    ],
    "consistency": [
        "Cloud Spanner", "distributed transactions", "Saga pattern",
        "eventual consistency", "CAP theorem", "event sourcing", "idempotent",
    ],
    "security": [
        "IAM", "VPC", "mTLS", "Secret Manager", "Cloud Armor", "encryption",
        "firewall rules", "service account",
    ],
}

_GENERIC_EXPANSION = [
    "GCP", "cloud-native", "microservices", "distributed systems",
    "Google Cloud", "Kubernetes", "GKE",
]


# ---------------------------------------------------------------------------
# Mock that perfectly mirrors vertexai.generativeai.GenerativeModel
# ---------------------------------------------------------------------------

class MockVertexGenerativeModel(BaseQueryExpander):
    """
    Drop-in mock for vertexai.generativeai.GenerativeModel.

    Implements BaseQueryExpander.expand() by calling generate_content()
    internally, so tests can spy on generate_content() to verify call counts.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model_name = model_name

    # -- BaseQueryExpander interface ----------------------------------------

    def expand(self, query: str) -> str:
        """Build a prompt, call generate_content, return the expanded text."""
        prompt = (
            "You are an expert in GCP cloud-native architecture.\n"
            "Expand the following query into a keyword-rich version for semantic search.\n"
            "Include technical synonyms and related GCP concepts.\n"
            "Return only the expanded query.\n\n"
            f"Query: {query}"
        )
        response = self.generate_content(prompt)
        return response.text

    # -- Vertex AI SDK surface -----------------------------------------------

    def generate_content(self, prompt: str) -> GenerateContentResponse:
        """
        Mirrors GenerativeModel.generate_content().
        Extracts the query from the prompt and applies domain expansion rules.
        """
        original_query = self._extract_query(prompt)
        expanded = self._apply_expansion_rules(original_query)
        return GenerateContentResponse(text=expanded)

    # -- Internal helpers -----------------------------------------------------

    def _extract_query(self, prompt: str) -> str:
        for line in reversed(prompt.strip().splitlines()):
            line = line.strip()
            if line.startswith("Query:"):
                return line.removeprefix("Query:").strip()
        return prompt

    def _apply_expansion_rules(self, query: str) -> str:
        query_lower = query.lower()
        additions: list[str] = []

        for keyword, synonyms in _EXPANSION_RULES.items():
            if keyword in query_lower:
                additions.extend(synonyms)

        if not additions:
            additions = _GENERIC_EXPANSION

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for term in additions:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique.append(term)

        return f"{query} {' '.join(unique[:12])}"
