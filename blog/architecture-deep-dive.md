# Architecture Deep Dive: How DermaMind.ai Fits Together

*Published: July 2026*

DermaMind.ai has evolved into a layered architecture rather than a collection of unrelated scripts. This post explains the major pieces and how they interact.

## Layer 1: Perception

The perception layer lives under `derma_agent/derma_core/perception/` and is responsible for turning a Whole Slide Image into features the rest of the system can reason about.

In practice, this layer handles:
- slide ingestion metadata,
- mock TITAN-style pathology summaries,
- mock APOLLO-style embeddings,
- pooled slide-level representations.

The important architectural point is that downstream components should never need to know how raw slide patches were processed. They consume stable slide-level outputs.

## Layer 2: Knowledge grounding

The knowledge fabric provides a graph-based memory of prior biology:
- genes,
- diseases,
- pathways,
- drugs,
- clinical features.

This layer is critical because it constrains search. Without grounding, the agent could generate statistically testable but biologically unhelpful hypotheses.

The graph layer also supports path-based prior scoring, which gives the system a way to express “how plausible does this hypothesis look before statistical testing?”

## Layer 3: Safe execution

DermaMind.ai uses a CodeAct-style pattern, but with explicit safeguards:
- AST validation before execution,
- a critic step for unsafe code patterns,
- a restricted execution context,
- execution-history capture for debugging.

This is essential in a research agent. If the system is allowed to generate code dynamically, safety cannot be an afterthought.

## Layer 4: Statistical validation

The clinical statistics layer turns candidate ideas into analyzable survival outputs. The current system supports:
- Kaplan–Meier analysis,
- Cox proportional hazards regression,
- ML-based discriminative summaries,
- DSL-driven analysis requests.

A major improvement in the latest revision is that significance is no longer judged only at the single-test level. Session-wide BH/FDR correction now treats discovery as an exploratory process rather than a sequence of isolated claims.

## Layer 5: Presentation

DermaMind.ai exposes results in two ways:
- Streamlit dashboards for interactive exploration,
- a Vite/Tailwind landing site for framework communication.

These are not just cosmetic layers. They shape how users understand the system. If the UI hides the analysis path, the system feels magical. If the UI exposes the graph path, the code path, and the adjusted statistical result, the system feels scientific.

## Why this separation helps

This layered structure improves the project in four ways:
1. **Replaceability** — you can upgrade one layer without rewriting everything.
2. **Auditability** — each step has a narrow responsibility.
3. **Testing** — statistical logic and perception logic can be validated separately.
4. **Documentation** — contributors can understand the system by layer.

## Recommended mental model

Think of DermaMind.ai as five cooperating subsystems:
- Eyes
- Memory
- Hands
- Statistics
- Interface

That framing is simple enough for new contributors, but accurate enough to describe the architecture to technical reviewers.
