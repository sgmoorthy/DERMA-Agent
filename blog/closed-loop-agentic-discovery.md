# Closed-Loop Agentic Discovery in DermaMind.ai

*Published: July 2026*

DermaMind.ai is not designed as a single predictive model. It is designed as a **closed-loop scientific system**: observe pathology data, ground the observation in prior knowledge, execute a statistical validation step, then feed the result back into the agent state.

## Why the closed loop matters

In many AI pathology demos, the model stops at classification. A slide is labeled, a probability is produced, and the workflow ends there. That is useful for automation, but it does not resemble how translational research is actually done.

Real cancer discovery is iterative. A researcher:
1. sees a pattern,
2. compares it with prior biology,
3. tests a hypothesis,
4. revises the question,
5. repeats.

DermaMind.ai mirrors that structure.

## The four phases

### 1. Perception
The WSI layer turns a slide into a compact representation of morphology. In the current codebase this is mock-backed but now uses an attention-style pooled slide representation so that tile-level signals contribute to a single slide summary.

### 2. Knowledge grounding
The knowledge fabric constrains hypothesis generation. The system is not supposed to hallucinate entirely free-form biology; it should reason from known genes, pathways, diseases, and therapies.

### 3. Execution
The agent writes or routes analysis logic into a sandboxed environment. Instead of treating statistical testing as an opaque black box, DermaMind.ai explicitly exposes the analysis path.

### 4. Validation
Kaplan–Meier analysis, Cox modeling, and machine-learning summaries convert a candidate idea into a quantitative result. The output is then logged and made available for inspection.

## Why this architecture is more trustworthy

A closed loop is more auditable than a pure generative system because each phase can be inspected:
- what the slide representation contained,
- what graph evidence was used,
- what code was executed,
- what the survival result actually showed.

This structure does not remove uncertainty, but it makes uncertainty easier to reason about.

## The research-grade upgrade

A key improvement in the current version is that discovery significance is now treated as a **session-wide multiple-testing problem**. Instead of using only raw p-values, the discovery engine applies Benjamini–Hochberg FDR correction across the exploratory session.

That makes the closed loop more realistic for exploratory cancer research, where many hypotheses are considered before a result is reported.

## Takeaway

DermaMind.ai is best understood as a **scientific workflow engine**, not just a pathology model. Its value comes from how perception, graph grounding, safe execution, and quantitative validation fit together into a single loop.
