# Research Math and Safety in DermaMind.ai

*Published: July 2026*

When an agentic research system makes scientific claims, two questions matter immediately:
1. **Is the math appropriate?**
2. **Is the execution path safe and auditable?**

This post explains how DermaMind.ai answers both.

## Survival analysis as the validation backbone

The framework uses standard survival-analysis tools because oncology discovery is ultimately about time-to-event outcomes:
- Kaplan–Meier curves for descriptive group comparisons,
- log-rank testing for survival separation,
- Cox proportional hazards models for multivariable interpretation.

These methods are familiar to clinical researchers and make the output easier to review.

## Why raw p-values are not enough

An agentic system can generate many hypotheses quickly. That creates a multiple-testing problem. If you inspect enough raw p-values, some will look significant by chance.

That is why the paper-level formalism matters. The current engine now applies **Benjamini–Hochberg false discovery rate control** across the full exploration session.

In practice, this means the system distinguishes between:
- a finding that is significant before correction,
- a finding that remains significant after correction.

This is a much more honest way to report exploratory discovery.

## Attention-style slide representations

The perception layer now mirrors the paper’s idea of attention-style pooling over tile embeddings. Rather than pretending a slide embedding appears from nowhere, the architecture now explicitly models:
- patch embeddings,
- attention weights,
- a pooled slide representation.

Even in mock mode, this matters because it forces the architecture to respect the shape of the scientific abstraction.

## Why safety must live next to the math

Statistical correctness is only half the story. If the agent writes analysis code dynamically, the execution path must be constrained.

DermaMind.ai therefore combines:
- AST validation,
- restricted imports,
- critic-based review,
- sandboxed execution,
- execution trace capture.

That combination makes it much easier to understand not only *what* the result was, but *how* it was produced.

## The practical principle

A research agent should not be judged only by whether it can produce interesting hypotheses. It should be judged by whether it can produce:
- interpretable hypotheses,
- reproducible tests,
- corrected significance metrics,
- safe execution traces.

That is the direction DermaMind.ai is moving toward.
