"""
models.py — Shared Pydantic Models for Syllabus Swarm
=====================================================

Issue #10: Machine-Readable Course Graph JSON Export (Phase 3 — Export Side)

Defines the **CourseGraph** and **ModuleSummary** Pydantic models that
compose (rather than duplicate) the existing ``CourseSpecification``
model.  Both the exporter layer (:mod:`src.exporters.tool`) and future
module-chaining tools can import from this single module.

Design constraint
-----------------
* ``CourseGraph`` **composes** ``CourseSpecification`` via a
  ``specification`` field — it never duplicates ``course_context``
  or ``primary_language``.
* ``ModuleSummary`` is a lightweight model representing a single
  course module.

Public API
----------
* ``ModuleSummary`` — title, duration_weeks, topics.
* ``CourseGraph`` — specification, course_slug, learning_objectives,
  key_concepts, prerequisites, modules, generated_at.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field

from src.main import CourseSpecification

# ---------------------------------------------------------------------------
# ModuleSummary
# ---------------------------------------------------------------------------


class ModuleSummary(BaseModel):
    """Lightweight representation of a single course module.

    Used by ``CourseGraph.modules`` to describe the module-level
    structure without duplicating full syllabus content.
    """

    title: str = Field(
        description="Human-readable module title (e.g. 'Python Fundamentals').",
        min_length=1,
    )
    duration_weeks: float = Field(
        description="Module duration in weeks (supports fractional weeks).",
        ge=0.0,
    )
    hours_per_week: float = Field(
        description=(
            "Contact hours per week for this module (e.g. 3.0 for "
            "a 3-hour/week module).  Combined with duration_weeks this "
            "gives total module effort: duration_weeks × hours_per_week."
        ),
        ge=0.0,
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Key topics covered in this module.",
    )


# ---------------------------------------------------------------------------
# CourseGraph
# ---------------------------------------------------------------------------


class CourseGraph(BaseModel):
    """Machine-readable course graph that composes the existing spec.

    This model is the primary export format for downstream tooling
    (curriculum memory engine, module chaining, LMS import).
    It **never** duplicates fields from ``CourseSpecification``;
    instead it holds a reference via ``specification``.

    Example
    -------
    >>> from src.main import CourseSpecification
    >>> spec = CourseSpecification(
    ...     course_context="Intro to Python for DS.",
    ...     primary_language="Python",
    ... )
    >>> graph = CourseGraph(
    ...     specification=spec,
    ...     course_slug="intro-python-ds",
    ...     learning_objectives=["Write Python scripts"],
    ...     key_concepts=["variables", "loops"],
    ...     prerequisites=["Basic computer literacy"],
    ...     modules=[
    ...         ModuleSummary(
    ...             title="Getting Started",
    ...             duration_weeks=1.0,
    ...             topics=["installation", "first script"],
    ...         ),
    ...     ],
    ... )
    """

    specification: CourseSpecification = Field(
        description="The existing course specification — never duplicated.",
    )
    course_slug: str = Field(
        description="URL- / filesystem-safe identifier for the course.",
        min_length=1,
    )
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Top-level learning objectives for the entire course.",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Core concepts / competencies learners will acquire.",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Required prior knowledge or courses.",
    )
    modules: list[ModuleSummary] = Field(
        default_factory=list,
        description="Ordered list of course modules.",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        description="ISO 8601 UTC timestamp when the graph was generated.",
    )
