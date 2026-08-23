"""
qa_review.py — QA Review Task (Technical + Didactic Checks)
============================================================

Issue #7: AI-Driven Feedback Loop — QA Reviewer Agent

Defines a CrewAI **Task** that, when executed by the QA Reviewer agent,
performs a thorough review of all generated lab files.  The task mandates
two distinct checks:

  1. **Technical Correctness Check** — Zero syntax errors, no missing
     imports, no hallucinated variables, completely self-contained execution.
  2. **Didactic & Clarity Check** — README instructions and code comments
     use clear, accessible language suited for MBO4 vocational students.
     TODO markers must be actionable and unambiguous.

If either check fails, the QA Reviewer is explicitly instructed to use
CrewAI's delegation mechanism to assign a fix task back to the Lab &
Project Developer with specific, actionable feedback.
"""

from __future__ import annotations

from crewai import Agent, Task


def create_qa_review_task(
    *,
    agent: Agent,
    course_name: str,
    run_id: str | None = None,
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that performs QA review of generated lab files.

    Parameters
    ----------
    agent : Agent
        The QA Reviewer agent that will execute this task.
    course_name : str
        The name of the vocational course (e.g. "Full-Stack Web Development").
    run_id : str or None
        The unique run identifier (e.g. ``"2026-08-23_120000_Course_Name"``).
        Used to locate the correct per-run output directory.
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ── Sanitize course name for filesystem paths ──────────────────────
    safe_course_name = (
        course_name.strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )

    # ── Determine the labs directory path ──────────────────────────────
    if run_id:
        labs_dir = f"output/{run_id}/labs/{safe_course_name}/"
    else:
        labs_dir = f"output/labs/{safe_course_name}/"

    # ── Build the description ──────────────────────────────────────────
    description = (
        f"## QA Review — Generated Labs for: {course_name}\n\n"
        f"**Labs Directory:** `{labs_dir}`\n\n"
        f"---\n\n"
        f"### Your Task\n\n"
        f"You must review ALL files in the labs directory above.  Use your "
        f"`DirectoryReadTool` to list the directory contents, then use your "
        f"`FileReadTool` to read every source file and README.\n\n"
        f"The labs are organised into three tiers:\n"
        f"- `tier1_foundations/` — Single-file exercises with TODO markers.\n"
        f"- `tier2_application/` — Multi-file mini-projects.\n"
        f"- `tier3_architecture/` — Service-oriented systems.\n\n"
        f"Each tier has `starter/` and `solution/` subdirectories.\n\n"
        f"---\n\n"
        f"### 🔍  Check 1: Technical Correctness\n\n"
        f"For EVERY source file (both starter/ and solution/), verify:\n\n"
        f"- **Zero syntax errors** — The code must be valid and parseable "
        f"in the target language.  Check for missing brackets, unmatched "
        f"quotes, incorrect indentation, and invalid keywords.\n"
        f"- **No missing imports** — Every function, class, or module used "
        f"in the code must be imported or defined.  Check that all "
        f"`import`/`require`/`use` statements are present and correct.\n"
        f"- **No hallucinated variables or functions** — Every variable "
        f"referenced must be declared.  Every function called must exist "
        f"(either defined in the file, imported, or part of the standard "
        f"library).  Watch for typos in variable/function names.\n"
        f"- **Completely self-contained execution** — After following the "
        f"README instructions, a student must be able to run the code "
        f"without any additional fixes.  Check that file paths in "
        f"instructions match the actual file names.  Check that dependency "
        f"files (requirements.txt, package.json, etc.) list everything "
        f"needed.\n"
        f"- **Solution code must actually work** — The solution/ files "
        f"must be complete, runnable implementations that produce the "
        f"expected output described in the README.\n\n"
        f"### 🔍  Check 2: Didactic & Clarity Check\n\n"
        f"For EVERY README.md and code comment, verify:\n\n"
        f"- **Clear, accessible language** — The text must be understandable "
        f"by MBO4 vocational students (Dutch HBO-equivalent, ages 16-20).  "
        f"Avoid overly academic jargon, complex theoretical explanations, "
        f"and unnecessarily long sentences.  Use concrete examples.\n"
        f"- **Actionable TODO markers** — Every `# TODO:` or `// TODO:` "
        f"marker must be a concrete, specific instruction.  A student "
        f"should know exactly *what* to implement and *how* to verify "
        f"their work.  Vague TODOs like 'implement this function' or "
        f"'add error handling' are UNACCEPTABLE.  Good examples:\n"
        f"  - `# TODO(t1): Write a function called 'calculate_average' "
        f"that takes a list of numbers and returns their mean.  Test it "
        f"with [1, 2, 3] — it should return 2.0.`\n"
        f"  - `# TODO(t2): Add a try/except block around the file.read() "
        f"call on line 42 to handle FileNotFoundError.  Print a friendly "
        f"message to the user.`\n"
        f"- **Practical, hands-on focus** — Instructions should emphasise "
        f"*doing* over *reading*.  Every concept should be immediately "
        f"followed by a code exercise.  Theory should be minimal and "
        f"directly relevant to the task at hand.\n"
        f"- **Complete README sections** — Every README must have all "
        f"required sections: Learning Objectives, Key Concepts, "
        f"Prerequisites, Getting Started, Project Structure, Lab "
        f"Instructions, Humanics Reflection, and Additional Resources.\n\n"
        f"---\n\n"
        f"### 🔴  CRITICAL: Delegation Mandate\n\n"
        f"If ANY lab file fails EITHER the Technical Correctness Check OR "
        f"the Didactic & Clarity Check, you **MUST** use CrewAI's "
        f"delegation mechanism to assign a fix task back to the **Lab & "
        f"Project Developer**.\n\n"
        f"Your delegation message must include:\n"
        f"1. **Which specific files** need to be fixed (full paths).\n"
        f"2. **What exactly is wrong** in each file (be specific — quote "
        f"the problematic code or text).\n"
        f"3. **Why it matters** for MBO4 students (explain the impact on "
        f"learning).\n"
        f"4. **How to fix it** (provide concrete guidance, not just "
        f"'fix this').\n\n"
        f"Do NOT attempt to fix the files yourself.  Your role is to "
        f"*review and delegate*, not to *rewrite*.  The Lab Developer "
        f"is responsible for implementing fixes based on your feedback.\n\n"
        f"---\n\n"
        f"### 📋  Expected Output\n\n"
        f"Produce a comprehensive **Markdown QA Report** with the following "
        f"sections:\n\n"
        f"```markdown\n"
        f"# QA Review Report — {course_name}\n\n"
        f"## Summary\n"
        f"- Total files reviewed: [count]\n"
        f"- Technical issues found: [count]\n"
        f"- Didactic issues found: [count]\n"
        f"- Delegation(s) sent to Lab Developer: [yes/no]\n"
        f"- Final verdict: [PASSED / NEEDS FIXES]\n\n"
        f"## Technical Correctness Check\n"
        f"[For each file: PASSED or FAILED with specific issues]\n\n"
        f"## Didactic & Clarity Check\n"
        f"[For each file: PASSED or FAILED with specific issues]\n\n"
        f"## Delegation Summary\n"
        f"[If fixes were delegated: what was sent to the Lab Developer]\n\n"
        f"## Sign-Off\n"
        f"[Final confirmation that the code is ready for students, "
        f"or a clear statement that fixes are still needed]\n"
        f"```\n\n"
        f"If ALL checks pass for ALL files, your sign-off must explicitly "
        f"state: '✅ **QA SIGN-OFF: All labs are technically correct and "
        f"didactically appropriate for MBO4 students.  Ready for classroom "
        f"use.**'\n"
    )

    expected_output = (
        "A comprehensive Markdown QA Report with Technical Correctness "
        "Check results, Didactic & Clarity Check results, a Delegation "
        "Summary (if any fixes were delegated to the Lab Developer), and "
        "a final Sign-Off confirming readiness for MBO4 students."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.qa_reviewer import create_qa_reviewer

    agent = create_qa_reviewer(verbose=True)
    task = create_qa_review_task(
        agent=agent,
        course_name="JavaScript OOP Basics",
        run_id="2026-08-23_120000_JavaScript_OOP_Basics",
        verbose=True,
    )
    print("✅ QA Review Task created successfully.\n")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
