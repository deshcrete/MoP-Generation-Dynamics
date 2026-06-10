## Instructions:

Do not make assumptions: if my instructions are unclear, ask me as many clarifying questions as required to remove all ambiguity.

This is a research codebase. Do not maintain any silent fallbacks as we change the code. Fail loudly.

This code should be explicit to the reader. Do not use complex engineering patterns when simple ones will suffice, unless I explicitly ask for modular or scalable solutions.

While working, do not remove previous comments unless the underlying logic changes and the comment is incorrect.

While designing solutions, please refer to codebase /path/to/code/a and /path/to/code/b for examples of good architectural decisions, correct implementations, required features, etc.

Prioritize reproducibility and clarity over clever design and efficiency.

When designing solutions, ensure they are aligned with the high level project plan in design_doc.md and paper.tex.

Each TODO item in TODO.md should be committed separately.

## Strucutre:

### design_doc.md
The problem we are trying to solve or question we are trying to ask, the methods we will use to address it, and other relevant details required to structure the project.
A linear roadmap of medium and short term objectives required to build out the project.
High level project architecture that the agent should respect when designing new features so that you don’t end up with parallel systems providing similar or identical features. This should be provided at a fairly high level of abstraction but can include class signatures etc if you have a desired API or data flow that you want the project to respect. In general, I find it best to copy the architectures of existing reference codebases, allowing the AI to fill in the details.

### nodes.md
On complex projects, while building, discoveries are often made. These might be quirks of the algorithms themselves, limitations of particular libraries, gotchas on particular compute clusters, the result of implementation research of other codebases, etc. These kinds of findings often don’t have a natural home in the plan documents but are important context to agents who are trying to solve problems in-flight. I like to maintain parallel notes.md files to store this information. These notes should periodically be compacted and maintained to keep track of only the elements which are not yet encoded in the codebase itself (as e.g., comments).

### task_plan.md
use this to explicility plan and low level experiment design decisions
