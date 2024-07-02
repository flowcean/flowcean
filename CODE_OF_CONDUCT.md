# Coding Guidelines

## Introduction

AGenC is a modular software framework designed to make model learning for cyber-physical systems accessible and versatile.
This document outlines the coding guidelines to ensure a consistent and high-quality codebase.
This document is periodically reviewed and updated based on the evolving needs of the project and the team's feedback.

## Programming Languages and Tools

- **Languages:**
  The project primarily uses Python for implementation, supporting multiple programming languages within the modular pipeline.
- **Version Control:**
  Git is employed for code versioning, hosted on GitHub.
  All changes must undergo a review process via merge requests before merging into the main branch.
- **CI/CD:**
  Continuous Integration and Continuous Deployment pipelines are used for testing, checking, and deploying the code.

## Coding Style

- **Style Guide:** Comply with PEP 8 and [Google documentation style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Naming Conventions:**
  - Classes and Types: `PascalCase`.
  - Functions, Methods, and Variables: `snake_case`.
  - Constants: `UPPERCASE`.
  - Avoid abbreviations to ensure code readability.
- **Documentation:**
  External documentation generated from inline comments following Google style.
  Avoid unnecessary repetition.
  That is, choose a senseful explaining variable name over written documentation.

## Documentation

- Clearly document the purpose, arguments, and return types in function/method docstrings.
- Documentation is mainly done in code by choosing appropriate variable, class, and function names.
- All functions must be annotated with types to ensure type correctness.

## Development Process

- Discuss and assign responsibilities for development changes via GitHub issues.
- Encourage pair programming when necessary.
- Every development change should undergo a short design phase and code review.
- Adhere to good scientific practice standards.
- Prioritize readable code over premature optimizations.
- Optimize only when proven by benchmarks.

## Code Review and Collaboration

- Create merge requests for code changes, ensuring all CI checks pass before merging.
- Follow a code review process with a different, non-author reviewer:
  - The author opens a new merge request to change code
  - The merge request should come with a short title and a useful description to explain the change to the reviewer.
  - A non-author should review the code and request changes, or approve the request if no changes are requested, i.e., no further conversations are open or needed.
  - All non-authors are implicitly tasked to review MRs
  - If there are multiple authors involved in one MR, approval by a non-author is not mandatory.
  - A review should check that the proposed changes comply to our coding guidelines, and carefully evaluate whether the changes are useful and intended.
  - The author is responsible to address the comments and answer questions.
  - During discussion, only the person who opened the discussion is allowed to resolve the discussion. This reduces the risk of misscommunication and maintains accountability. Only the author of the comment is able to decide whether a discussion is resolved.
  - The merge request must only be merged as soon as all comments are resolved and the CI pipeline passes.
- Team communication is mainly via GitHub comments and regular software meetings.
- Open issues for bugs, open tasks, or discussions to archive the decision and design process of the project.

## Testing

- Employ unit tests and integration tests to ensure code correctness.
- Utilize coverage metrics and the testing framework for evaluation.
- Strive for open-source contributions that drive research forward.

## Logging and Error Handling

- Utilize the standard Python logging framework.
- Clearly communicate errors, guiding users to possible causes.
- Avoid verbose logging for a clear and concise user experience.
