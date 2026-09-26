---
name: Nested Python dependencies
description: Package-manager side effects when Python code is added under the pnpm workspace
---

The Python package installation helper targets the workspace root even when the Python service has its own project file in a subdirectory.

**Why:** An installation for the nested service initialized a second Python project at the pnpm root and changed workspace configuration. Those side effects were unrelated to the service's requested repository structure.

**How to apply:** When adding Python dependencies to a nested service, inspect root-level changes immediately and keep the service's dependency declaration as the source of truth. Avoid leaving a duplicate root Python project or unrelated configuration changes.