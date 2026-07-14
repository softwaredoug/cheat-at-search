---
name: commit-to-git
description: When asked to commit changes to git, use this skill to generate a commit message and Co-authored-by trailer for the model.
compatibility: opencode
metadata:
  app_type: electron
---
The user will ask you to commit changes to git, here's some important guidance to create good commit messages

## Co-authored by:

Use a co-authored by to ensure we know this was co-developed by an AI agent.

Co-authored-by: opencode <noreply@opencode.ai>

## Commit message format

A good commit message should follow git commit best practices.

```
<imperative verb> <short description of the change>

<detailed description of the change>
```

IE lets say we optimized the code for a function called `calculateSum`, the commit message could look like this:

```
Optimize calculateSum function for performance

Previously calculateSum was a loop over an array, but by unrolling the loop
and performing SIMD operations, we were able to make better use of the hardware.
```

## Pre-commit hooks

Expect various pre-commit hooks to run. Those exist to protect the codebase against bad changes.

Try to resolve them. If they require changes to tests, then prompt the user with suggestions on where the tests
need to be updated.

## Unrelated changes

You're likely committing with specific changes in mind. Something the user asked you to do.

The repo might have other uncommitted changes. Those might be being performed by someone else in parallel and
should be respected. Do not try to undo these or revert these.

If you cannot commit without undoring these changes, stop the commit and ask the user to either commit them or stash them.
