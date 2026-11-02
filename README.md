# sandbox-charles

Charles's starting point for personal Liforma code experiments. This is a public, collaborative repository: every Liforma developer is welcome to check out, modify and contribute to any project here. The name indicates an initial curator, not exclusive ownership.

Keep each experiment/application in its own top-level folder, with a README describing its purpose, setup, environment variables and run commands. Dependencies and ports belong to each application; there is no shared root application or deployment. Use an `.env.example` for required configuration and keep real credentials and private assets out of this public repository.

Shared team experiments belong in [LiformaLtd/sandbox](https://github.com/LiformaLtd/sandbox), which lives at `sandboxes/sandbox` in the Liforma workspace. This repository lives at `sandboxes/sandbox-charles` and can also be cloned independently.

Use an isolated Git worktree for each implementing agent/task. Multiple agents must not share a checkout, switch another agent's branch or share writable dependencies. Coordinate development ports and use branches/PRs for contributions.
