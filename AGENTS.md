# Sandbox agent instructions

This public repository contains multiple independent experimental applications, normally one per top-level folder. All Liforma developers may contribute; the repository name does not restrict ownership.

- Use a dedicated Git worktree for each implementing agent/task. Preserve existing local work.
- Read the selected application's README and scoped instructions before editing. Limit changes and dependency installs to the application involved.
- Give each application its own setup/run instructions, dependencies, environment example and port configuration. Document any deliberate cross-repository dependency.
- Keep secrets, private user data, generated dependency/build directories and large model weights out of Git.
- Run checks appropriate to the application changed. Coordinate ports before starting servers.
- All sandbox app frontends (development, preview and custom local servers) start at port 5173 and automatically try the first free port after it. Do not pin app-specific ports, enable strictPort, or bake a fixed local origin into SvelteKit. Use the actual printed URL in tests. Internal model services may use independently configured or automatically allocated ports.
- In the Liforma workspace, shared guidance is at `../../AGENTS.md`; resolve sibling repo paths from the actual workspace root. When cloned independently, do not assume the parent workspace exists.
