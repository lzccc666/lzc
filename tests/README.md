# Test layout

New project-v2 code is tested independently by domain:

- `tests/nuscenes`: metadata selection and data-manifest tools.
- `tests/navigation`: stage-2 and stage-5 navigation tests.
- `tests/scenegraph`: stage-3 relationship tests.
- `tests/vlm`: stage-4 semantic-model tests.

Runtime test artifacts belong under `outputs/project_v2/tests`; committed fixtures must be small and generated in temporary directories.
