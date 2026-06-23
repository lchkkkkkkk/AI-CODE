# Legacy Skill

This is a legacy skill without YAML frontmatter.

It should still be discoverable and loadable by the skill system.
The router should treat it as having default/empty metadata.

## Behavior
When no frontmatter is present, the system should:
- Use the first non-heading paragraph as the description.
- Assign layer="unknown".
- Assign domain="general".
- Give the skill a default priority of 1.
- Include it in recall with zero intent/tag overlap (but still available).
