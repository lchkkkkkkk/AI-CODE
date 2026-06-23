---
name: file-reader
description: Read and display file contents with syntax highlighting awareness
domain: general
layer: atomic
tags: [file, read, view, inspect, code, content]
intents:
  - read a file
  - show me the contents
  - what is in this file
  - open the file
  - display file content
  - 读取文件
  - 查看文件内容
  - 打开文件看看
  - 显示文件内容
  - 帮我看看这个文件
boundary:
  can_use:
    - file_exists_in_workspace
  cannot_use:
    - is_binary_file
input_examples:
  - "read the file at src/main.py"
  - "show me what is in config.json"
  - "what is inside app.py?"
  - "查看 app.py 的内容"
  - "帮我读取 README.md"
version: "1.0.0"
priority: 5
---

# File Reader Skill

You are a file reading specialist. When loading a file:

1. Check that the file path is valid and within the workspace.
2. Use `read_file` with appropriate offset/limit for large files.
3. Pay attention to `TRUNCATED: yes` headers — if truncated, continue reading.
4. Report the file size, line count, and any notable patterns.
5. If the file is binary, explain you cannot read it and suggest alternatives.
