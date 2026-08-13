# Method implementations

每个子目录保存一个可加载的方法实现或一个共享多版本实现。不要根据文件夹名推断当前论文优先级。

- 可执行注册、别名和代码路径：`catalog.py`
- 人工方法索引：[`docs/METHODS.md`](../../../docs/METHODS.md)
- 命令行查看：`python -m survot_rank.cli methods`

历史方法仍保留用于复现。新增方法时先在 `catalog.py` 注册，再补配置、机制文档和测试；不要只增加一个孤立的 `run_*.py`。
