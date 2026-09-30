## 用途

本项目是「列式分析型数据库引擎」的代码仓库，用于逐步实现该方向的列式存储、查询执行与结果对账能力。

当前实现了**可独立读写的列式文件层**，作为后续查询执行的稳定数据入口；暂不实现 SQL 或执行算子。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

## 命令行入口

安装后提供 `columnar-analytics-engine` 命令：

```bash
columnar-analytics-engine version              # 打印版本号
columnar-analytics-engine inspect <file>       # 以 UTF-8 JSON 输出文件元数据
columnar-analytics-engine --help               # 打印用法
```

`inspect` 输出的顶层键顺序固定为 `format_version`、`row_count`、`columns`，
`columns` 保持 schema 顺序。退出码：

- `0`：成功；
- `1`：路径/权限等系统错误（`OSError`）；
- `2`：文件格式错误（`ColumnarFormatError`），错误信息输出到标准错误。

## Python 公开接口

```python
from columnar_analytics import (
    Field, Schema, Table, WriteOptions,
    write_table, read_table, inspect_file,
    ColumnarFormatError,
    BOOL, INT64, FLOAT64, UTF8,
)

schema = Schema([
    Field("id", INT64, nullable=False),
    Field("name", UTF8),
    Field("score", FLOAT64),
    Field("active", BOOL),
])
table = Table(schema, {
    "id": [1, 2, 3],
    "name": ["ada", None, "grace"],
    "score": [1.5, None, 3.25],
    "active": [True, False, None],
})

write_table("data.col", table,
            WriteOptions(compression="zlib", dict_encoding=["name"]))

restored = read_table("data.col")           # 保持列顺序的表对象
projected = restored.project(["name", "id"])  # 按调用方顺序投影部分列
inspect_file("data.col")                    # 只读元数据
```

### 数据模型约定

- 类型：`bool`、`int64`、`float64`、`utf8`，每列带 `nullable` 标记；
- 列名是非空且唯一的 Unicode 字符串；
- 各列行数必须一致；`None` 只允许出现在可空列中，表示 NULL；
- `float64` 不允许 NaN 或无穷值；
- 投影未知列抛 `KeyError`，投影含重复列抛 `ValueError`。

### 写入选项与元数据

- `compression`：`"none"`（默认）或 `"zlib"`；
- `dict_encoding`：指定需要字典编码的 utf8 列，字典编号按值首次出现顺序确定；
  非 utf8 列使用字典编码抛 `ValueError`；
- 每列元数据含行数、NULL 数，以及排除 NULL 后的 `min`/`max`；
  全 NULL 列的 `min`、`max` 均为 `null`；
- 相同 schema、列值和选项重复写出，文件字节完全一致；
- 写入采用临时文件 + 原子替换：任何校验失败都不会创建或改变目标文件。

### 文件格式

文件为自描述二进制格式，包含魔数 `CAEF` 与格式版本号，整体结构为
魔数 / 版本 / JSON 头帧 / 各列数据帧 / JSON 尾帧 / SHA-256 校验和。

错误魔数、未知版本、截断、校验和不一致、非法元数据等读取失败统一抛出
`ColumnarFormatError`；路径与权限等系统错误保留 `OSError` 语义。

## 限制

- 尚无 SQL、执行算子与索引；列式文件层是当前唯一的持久化能力。
