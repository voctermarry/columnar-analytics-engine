## 用途

本项目是「列式分析型数据库引擎」的代码仓库，用于逐步实现该方向的列式存储、查询执行与结果对账能力。

当前已实现可独立读写的**列式文件层**，作为后续查询的稳定数据入口；尚不包含 SQL 解析与执行算子。

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
columnar-analytics-engine version                       # 打印版本号
columnar-analytics-engine inspect <path>                # 输出文件元数据 JSON
columnar-analytics-engine --help                        # 打印用法
```

`inspect` 只读取文件元数据（不读取、不解码列数据），以 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `format_version`、`row_count`、`columns`，columns 保持 schema 顺序。
格式错误时向标准错误输出消息并以码 2 退出；路径等系统错误以码 1 退出。

## Python 公开接口

包 `columnar_analytics` 导出：

- `Schema` / `ColumnSchema` / `Table`：有序 schema 与按列数据表
- `write_file(path, table, *, compression="none", dictionary_encoding=())`：确定性、原子写出
- `read_file(path, *, columns=None)`：读回表；`columns` 按调用方顺序投影部分列
- `inspect_file(path)`：只读元数据（行数、每列 NULL 数、min/max）
- `ColumnarFormatError`：所有文件格式错误的统一异常；系统错误保留 `OSError` 语义
- `FORMAT_VERSION`、`__version__`

Schema 支持 `bool`、`int64`、`float64`、`utf8` 四种类型及 nullable 标记；列名是非空且
唯一的 Unicode 字符串。约束（列缺失/多余、行数不一致、类型不符、非空列出现 `None`、
float64 出现 NaN/无穷值、未知压缩名、对非 utf8 列请求字典编码等）统一抛 `ValueError`，
且失败时不会创建或修改目标文件。投影未知列抛 `KeyError`，投影重复列抛 `ValueError`。

### 示例

```python
from columnar_analytics import ColumnSchema, Schema, Table, write_file, read_file

schema = Schema([
    ColumnSchema("id", "int64"),
    ColumnSchema("name", "utf8", nullable=True),
])
table = Table(schema, {"id": [1, 2], "name": ["ada", None]})
write_file("data.caef", table, compression="zlib", dictionary_encoding=["name"])

restored = read_file("data.caef")                    # 保持列顺序
subset = read_file("data.caef", columns=["name"])    # 按给定顺序投影
```

## 文件格式概览

小端字节序：魔数 `CAEF` + 单字节格式版本 + uint32 头长度 + UTF-8 JSON 元数据头
（schema、每列偏移/校验/统计）+ 数据段（原始拼接或 zlib 压缩）+ uint32 CRC-32
（覆盖此前全部字节）+ 结束标记 `END1`。相同输入与选项重复写出的字节完全一致。
读取时拒绝错误魔数、未知版本、截断、校验不一致与非法元数据，统一抛
`ColumnarFormatError`。

## 限制

- 尚无 SQL 与查询执行算子。
- 压缩仅支持 `none` 与 `zlib`；字典编码仅可用于 utf8 列。
