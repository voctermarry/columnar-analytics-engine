## 用途

本项目是「列式分析型数据库引擎」的代码仓库，用于逐步实现该方向的列式存储、查询执行与结果对账能力。

当前已实现可独立读写的**列式文件层**，以及面向单个文件的 SQL 查询入口
（`SELECT` 投影 + `WHERE` 过滤）；查询只读取文件、不修改文件。

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
columnar-analytics-engine query <path> "<sql>"          # 对单个文件执行 SQL，输出结果 JSON
columnar-analytics-engine --help                        # 打印用法
```

`inspect` 只读取文件元数据（不读取、不解码列数据），以 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `format_version`、`row_count`、`columns`，columns 保持 schema 顺序。
格式错误时向标准错误输出消息并以码 2 退出；路径等系统错误以码 1 退出。

`query` 对单个文件执行一条 `SELECT ... FROM input [WHERE ...]` 语句，以单行
UTF-8 JSON 输出到标准输出，顶层键依次为 `columns`、`rows`；`columns` 按结果顺序
列出每列的 `name`、`type`、`nullable`，`rows` 是同序值数组的数组。空结果保留列
描述且 `rows` 为空；同一文件与 SQL 重复执行输出字节一致。语法错误
（`QuerySyntaxError`）、校验错误（`QueryValidationError`）与文件格式错误
（`ColumnarFormatError`）向标准错误输出消息并以码 2 退出；系统错误以码 1 退出。

## Python 公开接口

包 `columnar_analytics` 导出：

- `Schema` / `ColumnSchema` / `Table`：有序 schema 与按列数据表
- `write_file(path, table, *, compression="none", dictionary_encoding=())`：确定性、原子写出
- `read_file(path, *, columns=None)`：读回表；`columns` 按调用方顺序投影部分列
- `inspect_file(path)`：只读元数据（行数、每列 NULL 数、min/max）
- `query_file(path, sql)`：对单个文件执行 SQL，成功返回 `Table`
- `ColumnarFormatError`：所有文件格式错误的统一异常；系统错误保留 `OSError` 语义
- `QuerySyntaxError`：SQL 词法/语法错误；`QueryValidationError`：未知列、错误表名、类型不兼容
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

hits = query_file("data.caef", "SELECT id FROM input WHERE name IS NOT NULL")
```

## 单文件 SQL 查询

`query_file(path, sql)` 与 `columnar-analytics-engine query <path> <sql>` 接受以下
SQL 子集（关键字大小写不敏感）：

```sql
SELECT * | 列名 [, 列名 ...]
FROM input
[WHERE 表达式]
```

- 投影只允许星号或逗号分隔的列名，不支持别名；重复列、未知列抛
  `QueryValidationError`。`FROM` 只接受固定表名 `input`（裸写大小写不敏感）。
  列名与 schema 中的 Unicode 字符精确匹配；需要时可用双引号包裹标识符
  （内部用 `""` 转义一个双引号）。
- `WHERE` 支持括号、`NOT`、`AND`、`OR`、`=`、`!=`、`<`、`<=`、`>`、`>=`、
  `IS NULL`、`IS NOT NULL`；优先级从高到低为 `NOT`、比较、`AND`、`OR`。
  操作数仅限列引用与字面量：`TRUE`/`FALSE`、int64 整数、有限 float64 数
  （均支持前导 `+`/`-`）、单引号 utf8 字符串（内部用 `''` 转义单引号）。
- int64 与 float64 可互比，utf8 只与 utf8 比较，bool 只支持 `=`/`!=`；
  不兼容组合抛 `QueryValidationError`。
- 遵循 SQL 三值逻辑：普通比较遇到 NULL 得 UNKNOWN，逻辑运算继续传播 UNKNOWN，
  只有 TRUE 的行进入结果；省略 `WHERE` 保留全部行。结果列遵循投影顺序，行保持
  文件原始顺序。
- 语法不完整、非法字符或其他未支持的语法统一抛 `QuerySyntaxError`；文件损坏仍抛
  `ColumnarFormatError`，系统错误保留 `OSError`。

## 文件格式概览

小端字节序：魔数 `CAEF` + 单字节格式版本 + uint32 头长度 + UTF-8 JSON 元数据头
（schema、每列偏移/校验/统计）+ 数据段（原始拼接或 zlib 压缩）+ uint32 CRC-32
（覆盖此前全部字节）+ 结束标记 `END1`。相同输入与选项重复写出的字节完全一致。
读取时拒绝错误魔数、未知版本、截断、校验不一致与非法元数据，统一抛
`ColumnarFormatError`。

## 限制

- SQL 仅支持单文件查询：`SELECT`（星号/列名投影）+ 固定表名 `input` + 可选
  `WHERE`；不支持别名、表达式投影、连接、聚合、排序、`LIMIT` 等。
- 压缩仅支持 `none` 与 `zlib`；字典编码仅可用于 utf8 列。
