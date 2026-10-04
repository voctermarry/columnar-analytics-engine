## 用途

本项目是「列式分析型数据库引擎」的代码仓库，用于逐步实现该方向的列式存储、查询执行与结果对账能力。

当前已实现可独立读写的**列式文件层**，以及面向单个文件的 SQL 查询入口
（`SELECT` 投影、`SELECT DISTINCT` 结果行去重、`WHERE` 过滤、`GROUP BY` 分组与
`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`
聚合（均支持参数级 `DISTINCT`，如 `COUNT(DISTINCT 列)`）、分组后 `HAVING` 过滤、`ORDER BY` 稳定排序与 `LIMIT` Top-N）；查询只读取文件、不修改文件。
另提供多文件查询入口，支持在 FROM 后确定性地连续连接零个或多个表，每步为
`INNER JOIN` / `LEFT JOIN` / `RIGHT JOIN` / `FULL OUTER JOIN` 等值连接。
对应的 `explain_file` / `explain_files`（命令行 `explain` / `explain-files`）
只解析、绑定并生成逻辑计划，仅读文件元数据、不执行查询。
查询结果还可通过 `export_query_file` / `export_query_files`
（命令行 `export` / `export-files`）直接导出为 CSV 或 JSONL 复核文件。

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
columnar-analytics-engine query-files <sources-json> "<sql>" [--join-strategy hash|sort_merge]   # 对映射的多表执行 SQL（可含连续连接）
columnar-analytics-engine explain <path> "<sql>"        # 只解析/绑定并输出单文件语句的计划 JSON
columnar-analytics-engine explain-files <sources-json> "<sql>" [--join-strategy hash|sort_merge]  # 只解析/绑定并输出多表语句的计划 JSON
columnar-analytics-engine export <path> "<sql>" <dest> [--format csv|jsonl]   # 执行单文件查询并导出结果文件
columnar-analytics-engine export-files <sources-json> "<sql>" <dest> [--format csv|jsonl] [--join-strategy hash|sort_merge]  # 执行多表查询并导出
columnar-analytics-engine --help                        # 打印用法
```

`inspect` 只读取文件元数据（不读取、不解码列数据），以 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `format_version`、`row_count`、`columns`，columns 保持 schema 顺序。
格式错误时向标准错误输出消息并以码 2 退出；路径等系统错误以码 1 退出。

`query` 对单个文件执行一条
`SELECT [DISTINCT] ... FROM input [WHERE ...] [GROUP BY ... [HAVING ...]] [ORDER BY ...] [LIMIT n]` 语句，以单行
UTF-8 JSON 输出到标准输出，顶层键依次为 `columns`、`rows`；`columns` 按结果顺序
列出每列的 `name`、`type`、`nullable`，`rows` 是同序值数组的数组。空结果保留列
描述且 `rows` 为空（无 GROUP BY 的聚合查询在零入选行时仍返回一行，COUNT 为 0、
其余聚合为 null，该行再经 HAVING 过滤；有 GROUP BY 时返回零行）；同一文件与 SQL 重复执行输出字节一致。语法错误
（`QuerySyntaxError`）、校验错误（`QueryValidationError`）与文件格式错误
（`ColumnarFormatError`）向标准错误输出消息并以码 2 退出；系统错误以码 1 退出。

`query-files` 的 `<sources-json>` 是一个 JSON 对象，把表名映射到列式文件路径，例如
`'{"l": "left.caef", "r": "right.caef"}'`；SQL 中的 `FROM`/`JOIN` 表名即取这些键。
输出格式与退出码约定同 `query`；sources-json 本身非法（不是 JSON 对象、键不是非空
字符串、值不是路径）也以码 2 退出。可选的 `--join-strategy` 接受 `hash` 或
`sort_merge`：前者以右文件键构建哈希索引，后者把两侧按键稳定排序后归并等值键段；
两种策略返回完全相同的列描述、值、NULL 位置与行序。未提供时沿用引擎默认连接路径
（与 `hash` 相同的结果），不含 JOIN 的语句也可指定任一策略但不产生额外算子。
`--join-strategy` 取值非法（不是 `hash`/`sort_merge` 字符串）时以码 2 失败、
标准输出为空、标准错误只写异常消息，且在打开任何源文件之前判定。

`explain` / `explain-files` 与对应的查询入口接受完全相同的输入与 SQL 子集，但只做
解析、绑定与计划生成：不执行查询，只读取文件元数据头（不读取、不解压、不解码数据
段），也不会打开未被语句引用的 sources。成功时以单行紧凑 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `sources`、`operators`、`output`：

- `sources` 按 `FROM`、`JOIN` 顺序列出，每项含 `name`、`row_count` 与按源 schema
  顺序的 `columns`（每列 `name`、`type`、`nullable`）。
- `operators` 依次包含实际存在的阶段：有连接时所有 `Scan`（按 FROM/JOIN 顺序）
  先列出，然后每步一个同序 `Join`，之后依次为 `Filter`、
  `Aggregate`、`Having`、`Sort`、`Limit`、`Project`；`SELECT DISTINCT` 语句在扫描、
  连接与过滤阶段之后按 `Project`、`Distinct`、`Sort`、`Limit` 的顺序给出；缺少的阶段省略：
  - 每个被引用源一个 `Scan`，`required_columns` 按源 schema 顺序给出语句引用的列
    （投影、WHERE、GROUP BY、HAVING 分组列与聚合参数、ORDER BY 及连接键；仅 `COUNT(*)` 时为空）。
    单源语句与全 INNER 连接链中的 v2 源，其 `Scan` 在 `required_columns` 之后依次携带
    `row_groups_total`、`row_groups_selected` 与 `pushed_condition`（归属该源的下推叶子
    按 SQL 出现顺序组合成的条件树，保留限定列名；无合格叶子时选中数等于总数且
    `pushed_condition` 为 null）；v1 源与含外连接链的 `Scan` 不增加这些字段。
  - 每个连接步骤一个 `Join`（与对应步骤同序），给出 `type`（`INNER`/`LEFT`/`RIGHT`/`FULL`）
    与 `left`、`right` 两侧限定键（`table`、`column`；左侧为该步之前已引入的表，右侧为该步
    新引入的表，与 ON 书写方向无关）；显式传入 `join_strategy` 时每个 `Join` 额外给出
    `strategy`（`HASH`/`SORT_MERGE`），且与对应查询、导出实际使用的策略一致；未传策略时
    `Join` 算子保持无 `strategy` 字段的结构。
  - `Filter` 的 `condition` 是递归表达式树：内部节点含 `kind`、`operator`、`operands`，
    叶子是带 `type` 的 `literal`（`value` 为类型化字面量）或带 `name` 的绑定 `column`。
    searched CASE 节点的 `kind` 为 `case`，`cases` 按书写顺序给出
    `{when, then}`（两者均为递归表达式），`else` 为递归表达式或 `null`（省略 ELSE）。
    `IN` / `NOT IN` 谓词节点的 `kind` 为 `in`，携带 `negated` 布尔值、`operand`
    左侧表达式与按书写顺序排列的 `options` 类型化字面量数组（列表内 NULL 为
    `{"kind":"literal","type":null,"value":null}`）。
  - `Aggregate` 给出 `group_keys`（绑定列名列表，无 GROUP BY 时为空）与 `aggregates`
    （每项 `function`、`argument`（`COUNT(*)` 为 null）、`output`；DISTINCT 聚合
    额外给出 `distinct: true`，普通聚合不增加该字段）；`aggregates`
    收录 SELECT、ORDER BY、HAVING 所需的去重聚合，按首次引用顺序排列。
  - `Having`（仅含 HAVING 子句时）位于 `Aggregate` 之后、`Sort` 之前，`condition`
    沿用递归条件树；分组列叶子为带 `name` 的 `column`，聚合叶子额外给出
    `function`、`argument`（`COUNT(*)` 为 null）、`type` 与 `nullable`
    （DISTINCT 聚合叶子同样额外给出 `distinct: true`）。
  - `Sort` 的 `keys` 每项给出 `column`、`direction`（`ASC`/`DESC`）与
    `nulls`（`FIRST`/`LAST`）；`Limit` 给出 `count`；`Project` 的 `expressions`
    每项给出绑定表达式（列、聚合或递归标量表达式）与 `output` 输出名。
  - `Distinct`（仅 `SELECT DISTINCT`）位于 `Project` 之后、`Sort`/`Limit` 之前，
    `keys` 按结果列顺序给出投影输出名；DISTINCT 计划的顶层 `output` 与去重后的查询结果
    schema 一致。
- `output` 按结果顺序给出 `name`、`type`、`nullable`，与对应查询结果的 schema 一致。

相同元数据与 SQL 重复解释输出字节一致；关键字大小写与多余空白不改变计划内容。
语法错误在访问任何文件之前抛 `QuerySyntaxError`；非法 sources 抛 `ValueError`；
未知表/列、类型不兼容及其他绑定错误抛 `QueryValidationError`；非法元数据或声明尺寸
不符抛 `ColumnarFormatError`；系统访问失败保留 `OSError`（数据段 CRC 与值级统计因
未读取而不校验）。命令行对 `ValueError`、查询错误与格式错误以码 2 退出，对
`OSError` 以码 1 退出，失败时标准输出为空。

## 查询结果导出

`export` / `export-files` 与对应查询入口接受完全相同的源与 SQL（不增加查询语法），
把结果直接写成可复核文件，位置参数最后为目标路径，另加 `--format`（仅接受
`csv` 或 `jsonl`，默认 `csv`）；`export-files` 还支持
`--join-strategy hash|sort_merge`，语义与 `query-files` 完全相同，两种策略导出的
文件字节一致，策略非法时在打开任何源文件前以码 2 失败、不创建或改变目标文件。
成功时状态码为 0 且标准输出、标准错误均为空；
Python 入口 `export_query_file(path, sql, destination, format="csv")` 与
`export_query_files(sources, sql, destination, format="csv", join_strategy=None)`
成功时返回写出的结果行数。

- **CSV**：UTF-8 无 BOM、统一 LF 换行；第一行始终按结果 schema 顺序写列名
  （列名或字段含逗号、双引号、CR、LF 时同样按规则加引号）。字段含逗号、双引号、
  CR 或 LF 时整体加双引号，内部双引号重复一次；NULL 写成空的未加引号字段，空字符串
  固定写成 `""` 包围的空字段以与 NULL 区分；bool 写为 `true` / `false`；数值采用
  紧凑 JSON 结果的文本形式。零行结果只有表头行（仍以 LF 结束）。
- **JSONL**：每个结果行写一个紧凑 JSON 对象，键按结果列顺序排列，值保持查询 JSON
  的类型与 `null`，非 ASCII 字符不转义，每行以一个 LF 结束；零行结果为零字节文件。
- 未写 `ORDER BY` 时沿用查询本身的确定顺序；相同输入、SQL、格式与版本重复导出
  字节一致。
- 只有在查询完整成功且全部内容编码完成后才原子替换目标文件：任何失败都不会新建目标，
  也不会改变已有目标（临时文件与目标同目录、清理后不可见）。
- 目标路径与任一**实际引用**的源文件解析为同一路径时抛 `ValueError`（未被语句引用
  的 sources 不受此限制）；未知格式也抛 `ValueError`。SQL、绑定与列式格式问题分别
  抛 `QuerySyntaxError`、`QueryValidationError`、`ColumnarFormatError`；
  `sources` 校验仍抛 `ValueError`；目录不存在、权限不足等文件系统问题保留 `OSError`。
  命令行把 `ValueError` 与三类查询错误映射为状态码 2，把 `OSError` 映射为状态码 1，
  失败时标准输出为空、标准错误只写异常消息（`export-files` 的 sources-json 非法同样
  以码 2 退出）。

## Python 公开接口

包 `columnar_analytics` 导出：

- `Schema` / `ColumnSchema` / `Table`：有序 schema 与按列数据表
- `write_file(path, table, *, compression="none", dictionary_encoding=())`：确定性、原子写出（v1 格式）
- `write_partitioned_file(path, table, row_group_size, *, compression="none", dictionary_encoding=())`：
  按原始行序切成连续行组、确定性原子写出 v2 格式；每组记录行数与各列
  `null_count`/`min`/`max`，各列块独立校验、解压、解码；`row_group_size`
  非正整数或编码参数非法时抛 `ValueError` 且不创建或改变目标文件
- `read_file(path, *, columns=None)`：读回表（同时支持 v1/v2）；`columns` 按调用方顺序投影部分列，
  读取 v2 时只解压、解码被请求的列块
- `inspect_file(path)`：只读元数据（格式版本、行数、每列 NULL 数、min/max；v2 为各组统计的汇总）
- `inspect_row_groups(path)`：只读元数据，按文件顺序返回 v2 各行组的行数与 schema 顺序的列统计；
  v1 文件返回空列表
- `query_file(path, sql)`：对单个文件执行 SQL，成功返回 `Table`
- `query_files(sources, sql, join_strategy=None)`：对表名→路径映射执行 SQL（FROM 后可连续连接多个表），成功返回 `Table`；
  `join_strategy` 可选 `"hash"` / `"sort_merge"`，显式指定时应用于每一步，两种策略结果与行序完全相同，
  省略时使用默认连接路径
- `explain_file(path, sql)`：只读元数据，返回单文件语句的有序计划字典（键为
  `sources`、`operators`、`output`），不执行查询、不读数据段
- `explain_files(sources, sql, join_strategy=None)`：同上，面向多表语句；未被引用的 sources 不会被打开；
  sources/Scans 与每步一个的 `Join` 按 FROM/JOIN 顺序排列，显式传入 `join_strategy` 时每个
  `Join` 算子带 `strategy` 字段（`HASH`/`SORT_MERGE`），
  无 JOIN 时策略不产生算子；非法策略在打开任何源文件前抛 `ValueError`
- `export_query_file(path, sql, destination, format="csv")`：执行单文件查询并把结果
  原子导出到 `destination`（`csv` 或 `jsonl`，默认 `csv`），成功返回写出行数
- `export_query_files(sources, sql, destination, format="csv", join_strategy=None)`：对表名→路径映射执行
  查询并导出；目标与任一实际引用源同路径、格式未知或 `join_strategy` 非法时抛 `ValueError`，其他异常沿用
  查询入口的分类
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

hits = query_file(
    "data.caef",
    "SELECT id FROM input WHERE name IS NOT NULL ORDER BY id DESC LIMIT 10",
)

counts = query_file(
    "data.caef",
    "SELECT name, COUNT(*), SUM(id), AVG(id) FROM input "
    "GROUP BY name ORDER BY COUNT(*) DESC NULLS LAST LIMIT 10",
)
```

## 单文件 SQL 查询

`query_file(path, sql)` 与 `columnar-analytics-engine query <path> <sql>` 接受以下
SQL 子集（关键字大小写不敏感）：

```sql
SELECT [DISTINCT] * | 投影项 [, 投影项 ...]
FROM input
[WHERE 表达式]
[GROUP BY 列名 [, 列名 ...]]
[HAVING 分组条件]
[ORDER BY 排序项 [, ...]]
[LIMIT 无符号整数]

投影项 := 列名
        | COUNT (*)
        | COUNT | SUM | AVG | MIN | MAX ([DISTINCT] 列名)
        | 标量表达式 AS 别名
排序项 := (列名 | 聚合调用 | SELECT 别名) [ASC | DESC] [NULLS FIRST | NULLS LAST]
标量表达式 := 数值列 | 数值字面量 | (标量表达式)
           | +标量表达式 | -标量表达式
           | 标量表达式 + - * / 标量表达式   （优先级：括号、一元、乘除、加减）
           | CASE WHEN 条件 THEN 标量表达式
             {WHEN 条件 THEN 标量表达式} [ELSE 标量表达式] END
```

- 投影允许星号、逗号分隔的列名、聚合调用或带 `AS` 别名的标量表达式；算术表达式操作数须为
  数值，`CASE` 表达式还可产生 bool 或 utf8 结果。别名沿用
  标识符规则（可用双引号包裹），且不得与其他输出名重复。裸列与聚合调用不支持别名、
  名称保持不变；重复列、重复别名、重复输出名、未知列抛 `QueryValidationError`。
  `FROM` 只接受固定表名 `input`（裸写大小写不敏感）。
  列名与 schema 中的 Unicode 字符精确匹配；需要时可用双引号包裹标识符
  （内部用 `""` 转义一个双引号）。
- `SELECT DISTINCT` 对非聚合投影的完整结果行去重：投影项仍只能是星号、裸列或带
  `AS` 别名的标量表达式（不支持聚合），去重在 `WHERE` 过滤之后按结果 schema 逐列比较，
  全部列相等才算重复。两个 NULL 在同一列判为相等、NULL 与非 NULL 不等；bool、utf8 与
  数值沿用现有类型语义，float64 的 `0.0` 与 `-0.0` 视为相等。省略 `ORDER BY` 时每种
  不同结果行按其第一次出现的顺序保留；随后执行 `ORDER BY` 与 `LIMIT`（`LIMIT 0` 与
  空结果仍保留列描述）。DISTINCT 的执行顺序为 `WHERE` → 投影 → 去重 → 排序 → `LIMIT`：
  投影表达式只对通过 WHERE 的行求值，实际发生的除零、int64 溢出或非有限 float64 结果仍抛
  `QueryValidationError`；被排序或 LIMIT 截掉的行也会完成投影求值。DISTINCT 查询的
  `ORDER BY` 只能引用投影中的裸列或显式别名，引用未投影列或未知名称抛
  `QueryValidationError`，排序方向、NULL 位置与稳定性维持现状。DISTINCT 与
  `GROUP BY`、`HAVING` 或任何聚合投影同时使用抛 `QueryValidationError`；
  聚合参数内的 `DISTINCT`（`COUNT(DISTINCT ...)`) 等）是另一特性，见下文聚合条目；
  DISTINCT 缺少投影、重复出现或位置错误抛
  `QuerySyntaxError`，且在访问文件之前判定。连接查询中 DISTINCT 沿用限定列名规则，
  星号结果的列名仍为 `表名.列名`。
- 标量表达式支持括号、一元 `+`/`-` 与二元 `+`、`-`、`*`、`/`，操作数为 int64/float64
  列与数值字面量。两个 int64 相加/减/乘结果为 int64；任一操作数为 float64 或执行除法
  时结果为 float64；一元运算保留类型。任一操作数为 NULL 时结果为 NULL；输出列的
  nullable 由参与列推导，纯常量表达式非空。int64 运算越界、除零、float64 结果非有限
  时抛 `QueryValidationError`。非聚合查询的执行顺序为 `WHERE` → 计算排序键 → 稳定排序
  → `LIMIT` → 结果表达式，因此被过滤或截掉的行不会触发 SELECT 中的除零或溢出。
  `ORDER BY` 可引用 SELECT 的显式别名（别名与输入列同名时优先解析别名），并按表达式
  的结果类型、NULL 位置与稳定排序规则排序。聚合查询不扩展：GROUP BY 键与聚合参数仍只
  接受列引用或 `COUNT(*)`，聚合查询混入标量表达式时抛 `QueryValidationError`。
- 标量表达式可使用 searched `CASE`（不支持 simple CASE）：
  `CASE WHEN 条件 THEN 结果 [WHEN ...] [ELSE 结果] END`，至少一个 WHEN，`CASE`/
  `WHEN`/`THEN`/`ELSE`/`END` 大小写不敏感且可递归嵌套，能出现在非聚合查询的 SELECT 表达式、
  算术运算数以及 WHERE 比较两侧，SELECT 中仍以 `AS` 命名并可由 ORDER BY 引用。每行按书写顺序
  判断条件，仅 TRUE 命中，FALSE 与 UNKNOWN 继续向下；均未命中时取显式 ELSE，省略 ELSE 返回 NULL。
  只计算命中的结果，未命中分支的除零、int64 溢出、非有限 float64 不报错，实际命中时仍抛
  `QueryValidationError`。所有可达 THEN 与显式 ELSE 的类型须一致：int64 与 float64 可混合并
  统一为 float64，bool、utf8 与数值或彼此混合抛 `QueryValidationError`。任一可达结果可空或省略
  ELSE 时结果列 nullable，否则非空（NULL 条件只改变走向，不使非空结果变空）。WHEN 条件沿用现有
  布尔类型检查与三值逻辑；CASE 在聚合参数、GROUP BY 及聚合查询投影中仍按非法位置抛
  `QueryValidationError`。缺失关键字、空分支、孤立关键字或嵌套未闭合均在访问文件前抛
  `QuerySyntaxError`。
- 聚合函数为 `COUNT(*)`、`COUNT(列)`、`SUM(列)`、`AVG(列)`、`MIN(列)`、`MAX(列)`，
  函数名大小写不敏感，结果列名为大写函数名加括号（列引用使用 schema 真实名）。
  `COUNT(*)` 计入选行，`COUNT(列)` 忽略 NULL，二者均为非空 int64；`SUM`/`AVG`
  只接受 int64/float64 且忽略 NULL，`SUM` 保持输入类型、`AVG` 输出 float64；
  `MIN`/`MAX` 接受现有四类并沿用既有比较规则。除 COUNT 外的聚合在没有非 NULL
  输入时返回 NULL，结果列 nullable。int64 求和越界或浮点聚合得到非有限值抛
  `QueryValidationError`。
- `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` 的参数前可加 `DISTINCT`（如 `COUNT(DISTINCT 列)`）：
  先按 WHERE 选行，再在每个分组（或唯一的全局分组）内忽略 NULL 并按列类型对非 NULL
  参数值去重（float64 的 `0.0` 与 `-0.0` 视为同一值），然后执行聚合。
  `COUNT(DISTINCT 列)` 返回不同非 NULL 值的数量，空输入返回 0 且为非空 int64；其余
  DISTINCT 聚合在没有不同非 NULL 值时返回 NULL，结果类型、可空性、比较规则以及
  SUM 溢出或非有限浮点的 `QueryValidationError` 与对应普通聚合一致。参数级 DISTINCT
  只作用于该调用的参数值，不改变 `SELECT DISTINCT` 的整行去重；参数仍只能是单列引用
  （连接查询中带表限定），星号、多参数、表达式与嵌套聚合均不接受。DISTINCT 位置错误、
  缺参数、用于星号、表达式或多参数时在访问文件前抛 `QuerySyntaxError`；未知列、类型
  不兼容与既有绑定冲突抛 `QueryValidationError`。DISTINCT 聚合可出现在 SELECT、HAVING
  与 ORDER BY 允许普通聚合出现的位置（HAVING 中未投影的调用仍过滤分组，ORDER BY 仍只能
  引用已选聚合），文本相同的调用按现有聚合去重规则只计算一次；输出名使用大写函数名与
  `DISTINCT` 关键字并保留绑定后的真实列名（如 `COUNT(DISTINCT x)`）。
- 无 `GROUP BY` 时投影只能包含聚合表达式（筛选为空也返回一行：COUNT 为 0、其余为
  NULL）；有 `GROUP BY` 时可投影分组列与聚合，普通列必须已分组，星号不得与聚合或
  分组混用（抛 `QueryValidationError`）。分组按 GROUP BY 列顺序成键，NULL 键归为
  一组；没有 ORDER BY 时各分组按首条入选行顺序输出，筛选为空时返回零行。
- `HAVING` 位于 GROUP BY 之后、ORDER BY 之前，对已形成的分组做三值逻辑过滤：条件
  用现有 `NOT`、`AND`、`OR`、比较、`IS [NOT] NULL` 与 `[NOT] IN` 语义组合**分组列、聚合调用与
  类型兼容的字面量**，只有条件为 TRUE 的组保留，FALSE 与 UNKNOWN 均丢弃。`IN` 左侧仍只允许
  分组列、聚合调用或字面量，列表规则与 WHERE 相同；HAVING
  中的聚合不必出现在 SELECT（仍按聚合现有 NULL、空输入、数值异常与类型规则计算并
  参与去重）；不解析 SELECT 别名，不接受未分组普通列、星号、CASE、标量算术或嵌套
  聚合，最终条件非 bool、类型不兼容或聚合参数非法抛 `QueryValidationError`。完全
  非聚合查询使用 HAVING 抛 `QueryValidationError`；HAVING 缺条件、重复、错位（如
  位于 LIMIT 后）在访问文件前抛 `QuerySyntaxError`。无 GROUP BY 时 HAVING 过滤
  那一行全局聚合（WHERE 零入选行时仍先生成该行再过滤）；有 GROUP BY 且无入选行时
  返回零组。执行顺序固定为 `WHERE` → 分组聚合 → HAVING → 排序 → `LIMIT` → 投影，
  因此被 HAVING 丢弃的组不参与排序与 LIMIT，结果与导出行数以过滤后为准。
- `ORDER BY` 在非聚合查询中可引用任意 schema 列（无需出现在投影中）；在聚合查询中
  只能引用已选分组列或已选聚合表达式（按 schema 真实名匹配）。未知列、同一
  `ORDER BY` 中的重复项或引用未选结果抛 `QueryValidationError`。多列按书写顺序比较：
  int64/float64 按数值、utf8 按 Unicode 码点、bool 按 `FALSE < TRUE`；全部排序键
  相等的行（或分组）保持其原始（首条入选行）相对顺序。省略方向为 `ASC`；省略
  `NULLS` 时不论方向 NULL 都排在末尾，显式 `NULLS FIRST`/`NULLS LAST` 覆盖默认值。
- `LIMIT` 只接受 `0` 至 `9223372036854775807` 的无符号十进制整数，可独立出现；
  `0` 返回保留列描述的空表，大于入选行数（或分组数、不同结果行数）时返回全部。非聚合查询的执行
  顺序固定为 `WHERE` → 排序 → `LIMIT` → 投影；DISTINCT 查询为
  `WHERE` → 投影 → 去重 → 排序 → `LIMIT`；聚合查询为
  `WHERE` → 分组/聚合 → HAVING → 排序 → `LIMIT` → 投影（无 HAVING 时跳过该阶段）。
- `WHERE` 支持括号、`NOT`、`AND`、`OR`、`=`、`!=`、`<`、`<=`、`>`、`>=`、
  `IS NULL`、`IS NOT NULL`、`IN`、`NOT IN`；优先级从高到低为 `NOT`、比较
  （含 `[NOT] IN`）、`AND`、`OR`。
  比较两侧可使用列引用、字面量或数值标量表达式（含 `CASE`）：字面量为 `TRUE`/`FALSE`、int64 整数、
  有限 float64 数（均支持前导 `+`/`-`）、单引号 utf8 字符串（内部用 `''` 转义单引号）；
  结果为 bool 的 `CASE`（或其他布尔表达式）可直接充当条件，数值表达式直接充当布尔条件抛
  `QueryValidationError`；
  WHERE 内不允许聚合（抛 `QueryValidationError`）。
- `IN` / `NOT IN` 的形式为 `左侧值表达式 [NOT] IN (常量列表)`：左侧沿用 WHERE 与
  CASE WHEN 中现有的列、字面量及标量表达式（HAVING 中仍只允许分组列、聚合调用与字面量，
  不允许 CASE 或算术）；列表至少一项，只允许 bool、int64、float64、utf8 字面量以及
  仅能写在列表内的 `NULL`，不接受列引用、表达式或聚合调用，重复项不改变结果。绑定时左侧按
  现有比较规则与每个非 NULL 项检查类型：int64 与 float64 可混合，bool 与 utf8 只能同类，
  类型不兼容抛 `QueryValidationError`；全 NULL 列表合法。三值逻辑：左侧为 NULL 时结果为
  UNKNOWN；否则任一非 NULL 项相等时 `IN` 为 TRUE、`NOT IN` 为 FALSE；没有匹配但列表含
  NULL 时为 UNKNOWN，其余情况分别为 FALSE 与 TRUE。左侧表达式既有的除零、int64 溢出与
  非有限 float64 结果仍抛 `QueryValidationError`。缺括号、空列表、尾随逗号、非法列表项或
  `NOT` 与 `IN` 次序错误均在访问文件前抛 `QuerySyntaxError`。explain 的递归条件树为该谓词
  输出 `kind` 为 `in`、`negated` 布尔值、`operand` 左侧表达式和按书写顺序排列的 `options`
  类型化字面量数组（NULL 选项渲染为 `{"kind":"literal","type":null,"value":null}`）。
- int64 与 float64 可互比，utf8 只与 utf8 比较，bool 只支持 `=`/`!=`；
  不兼容组合抛 `QueryValidationError`。
- 遵循 SQL 三值逻辑：普通比较遇到 NULL 得 UNKNOWN，逻辑运算继续传播 UNKNOWN，
  只有 TRUE 的行进入分组与结果；省略 `WHERE` 保留全部行。
- 语法不完整、聚合括号或参数个数错误、`GROUP BY` 空列表、修饰词错位或重复、
  子句乱序或重复（GROUP BY 须位于 WHERE 之后、ORDER BY 之前）、`LIMIT`
  缺值/负数/小数/越界等统一抛 `QuerySyntaxError`，且在访问文件之前识别；
  文件损坏仍抛 `ColumnarFormatError`，系统错误保留 `OSError`。

## 连续连接查询

`query_files(sources, sql)` 与 `columnar-analytics-engine query-files <sources-json> <sql>`
在单文件语法基础上支持确定性的连续连接；`sources` 是表名到文件路径的非空映射
（命令行用同一结构的 JSON 对象），只有被语句引用的表对应的文件会被读取。FROM 表之后
可连续出现零个或多个连接子句，每步引入一个此前未使用的表：

```sql
SELECT ... FROM 起始表
  {INNER JOIN | LEFT JOIN | RIGHT JOIN | FULL OUTER JOIN} 新表
  ON (此前任一表.列 = 新表.列 | 新表.列 = 此前任一表.列)
  [ ...更多连接步骤... ]
[WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT n]
```

- 不支持别名、重复表、复合 `ON`、非等值 `ON`、其他连接类型、`FULL` 后缺 `OUTER`
  （`RIGHT`/`LEFT` 后不接 `OUTER`）；词法错误、连接关键字残缺、缺少 `ON`、非法 `ON`
  抛 `QuerySyntaxError`，且在读取任何源文件之前判定。`RIGHT`、`FULL`、`OUTER` 仅在
  连接子句位置具有关键字含义，其他位置（列名、表名）仍按普通标识符处理，单文件
  查询与既有行为完全一致。
- 连接查询中除 `COUNT(*)` 外的列引用都必须写成 `表名.列名`（两部分都可用双引号
  标识符，延续精确匹配与 `""` 转义语义）；未限定列、未知或重复表、未知列、ON 未把
  新表与此前任一表连接（同一侧或两表都不是新表）、键类型不兼容、重复结果列抛
  `QueryValidationError`。ON 等号两侧可交换。非连接查询沿用单表语义，列可限定也可
  不限定。
- `SELECT *` 按 FROM/JOIN 顺序展开所有引用表的 schema，列名为 `表名.列名`；显式投影
  保留限定名，聚合结果列名沿用大写函数格式并含限定参数（如 `SUM(r.x)`）。
- 连接键类型须相同，或一为 int64 一为 float64；NULL 键永不匹配，重复键产生完整
  组合。每一步以当前中间结果为左、新表为右：`INNER JOIN` 输出全部匹配组合；
  `LEFT JOIN` 还输出未匹配的当前行并把新表一侧值置为 NULL（新表结果列 nullable 为
  true）；`RIGHT JOIN` 按新表文件原始行序输出全部新行，同一新行的匹配组合按当前行序
  展开，未匹配时把此前所有表的列补 NULL（中间结果各列均变为 nullable）；
  `FULL OUTER JOIN` 先按 LEFT JOIN 顺序输出匹配组合与未匹配当前行，再按新表文件原始
  行序追加未匹配新行（全部结果列 nullable）。INNER/LEFT/FULL 按当前行序展开、同一
  当前行内按新表文件行序展开匹配组合；逐步推导 nullable（某步变 nullable 的列在后续
  步骤保持 nullable）。全部连接完成后再执行 `WHERE`、`DISTINCT` 去重、`GROUP BY`、
  聚合、`HAVING`、`ORDER BY`、`LIMIT`，补出的 NULL 参与既有三值逻辑、分组与排序；
  没有显式排序时相同输入与 SQL 重复执行输出字节一致。
- 多文件 Python 入口（`query_files`、`explain_files`）与命令行（`query-files`、
  `explain-files`、`export-files` 的 `--join-strategy`）可选指定连接算法：
  `hash` 对右侧非空键建哈希索引，`sort_merge` 将两侧按键稳定排序后归并等值键段；
  显式策略应用于每一个连接步骤。两种策略接受完全相同的 INNER/LEFT/RIGHT/FULL 等值
  连接、键类型兼容、NULL 与重复键规则，对相同 sources 与 SQL 返回的列描述、值、
  NULL 位置和行序完全一致（int64/float64 混合键、正负零、重复值、可空键均无差异），
  查询 JSON 与导出文件重复执行字节稳定。未传参数时继续使用默认连接路径，计划中的
  `Join` 无 `strategy` 字段；无 JOIN 的合法语句可传任一策略但不产生额外算子。显式传
  策略时，explain 每个 `Join` 算子增加 `strategy` 字段（值为 `HASH` 或
  `SORT_MERGE`），连接类型分别显示 `INNER` / `LEFT` / `RIGHT` / `FULL`，`left`/`right`
  分别记录该步此前表与新表的限定键，并与查询、导出实际策略一致；解释仍只读被引用
  源元数据，sources 与 Scans 按 FROM/JOIN 顺序排列、`required_columns` 归属各源。
  参数不是字符串或不是两个允许值之一时，Python 入口在打开任何源文件前抛
  `ValueError`，命令行以状态码 2 失败、标准输出为空、标准错误只写异常消息。
- `sources` 非映射或为空、键不是非空字符串、值不是路径对象时抛 `ValueError`，
  且不会访问任何文件；导出时目标路径与任一实际引用源同路径也抛 `ValueError`，且仅在
  查询与编码成功后原子替换目标；已引用文件损坏仍抛 `ColumnarFormatError`，系统错误
  保留 `OSError`。

## 文件格式概览

小端字节序：魔数 `CAEF` + 单字节格式版本 + uint32 头长度 + UTF-8 JSON 元数据头
（schema、每列偏移/校验/统计）+ 数据段（原始拼接或 zlib 压缩）+ uint32 CRC-32
（覆盖此前全部字节）+ 结束标记 `END1`。相同输入与选项重复写出的字节完全一致。
读取时拒绝错误魔数、未知版本、截断、校验不一致与非法元数据，统一抛
`ColumnarFormatError`。

格式版本 2（`write_partitioned_file`）把行按原始顺序切成定长连续行组，头部记录
每组行数及每列 `null_count`/`min`/`max` 统计；每个（行组 × 列）块独立存放、
独立 CRC-32 校验，可按列独立 zlib 压缩与解码，尾部为裸 `END1` 标记。
`read_file` / `inspect_file` 同时识别两个版本；`inspect_row_groups` 按文件顺序
返回 v2 各组行数与 schema 顺序的列统计（v1 返回空列表）。读取 v2 时只解压、
解码实际请求的列块；单文件查询、不含 JOIN 的多文件查询以及整条连接链均为
INNER JOIN 的多文件查询，还会把 WHERE 顶层 AND 中只引用一个来源的合格叶子
（限定列与类型兼容字面量间的 `=`/`!=`/`<`/`<=`/`>`/`>=`，字面量在任一侧等价，
该限定列的 `IS [NOT] NULL`，以及「绑定列 `IN` 常量列表」——统计证明组内不存在
任何非 NULL 候选值时跳过该组，列表中的 NULL 不参与判定）按来源分别下推到组统计——同一来源的多个叶子按
AND 合并，仅当统计能证明它们不可能在该组某行同时为 TRUE 时才跳过整组
（OR、NOT、CASE、`NOT IN`、表达式左侧的 IN、算术、列间或跨来源比较、统计无法判定的范围不参与裁剪，也不阻止
同级合格叶子下推）；链中含任一 LEFT/RIGHT/FULL OUTER JOIN 时不做行组裁剪，
v1 源始终完整读取，混合 v1/v2 的全 INNER 链只裁剪 v2 源。被排除的行组即使其
数据块损坏也不会被读取，入选组的损坏块抛 `ColumnarFormatError`；两种版本对同一
数据和语句返回完全相同的结果，完整 WHERE 仍对连接结果逐行求值。explain 计划中
合格的 v2 `Scan` 算子在 `required_columns` 之后携带 `row_groups_total`、
`row_groups_selected` 与 `pushed_condition`（归属该源的下推叶子按 SQL 出现顺序
组合的条件树，保留限定列名；无合格叶子时选中数等于总数且为 null），其计数与
查询实际读取的行组一致；v1 `Scan` 与含外连接链的 `Scan` 结构不变。

## 限制

- 单文件 SQL 查询：`SELECT`（可选 `DISTINCT` 行去重；星号/列名/单层聚合/带 `AS`
  别名的标量表达式，标量表达式含
  数值算术与 searched `CASE`）+
  固定表名 `input` + 可选 `WHERE`、`GROUP BY`、`HAVING`（仅限分组列/聚合/字面量条件）、
  `ORDER BY`、`LIMIT`；DISTINCT 不与 `GROUP BY`/`HAVING`/聚合投影组合，不支持
  simple CASE、
  裸列或聚合的别名、SELECT 别名用于 HAVING、聚合嵌套、WHERE 内聚合、HAVING 内标量
  表达式或未分组列、聚合查询 SELECT 中的标量表达式、连接等。多文件入口额外支持
  在 FROM 后确定性地连续连接零个或多个表，每步为
  `INNER JOIN` / `LEFT JOIN` / `RIGHT JOIN` / `FULL OUTER JOIN` 等值连接
  （每步仅引入一个未使用的表；无别名、无重复表、无复合或非等值 ON；
  `FULL` 必须写作 `FULL OUTER`，`LEFT`/`RIGHT` 不接 `OUTER`），连接查询中标量
  表达式的列引用同样必须限定表名。
- 压缩仅支持 `none` 与 `zlib`；字典编码仅可用于 utf8 列。
