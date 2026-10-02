## 用途

本项目是「列式分析型数据库引擎」的代码仓库，用于逐步实现该方向的列式存储、查询执行与结果对账能力。

当前已实现可独立读写的**列式文件层**，以及面向单个文件的 SQL 查询入口
（`SELECT` 投影、`WHERE` 过滤、`GROUP BY` 分组与 `COUNT`/`SUM`/`AVG`/`MIN`/`MAX`
聚合、`ORDER BY` 稳定排序与 `LIMIT` Top-N）；查询只读取文件、不修改文件。
另提供两文件查询入口，支持一次 `INNER JOIN` / `LEFT JOIN` 等值连接。
对应的 `explain_file` / `explain_files`（命令行 `explain` / `explain-files`）
只解析、绑定并生成逻辑计划，仅读文件元数据、不执行查询。
`export_query_file` / `export_query_files`（命令行 `export` / `export-files`）
复用同一 SQL 子集与结果语义，把查询结果确定性、原子地导出为 CSV 或 JSONL 文件。

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
columnar-analytics-engine query-files <sources-json> "<sql>"   # 对映射的多表执行 SQL（可含一次连接）
columnar-analytics-engine explain <path> "<sql>"        # 只解析/绑定并输出单文件语句的计划 JSON
columnar-analytics-engine explain-files <sources-json> "<sql>"  # 只解析/绑定并输出多表语句的计划 JSON
columnar-analytics-engine export <path> "<sql>" <target> [--format csv|jsonl]   # 导出单文件查询结果
columnar-analytics-engine export-files <sources-json> "<sql>" <target> [--format csv|jsonl]  # 导出多表查询结果
columnar-analytics-engine --help                        # 打印用法
```

`inspect` 只读取文件元数据（不读取、不解码列数据），以 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `format_version`、`row_count`、`columns`，columns 保持 schema 顺序。
格式错误时向标准错误输出消息并以码 2 退出；路径等系统错误以码 1 退出。

`query` 对单个文件执行一条
`SELECT ... FROM input [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT n]` 语句，以单行
UTF-8 JSON 输出到标准输出，顶层键依次为 `columns`、`rows`；`columns` 按结果顺序
列出每列的 `name`、`type`、`nullable`，`rows` 是同序值数组的数组。空结果保留列
描述且 `rows` 为空（无 GROUP BY 的聚合查询在零入选行时仍返回一行，COUNT 为 0、
其余聚合为 null；有 GROUP BY 时返回零行）；同一文件与 SQL 重复执行输出字节一致。语法错误
（`QuerySyntaxError`）、校验错误（`QueryValidationError`）与文件格式错误
（`ColumnarFormatError`）向标准错误输出消息并以码 2 退出；系统错误以码 1 退出。

`query-files` 的 `<sources-json>` 是一个 JSON 对象，把表名映射到列式文件路径，例如
`'{"l": "left.caef", "r": "right.caef"}'`；SQL 中的 `FROM`/`JOIN` 表名即取这些键。
输出格式与退出码约定同 `query`；sources-json 本身非法（不是 JSON 对象、键不是非空
字符串、值不是路径）也以码 2 退出。

`explain` / `explain-files` 与对应的查询入口接受完全相同的输入与 SQL 子集，但只做
解析、绑定与计划生成：不执行查询，只读取文件元数据头（不读取、不解压、不解码数据
段），也不会打开未被语句引用的 sources。成功时以单行紧凑 UTF-8 JSON 输出到标准输出，
顶层键顺序固定为 `sources`、`operators`、`output`：

- `sources` 按 `FROM`、`JOIN` 顺序列出，每项含 `name`、`row_count` 与按源 schema
  顺序的 `columns`（每列 `name`、`type`、`nullable`）。
- `operators` 依次包含实际存在的阶段，顺序为 `Scan`、`Join`、`Filter`、`Aggregate`、
  `Sort`、`Limit`、`Project`；缺少的阶段省略：
  - 每个被引用源一个 `Scan`，`required_columns` 按源 schema 顺序给出语句引用的列
    （投影、WHERE、GROUP BY、ORDER BY 及连接键；仅 `COUNT(*)` 时为空）。
  - `Join`（仅连接查询）给出 `type`（`INNER`/`LEFT`）与 `left`、`right` 两侧限定键
    （`table`、`column`）。
  - `Filter` 的 `condition` 是递归表达式树：内部节点含 `kind`、`operator`、`operands`，
    叶子是带 `type` 的 `literal`（`value` 为类型化字面量）或带 `name` 的绑定 `column`。
  - `Aggregate` 给出 `group_keys`（绑定列名列表，无 GROUP BY 时为空）与 `aggregates`
    （每项 `function`、`argument`（`COUNT(*)` 为 null）、`output`）。
  - `Sort` 的 `keys` 每项给出 `column`、`direction`（`ASC`/`DESC`）与
    `nulls`（`FIRST`/`LAST`）；`Limit` 给出 `count`；`Project` 的 `expressions`
    每项给出绑定表达式（列、聚合或递归标量表达式）与 `output` 输出名。
- `output` 按结果顺序给出 `name`、`type`、`nullable`，与对应查询结果的 schema 一致。

相同元数据与 SQL 重复解释输出字节一致；关键字大小写与多余空白不改变计划内容。
语法错误在访问任何文件之前抛 `QuerySyntaxError`；非法 sources 抛 `ValueError`；
未知表/列、类型不兼容及其他绑定错误抛 `QueryValidationError`；非法元数据或声明尺寸
不符抛 `ColumnarFormatError`；系统访问失败保留 `OSError`（数据段 CRC 与值级统计因
未读取而不校验）。命令行对 `ValueError`、查询错误与格式错误以码 2 退出，对
`OSError` 以码 1 退出，失败时标准输出为空。

## 查询结果导出

`export_query_file(path, sql, target, format="csv")` 与
`export_query_files(sources, sql, target, format="csv")`（命令行 `export` /
`export-files`，对应 `--format` 选项）把查询结果写入目标文件；它们复用现有 SQL
子集、绑定规则、连接顺序、NULL 与类型语义及稳定排序，不增加查询语法。`format`
只接受 `csv`（默认）或 `jsonl`；成功时 Python 入口返回写出的结果行数，命令行以
状态码 0 结束且标准输出、标准错误均为空。

- **CSV**：UTF-8 无 BOM、LF 换行；第一行始终按结果 schema 顺序写列名。含逗号、
  双引号、CR 或 LF 的字段整体加双引号，内部双引号重复；NULL 写为空的未加引号
  字段，空字符串固定写成 `""`，bool 写为 `true`/`false`，数值采用与紧凑 JSON
  结果相同的文本形式。零行结果只含表头行。
- **JSONL**：每个结果行写成一个紧凑 JSON 对象，键按结果列顺序排列，值保持现有
  JSON 类型和 null，非 ASCII 字符不转义，每行以一个 LF 结束；零行结果为零字节文件。

相同输入、SQL、格式与版本重复导出字节一致；未写 `ORDER BY` 时沿用现有查询的确定
顺序。导出只在查询完整成功且全部内容可编码后原子替换目标文件，任何失败都不会新建
目标或改变已有目标。目标与任一实际引用的源文件解析为同一路径时抛 `ValueError`，
未知格式也抛 `ValueError`；SQL、绑定与列式格式问题继续分别抛 `QuerySyntaxError`、
`QueryValidationError`、`ColumnarFormatError`，sources 校验仍抛 `ValueError`，目录、
权限及其他文件系统问题保留 `OSError`。命令行把 `ValueError` 与三类查询错误映射为
状态码 2，把 `OSError` 映射为状态码 1，失败时标准输出为空、标准错误只写异常消息。

## Python 公开接口

包 `columnar_analytics` 导出：

- `Schema` / `ColumnSchema` / `Table`：有序 schema 与按列数据表
- `write_file(path, table, *, compression="none", dictionary_encoding=())`：确定性、原子写出
- `read_file(path, *, columns=None)`：读回表；`columns` 按调用方顺序投影部分列
- `inspect_file(path)`：只读元数据（行数、每列 NULL 数、min/max）
- `query_file(path, sql)`：对单个文件执行 SQL，成功返回 `Table`
- `query_files(sources, sql)`：对表名→路径映射执行 SQL（可含一次两表连接），成功返回 `Table`
- `explain_file(path, sql)`：只读元数据，返回单文件语句的有序计划字典（键为
  `sources`、`operators`、`output`），不执行查询、不读数据段
- `explain_files(sources, sql)`：同上，面向多表语句；未被引用的 sources 不会被打开
- `export_query_file(path, sql, target, format="csv")`：执行单文件查询并把结果导出为
  CSV/JSONL 文件，返回写出行数
- `export_query_files(sources, sql, target, format="csv")`：同上，面向多表语句
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
SELECT * | 投影项 [, 投影项 ...]
FROM input
[WHERE 表达式]
[GROUP BY 列名 [, 列名 ...]]
[ORDER BY 排序项 [, ...]]
[LIMIT 无符号整数]

投影项 := 列名
        | COUNT (*)
        | COUNT | SUM | AVG | MIN | MAX (列名)
        | 标量表达式 AS 别名
排序项 := (列名 | 聚合调用 | SELECT 别名) [ASC | DESC] [NULLS FIRST | NULLS LAST]
标量表达式 := 数值列 | 数值字面量 | (标量表达式)
           | +标量表达式 | -标量表达式
           | 标量表达式 + - * / 标量表达式   （优先级：括号、一元、乘除、加减）
```

- 投影允许星号、逗号分隔的列名、聚合调用或带 `AS` 别名的数值标量表达式；别名沿用
  标识符规则（可用双引号包裹），且不得与其他输出名重复。裸列与聚合调用不支持别名、
  名称保持不变；重复列、重复别名、重复输出名、未知列抛 `QueryValidationError`。
  `FROM` 只接受固定表名 `input`（裸写大小写不敏感）。
  列名与 schema 中的 Unicode 字符精确匹配；需要时可用双引号包裹标识符
  （内部用 `""` 转义一个双引号）。
- 标量表达式支持括号、一元 `+`/`-` 与二元 `+`、`-`、`*`、`/`，操作数为 int64/float64
  列与数值字面量。两个 int64 相加/减/乘结果为 int64；任一操作数为 float64 或执行除法
  时结果为 float64；一元运算保留类型。任一操作数为 NULL 时结果为 NULL；输出列的
  nullable 由参与列推导，纯常量表达式非空。int64 运算越界、除零、float64 结果非有限
  时抛 `QueryValidationError`。非聚合查询的执行顺序为 `WHERE` → 计算排序键 → 稳定排序
  → `LIMIT` → 结果表达式，因此被过滤或截掉的行不会触发 SELECT 中的除零或溢出。
  `ORDER BY` 可引用 SELECT 的显式别名（别名与输入列同名时优先解析别名），并按表达式
  的结果类型、NULL 位置与稳定排序规则排序。聚合查询不扩展：GROUP BY 键与聚合参数仍只
  接受列引用或 `COUNT(*)`，聚合查询混入标量表达式时抛 `QueryValidationError`。
- 聚合函数为 `COUNT(*)`、`COUNT(列)`、`SUM(列)`、`AVG(列)`、`MIN(列)`、`MAX(列)`，
  函数名大小写不敏感，结果列名为大写函数名加括号（列引用使用 schema 真实名）。
  `COUNT(*)` 计入选行，`COUNT(列)` 忽略 NULL，二者均为非空 int64；`SUM`/`AVG`
  只接受 int64/float64 且忽略 NULL，`SUM` 保持输入类型、`AVG` 输出 float64；
  `MIN`/`MAX` 接受现有四类并沿用既有比较规则。除 COUNT 外的聚合在没有非 NULL
  输入时返回 NULL，结果列 nullable。int64 求和越界或浮点聚合得到非有限值抛
  `QueryValidationError`。
- 无 `GROUP BY` 时投影只能包含聚合表达式（筛选为空也返回一行：COUNT 为 0、其余为
  NULL）；有 `GROUP BY` 时可投影分组列与聚合，普通列必须已分组，星号不得与聚合或
  分组混用（抛 `QueryValidationError`）。分组按 GROUP BY 列顺序成键，NULL 键归为
  一组；没有 ORDER BY 时各分组按首条入选行顺序输出，筛选为空时返回零行。
- `ORDER BY` 在非聚合查询中可引用任意 schema 列（无需出现在投影中）；在聚合查询中
  只能引用已选分组列或已选聚合表达式（按 schema 真实名匹配）。未知列、同一
  `ORDER BY` 中的重复项或引用未选结果抛 `QueryValidationError`。多列按书写顺序比较：
  int64/float64 按数值、utf8 按 Unicode 码点、bool 按 `FALSE < TRUE`；全部排序键
  相等的行（或分组）保持其原始（首条入选行）相对顺序。省略方向为 `ASC`；省略
  `NULLS` 时不论方向 NULL 都排在末尾，显式 `NULLS FIRST`/`NULLS LAST` 覆盖默认值。
- `LIMIT` 只接受 `0` 至 `9223372036854775807` 的无符号十进制整数，可独立出现；
  `0` 返回保留列描述的空表，大于入选行数（或分组数）时返回全部。非聚合查询的执行
  顺序固定为 `WHERE` → 排序 → `LIMIT` → 投影；聚合查询为
  `WHERE` → 分组/聚合 → 排序 → `LIMIT`。
- `WHERE` 支持括号、`NOT`、`AND`、`OR`、`=`、`!=`、`<`、`<=`、`>`、`>=`、
  `IS NULL`、`IS NOT NULL`；优先级从高到低为 `NOT`、比较、`AND`、`OR`。
  比较两侧可使用列引用、字面量或数值标量表达式：字面量为 `TRUE`/`FALSE`、int64 整数、
  有限 float64 数（均支持前导 `+`/`-`）、单引号 utf8 字符串（内部用 `''` 转义单引号）；
  数值表达式直接充当布尔条件抛 `QueryValidationError`；
  WHERE 内不允许聚合（抛 `QueryValidationError`）。
- int64 与 float64 可互比，utf8 只与 utf8 比较，bool 只支持 `=`/`!=`；
  不兼容组合抛 `QueryValidationError`。
- 遵循 SQL 三值逻辑：普通比较遇到 NULL 得 UNKNOWN，逻辑运算继续传播 UNKNOWN，
  只有 TRUE 的行进入分组与结果；省略 `WHERE` 保留全部行。
- 语法不完整、聚合括号或参数个数错误、`GROUP BY` 空列表、修饰词错位或重复、
  子句乱序或重复（GROUP BY 须位于 WHERE 之后、ORDER BY 之前）、`LIMIT`
  缺值/负数/小数/越界等统一抛 `QuerySyntaxError`，且在访问文件之前识别；
  文件损坏仍抛 `ColumnarFormatError`，系统错误保留 `OSError`。

## 两文件连接查询

`query_files(sources, sql)` 与 `columnar-analytics-engine query-files <sources-json> <sql>`
在单文件语法基础上支持一次等值连接；`sources` 是表名到文件路径的非空映射
（命令行用同一结构的 JSON 对象），只有被语句引用的表对应的文件会被读取：

```sql
SELECT ... FROM 左表 [INNER JOIN | LEFT JOIN] 右表 ON 左表.列 = 右表.列
[WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT n]
```

- 不支持别名、复合 `ON`、其他连接类型或第二次连接；词法错误、连接关键字缺失、
  非法 `ON`、超过一次连接抛 `QuerySyntaxError`，且在读取任何源文件之前判定。
- 连接查询中除 `COUNT(*)` 外的列引用都必须写成 `表名.列名`（两部分都可用双引号
  标识符，延续精确匹配与 `""` 转义语义）；未限定列、未知表或列、重复表、连接键
  来源错误（左右颠倒或来自同一表）、键类型不兼容、重复结果列抛
  `QueryValidationError`。非连接查询沿用单表语义，列可限定也可不限定。
- `SELECT *` 按左、右 schema 顺序输出，列名为 `表名.列名`；显式投影保留限定名，
  聚合结果列名沿用大写函数格式并含限定参数（如 `SUM(r.x)`）。
- 连接键类型须相同，或一为 int64 一为 float64；NULL 键永不匹配。`INNER JOIN`
  输出全部匹配组合；`LEFT JOIN` 还输出未匹配左行并把右侧值置为 NULL，
  右侧结果列 nullable 为 true。结果按左文件原始行序、同一左行内按右文件原始行序
  展开，之后 `WHERE`、`GROUP BY`、聚合、`ORDER BY`、`LIMIT` 按单文件语义处理；
  没有显式排序时相同输入与 SQL 重复执行输出字节一致。
- `sources` 非映射或为空、键不是非空字符串、值不是路径对象时抛 `ValueError`，
  且不会访问任何文件；已引用文件损坏仍抛 `ColumnarFormatError`，系统错误保留
  `OSError`。

## 文件格式概览

小端字节序：魔数 `CAEF` + 单字节格式版本 + uint32 头长度 + UTF-8 JSON 元数据头
（schema、每列偏移/校验/统计）+ 数据段（原始拼接或 zlib 压缩）+ uint32 CRC-32
（覆盖此前全部字节）+ 结束标记 `END1`。相同输入与选项重复写出的字节完全一致。
读取时拒绝错误魔数、未知版本、截断、校验不一致与非法元数据，统一抛
`ColumnarFormatError`。

## 限制

- 单文件 SQL 查询：`SELECT`（星号/列名/单层聚合/带 `AS` 别名的数值标量表达式）+
  固定表名 `input` + 可选 `WHERE`、`GROUP BY`、`ORDER BY`、`LIMIT`；不支持裸列或聚合的
  别名、聚合嵌套、WHERE 内聚合、聚合查询中的标量表达式、连接等。两文件入口额外支持一次
  `INNER JOIN` / `LEFT JOIN` 等值连接（无别名、无复合 ON、无第二次连接），连接查询中
  标量表达式的列引用同样必须限定表名。
- 压缩仅支持 `none` 与 `zlib`；字典编码仅可用于 utf8 列。
