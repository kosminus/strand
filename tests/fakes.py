"""A stand-in Spark session that records what generated code does to it.

There is no JVM on the build machine, so these tests check the plan, the
generated code paths and the LangGraph control flow - not Spark semantics.
"""
from __future__ import annotations


class _Type:
    def simpleString(self):
        return "string"


class _Field:
    def __init__(self, name):
        self.name, self.dataType = name, _Type()


class _Schema:
    fields = [_Field("id")]


class FakeDF:
    schema = _Schema()

    def __init__(self, spark, lineage):
        self.spark, self.lineage = spark, lineage

    def _op(self, name, *args):
        return FakeDF(self.spark, f"{name}({self.lineage}{''.join(', ' + str(a) for a in args)})")

    def filter(self, cond):
        return self._op("filter", cond)

    def selectExpr(self, *exprs):
        return self._op("selectExpr", *exprs)

    def join(self, other, on=None, how="inner"):
        return self._op("join", other.lineage, on, how)

    def limit(self, n):
        return self._op("limit", n)

    def alias(self, a):
        return self

    def count(self):
        return 3

    def collect(self):
        if self.lineage.startswith("sql(DESCRIBE"):
            raise RuntimeError("no history")
        return []

    @property
    def write(self):
        return FakeWriter(self)


class FakeWriter:
    def __init__(self, df):
        self.df, self._mode, self.options = df, "error", {}

    def format(self, f):
        return self

    def mode(self, m):
        self._mode = m
        return self

    def option(self, k, v):
        self.options[k] = v
        return self

    def saveAsTable(self, name):
        spark = self.df.spark
        key = (self.options.get("txnAppId"), self.options.get("txnVersion"))
        if key[0] and key in spark.txns:  # Delta's idempotent-write behaviour
            return
        spark.txns.add(key)
        spark.tables[name] = self.df
        spark.writes.append((name, self._mode, dict(self.options), self.df.lineage))


class _Reader:
    def __init__(self, spark):
        self.spark = spark

    def table(self, name):
        self.spark.reads.append(name)
        return FakeDF(self.spark, f"table({name})")

    def format(self, f):
        return self

    def load(self, path):
        return FakeDF(self.spark, f"load({path})")


class _Catalog:
    def __init__(self, spark):
        self.spark = spark

    def tableExists(self, name):
        return name in self.spark.tables


class FakeSpark:
    def __init__(self):
        self.tables, self.writes, self.reads, self.txns = {}, [], [], set()
        self.read, self.catalog = _Reader(self), _Catalog(self)

    def sql(self, query, args=None, **kwargs):
        ins = ", ".join(f"{k}={v.lineage}" for k, v in kwargs.items())
        if args:
            ins = f"args[{', '.join(args)}] " + ins
        return FakeDF(self, f"sql({query}{' | ' + ins if ins else ''})")

    def createDataFrame(self, rows, schema=None):
        return FakeDF(self, f"rows({len(rows)})")

    def user_writes(self):
        return [w for w in self.writes if "strand_" not in w[0].split(".")[-1] and "__" not in w[0]]
