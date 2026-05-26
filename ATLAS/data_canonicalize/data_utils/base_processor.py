#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sqlite3
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from data_utils.tools import find_parquet_file, pretty_print_sample, prompt_yes_no, sha1_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from utils import (
    TEST_MATH_SPLIT_DICT,
    TRAIN_MATH_SPLIT_DICT,
    TRAIN_CODE_SPLIT_DICT,
    TEST_CODE_SPLIT_DICT,
)

# Normalize split dicts: convert to {hf_id: [split1, split2, ...]} format
SPLIT_MAP: Dict[str, List[str]] = {}
for hf_id, value in {
    **TEST_MATH_SPLIT_DICT,
    **TRAIN_MATH_SPLIT_DICT,
    **TRAIN_CODE_SPLIT_DICT,
    **TEST_CODE_SPLIT_DICT,
}.items():
    if isinstance(value, dict):
        splits = value.get("splits", [])
        SPLIT_MAP[hf_id] = list(splits) if splits else []
    else:
        SPLIT_MAP[hf_id] = list(value) if value else []


class DatasetProcessor(ABC):
    """数据集处理基类，包含通用的数据处理逻辑"""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        """初始化处理器，如果未提供 args 则从命令行解析"""
        if args is None:
            args = self._parse_args()
        self.args = args

        if args.hf_id is None:
            args.hf_id = self.get_default_hf_id()

        if args.database_root is None:
            args.database_root = "$HOME/ATLAS/train/math"

        self.dataset_name = args.hf_id.split("/")[-1]
        self.dataset_dir = Path(args.database_root).resolve() / self.dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        split_names = SPLIT_MAP.get(args.hf_id, [])
        if not split_names:
            raise ValueError(f"Split not found for hf_id: {args.hf_id}")

        # If user specifies --split, only process that one
        if getattr(args, "split", None):
            split_names = [args.split]

        self.split_names: List[str] = split_names
        self.parquet_paths: List[Tuple[str, Path]] = []
        for s in self.split_names:
            try:
                p = find_parquet_file(self.dataset_dir, s)
            except Exception as e:
                print(f"[WARN] Cannot find parquet for split='{s}' under {self.dataset_dir}: {e}")
                continue
            self.parquet_paths.append((s, p))

        if not self.parquet_paths:
            raise FileNotFoundError(
                f"No parquet files found for hf_id={args.hf_id} "
                f"splits={self.split_names} under {self.dataset_dir}"
            )

        # Back-compat
        self.parquet_path = self.parquet_paths[0][1]

    def _parse_args(self) -> argparse.Namespace:
        """解析命令行参数，子类可以覆盖以设置默认值"""
        parser = argparse.ArgumentParser()

        parser.add_argument("--hf_id", type=str, default=None, help="HuggingFace dataset ID")
        parser.add_argument(
            "--split",
            type=str,
            default=None,
            help="Only process a specific split (split name or exact .parquet filename)",
        )

        defaults = self.__class__.get_default_args()
        parser.add_argument("--database_root", type=str, default=defaults.get("database_root"))
        parser.add_argument(
            "--prompt_field",
            type=str,
            default=defaults.get("prompt_field"),
            help="Field name for prompt input",
        )
        parser.add_argument(
            "--keep_org_prompt",
            action="store_true",
            default=defaults.get("keep_org_prompt", False),
            help="Keep original prompt field as org_prompt",
        )
        parser.add_argument(
            "--answer_key",
            type=str,
            default=defaults.get("answer_key"),
            help="Field name for answer input",
        )
        parser.add_argument(
            "--keep_org_answer",
            action="store_true",
            default=defaults.get("keep_org_answer", False),
            help="Keep original answer field as org_answer",
        )
        parser.add_argument(
            "--source_field",
            type=str,
            default=defaults.get("source_field"),
            help="Field name for source input",
        )
        parser.add_argument(
            "--keep_org_source",
            action="store_true",
            default=defaults.get("keep_org_source", False),
            help="Keep original source field as org_source",
        )
        parser.add_argument(
            "--sample_row",
            type=int,
            default=defaults.get("sample_row", 0),
            help="Row index for preview",
        )
        parser.add_argument(
            "--write_tmp",
            action="store_true",
            help="Keep promoted tmp parquet if user declines overwrite",
        )
        parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
        parser.add_argument(
            "--print_max_len",
            type=int,
            default=defaults.get("print_max_len", 100),
            help="Maximum length of sample to print",
        )
        parser.add_argument(
            "--filter_column",
            type=str,
            default=defaults.get("filter_column"),
            help="Column name to filter on",
        )
        parser.add_argument(
            "--filter_values",
            type=str,
            nargs="+",
            default=defaults.get("filter_values"),
            help="Rows with these values in filter_column will be discarded",
        )
        parser.add_argument(
            "--output_split",
            type=str,
            default=defaults.get("output_split"),
            help="Output split name used for saved parquet/sqlite source parquet",
        )
        parser.add_argument(
            "--skip_parquet",
            action="store_true",
            default=defaults.get("skip_parquet", True),
            help="Skip saving parquet file",
        )
        parser.add_argument(
            "--parquet_output_root",
            type=str,
            default=defaults.get("parquet_output_root"),
            help="Root for processed parquet (if set, do not write to database_root)",
        )
        parser.add_argument(
            "--sqlite_output_root",
            type=str,
            default=defaults.get("sqlite_output_root"),
            help="Root directory for SQLite output",
        )
        return parser.parse_args()

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        """获取默认参数，子类可以覆盖此方法设置默认值"""
        return {
            "prompt_field": None,
            "answer_key": None,
            "source_field": None,
            "keep_org_prompt": False,
            "keep_org_answer": False,
            "keep_org_source": False,
            "sample_row": 0,
            "print_max_len": 1000,
            "filter_column": None,
            "filter_values": None,
            "output_split": None,
            "skip_parquet": True,
            "parquet_output_root": None,
            "sqlite_output_root": None,
        }

    @abstractmethod
    def process_prompt(self, text: Any) -> Any:
        """处理 prompt 字段，子类必须实现"""
        pass

    @abstractmethod
    def process_answer(self, val: Any) -> Any:
        """处理 answer 字段，子类必须实现"""
        pass

    def process_solution(self, val: Any) -> Any:
        """处理 solution 字段，子类可选实现（默认返回原值）"""
        return val

    def process_source(self, val: Any) -> Any:
        """处理 source 字段，子类可选实现（默认返回原值）"""
        return val

    def process_column_value_mapping(self, table: pa.Table) -> pa.Table:
        """处理列值映射，子类可以覆盖此方法以应用值映射或过滤行"""
        return table

    def get_default_hf_id(self) -> str:
        """获取默认的 hf_id，子类可以覆盖"""
        raise ValueError("--hf_id is required or override get_default_hf_id() in subclass")

    def _normalize_prompt_for_hash(self, prompt_val: Any) -> Optional[str]:
        if prompt_val is None:
            return None
        s = str(prompt_val).strip()
        return s if s else None

    def _process_row_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """核心单行处理逻辑。process_row 和 process_table 共用，保证一致。"""
        after_row = dict(row)

        # Process prompt
        if self.args.prompt_field and self.args.prompt_field in row:
            raw_prompt = row[self.args.prompt_field]
            processed_prompt = self.process_prompt(raw_prompt)

            if self.args.keep_org_prompt:
                after_row["org_prompt"] = raw_prompt

            if self.args.prompt_field != "prompt":
                after_row.pop(self.args.prompt_field, None)

            after_row["prompt"] = processed_prompt

        # Process answer
        if self.args.answer_key and self.args.answer_key in row:
            raw_answer = row[self.args.answer_key]
            processed_answer = self.process_answer(raw_answer)

            if self.args.keep_org_answer:
                after_row["org_answer"] = raw_answer

            if self.args.answer_key != "answer":
                after_row.pop(self.args.answer_key, None)

            after_row["answer"] = processed_answer

        # Process source
        if self.args.source_field and self.args.source_field in row:
            raw_source = row[self.args.source_field]
            processed_source = self.process_source(raw_source)

            if self.args.keep_org_source:
                after_row["org_source"] = raw_source

            if self.args.source_field != "source":
                after_row.pop(self.args.source_field, None)

            after_row["source"] = processed_source

        # Compute prompt_sha1
        prompt_norm = self._normalize_prompt_for_hash(after_row.get("prompt"))
        after_row["prompt_sha1"] = sha1_text(prompt_norm) if prompt_norm else None

        return after_row

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """处理单行数据"""
        return self._process_row_dict(row)

    def process_table(self, table: pa.Table) -> pa.Table:
        """处理整个表。通过复用单行逻辑，保证与 process_row 完全一致。"""
        rows = table.to_pylist()
        processed_rows = [self._process_row_dict(r) for r in rows]

        # Filter rows based on filter_column and filter_values
        if self.args.filter_column and self.args.filter_values:
            filter_values_set = set(self.args.filter_values)
            before_n = len(processed_rows)
            processed_rows = [
                r for r in processed_rows
                if r.get(self.args.filter_column) not in filter_values_set
            ]
            removed = before_n - len(processed_rows)
            print(
                f"[FILTER] Filtered out {removed} rows where "
                f"{self.args.filter_column} in {self.args.filter_values}"
            )

        if not processed_rows:
            return pa.table({})

        return pa.Table.from_pylist(processed_rows)

    def normalize_sqlite_value(self, v):
        """Convert Parquet / Arrow values into SQLite-safe scalars."""
        if v is None:
            return None
        if isinstance(v, (int, float, str, bytes)):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    def parquet_to_sqlite(
        self,
        parquet_path: Path,
        sqlite_path: Path,
        table_name: str = "data",
        batch_size: int = 500000,
        overwrite: bool = False,
    ):
        """Convert parquet to SQLite"""
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        if sqlite_path.exists():
            if overwrite:
                sqlite_path.unlink()
            elif prompt_yes_no("Overwrite existing SQLite? [y/n]: "):
                sqlite_path.unlink()
            else:
                print(f"[SKIP] SQLite exists: {sqlite_path}")
                return

        print(f"[CONVERT] {parquet_path} -> {sqlite_path}")

        pf = pq.ParquetFile(parquet_path)
        schema = pf.schema_arrow
        selected_columns = schema.names

        if not selected_columns:
            print("  [SKIP] no columns")
            return

        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")

        col_defs = [f'"{c}" TEXT' for c in selected_columns]
        cur.execute(f'CREATE TABLE {table_name} ({", ".join(col_defs)});')
        conn.commit()

        placeholders = ",".join(["?"] * len(selected_columns))
        insert_sql = f'INSERT INTO {table_name} VALUES ({placeholders});'

        conn.execute("BEGIN")
        total_rows = 0

        for batch in pf.iter_batches(batch_size=batch_size, columns=selected_columns):
            col_lists = [
                [self.normalize_sqlite_value(v) for v in batch.column(c).to_pylist()]
                for c in selected_columns
            ]
            rows = list(zip(*col_lists))
            cur.executemany(insert_sql, rows)
            total_rows += len(rows)

        conn.execute("COMMIT")
        conn.close()

        print(f"  [OK] rows inserted: {total_rows}")

    def add_dl_row_idx_to_sqlite(
        self,
        sqlite_path: Path,
        table: str = "data",
        column: str = "DL_row_idx",
        batch_size: int = 16384,
        force: bool = False,
    ):
        """Add DL_row_idx column to SQLite table"""
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        )
        if cur.fetchone() is None:
            conn.close()
            raise RuntimeError(f"Missing table '{table}' in {sqlite_path}")

        cur.execute(f"SELECT COUNT(1) FROM {table}")
        total = int(cur.fetchone()[0])

        cur.execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in cur.fetchall()]
        column_exists = column in cols

        if not column_exists:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER")
            conn.commit()

        if column_exists:
            cur.execute(f"SELECT COUNT(1) FROM {table} WHERE {column} IS NOT NULL")
            filled = int(cur.fetchone()[0])
            if filled > 0 and not force:
                conn.close()
                return

        cur.execute(f"SELECT rowid AS _rid FROM {table}")
        rid_rows = cur.fetchall()

        updates: List[Tuple[int, int]] = []
        cur.execute("BEGIN;")
        try:
            for idx, r in enumerate(rid_rows):
                updates.append((idx, int(r["_rid"])))
                if len(updates) >= batch_size:
                    cur.executemany(
                        f"UPDATE {table} SET {column}=? WHERE rowid=?",
                        updates,
                    )
                    updates.clear()

            if updates:
                cur.executemany(
                    f"UPDATE {table} SET {column}=? WHERE rowid=?",
                    updates,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise

        conn.close()
        print(f"  [OK] Added {column} to SQLite: {total} rows")

    def _get_output_split_name(self) -> str:
        """输出文件名使用的 split 名。"""
        if self.args.output_split:
            return self.args.output_split
        if len(self.split_names) == 1:
            return self.split_names[0]
        return "all4DL"

    def _write_tmp_then_promote(
        self,
        table: pa.Table,
        final_path: Path,
        force: bool = False,
    ) -> Optional[Path]:
        """
        严格执行：
        1) 先写 tmp
        2) 如 final 已存在，先挪到 bak
        3) 再把 tmp promote 成 final

        返回：
        - 若最终成功保存，返回 final_path
        - 若用户拒绝覆盖，返回 tmp_path
        """
        final_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = final_path.parent / f"{final_path.stem}.tmp{final_path.suffix}"
        bak_path = final_path.parent / f"{final_path.stem}.bak{final_path.suffix}"

        pq.write_table(table, tmp_path)
        print(f"[INFO] Wrote temporary parquet to: {tmp_path}")

        if final_path.exists() and not force:
            if not prompt_yes_no(f"Overwrite {final_path.name}? [y/n]: "):
                print(f"[DONE] Not overwritten. Processed file kept at: {tmp_path}")
                return tmp_path

        if final_path.exists():
            if bak_path.exists():
                bak_path.unlink()
            final_path.replace(bak_path)
            print(f"[INFO] Backup saved at: {bak_path}")

        tmp_path.replace(final_path)
        print(f"[OK] Saved parquet: {final_path}")
        return final_path

    def run(self):
        """主执行流程"""
        processed_tables: List[pa.Table] = []

        # Step 1: Process each split separately
        for split_name, parquet_path in self.parquet_paths:
            print(f"\n[INFO] Processing split='{split_name}' parquet={parquet_path}")

            table = pq.read_table(parquet_path)
            if table.num_rows == 0:
                print("[WARN] Empty parquet table. Skipping.")
                continue

            if len(processed_tables) == 0:
                if 0 <= self.args.sample_row < table.num_rows:
                    for _ in range(5):
                        import random
                        row_idx = random.randint(0, table.num_rows - 1)
                        before_row = table.slice(row_idx, 1).to_pylist()[0]
                        after_row = self.process_row(before_row)
                        pretty_print_sample(before_row, after_row, max_len=self.args.print_max_len)
                        if not self.args.force:
                            input("Press Enter to continue...")

                    if not self.args.force and not prompt_yes_no("\nContinue? [y/n]: "):
                        print("[ABORT] User chose not to proceed.")
                        return

            # Step 2: Base processing
            processed_table = self.process_table(table)

            # Step 3: Subclass mapping function
            processed_table = self.process_column_value_mapping(processed_table)

            # Step 4: Add org_split column
            num_rows = processed_table.num_rows
            if num_rows > 0:
                cols_dict = {name: processed_table.column(name) for name in processed_table.schema.names}
                cols_dict["org_split"] = pa.array([split_name] * num_rows)
                processed_table = pa.table(cols_dict)

            processed_tables.append(processed_table)
            print(f"[INFO] Processed split '{split_name}': {num_rows} rows")

        if not processed_tables:
            print("[ERROR] No tables processed. Nothing to do.")
            return

        # Step 5: Merge all splits
        print(f"\n[INFO] Merging {len(processed_tables)} splits...")
        merged_table = pa.concat_tables(processed_tables, promote=True)
        print(
            f"[INFO] Merged table: {merged_table.num_rows} rows, "
            f"{len(merged_table.schema.names)} columns"
        )

        output_split_name = self._get_output_split_name()

        parquet_output_dir = (
            Path(self.args.parquet_output_root).resolve() / self.dataset_name
            if getattr(self.args, "parquet_output_root", None)
            else self.dataset_dir
        )
        final_parquet_path = parquet_output_dir / f"{output_split_name}.parquet"

        # Step 6: Save parquet
        saved_parquet_path: Optional[Path] = None

        if not self.args.skip_parquet:
            saved_parquet_path = self._write_tmp_then_promote(
                table=merged_table,
                final_path=final_parquet_path,
                force=self.args.force,
            )
        else:
            print("[INFO] Skipping parquet save (--skip_parquet)")

        # Step 7: Convert to SQLite
        if self.args.sqlite_output_root:
            sqlite_output_root = Path(self.args.sqlite_output_root).resolve()
            sqlite_path = sqlite_output_root / self.dataset_name / "all4DL.sqlite"

            # SQLite source parquet priority:
            # 1) final saved parquet if available
            # 2) if skip_parquet, create a dedicated tmp parquet for sqlite
            # 3) if user declined overwrite, saved_parquet_path may be tmp file -> still usable
            sqlite_source_parquet: Optional[Path] = None

            if saved_parquet_path is not None:
                sqlite_source_parquet = saved_parquet_path
            else:
                parquet_output_dir.mkdir(parents=True, exist_ok=True)
                sqlite_source_parquet = parquet_output_dir / f"{output_split_name}.tmp_for_sqlite.parquet"
                pq.write_table(merged_table, sqlite_source_parquet)
                print(f"[INFO] Wrote temporary parquet for SQLite conversion: {sqlite_source_parquet}")

            print(f"\n[INFO] Converting to SQLite: {sqlite_path}")
            self.parquet_to_sqlite(
                parquet_path=sqlite_source_parquet,
                sqlite_path=sqlite_path,
                table_name="data",
                batch_size=50000,
                overwrite=self.args.force,
            )

            # Print one random row from SQLite
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM data ORDER BY RANDOM() LIMIT 1")
            row = cur.fetchone()
            if row is not None:
                print("[SAMPLE] Random row from SQLite:")
                for k in row.keys():
                    v = row[k]
                    if isinstance(v, str) and len(v) > 200:
                        v = v[:200] + "..."
                    print(f"  {k}: {v}")
            conn.close()

            print("[INFO] Adding DL_row_idx to SQLite...")
            self.add_dl_row_idx_to_sqlite(
                sqlite_path=sqlite_path,
                table="data",
                column="DL_row_idx",
                batch_size=16384,
                force=self.args.force,
            )

            # Clean up dedicated sqlite tmp parquet only when skip_parquet=True
            if self.args.skip_parquet and sqlite_source_parquet.name.endswith(".tmp_for_sqlite.parquet"):
                sqlite_source_parquet.unlink(missing_ok=True)
                print("[INFO] Cleaned up temporary parquet file")
        else:
            print("[INFO] Skipping SQLite conversion (--sqlite_output_root not specified)")

    def main(self):
        """命令行入口"""
        self.run()