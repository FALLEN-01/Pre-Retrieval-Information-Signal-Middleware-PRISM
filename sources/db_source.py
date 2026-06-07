from typing import Optional
from sqlalchemy import create_engine, text


class DBSource:
    """Queries SQL databases and returns results as readable text.
    """

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def query(self, sql: str) -> Optional[str]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                keys = list(result.keys())
                rows = result.fetchall()

            lines = [
                " | ".join(f"{k}: {v}" for k, v in zip(keys, row))
                for row in rows
            ]
            return "\n".join(lines) if lines else None
        except Exception as e:
            print(f"[DBSource] Query failed: {e}")
            return None

    def query_to_doc(self, sql: str, label: str = "database") -> Optional[dict]:
        content = self.query(sql)
        if content:
            return {"source": label, "source_type": "db", "content": content}
        return None
