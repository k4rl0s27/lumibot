import sqlite3, os

db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'memory', 'AITradingTeamOptionsDebateStrategy', 'memory.sqlite')
print(f"DB path: {db_path}")
print(f"DB exists: {os.path.exists(db_path)}")
if not os.path.exists(db_path):
    print("No memory DB found.")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM [{t[0]}]")
    count = cursor.fetchone()[0]
    print(f"\n{t[0]}: {count} rows")
    if count > 0:
        cursor.execute(f"SELECT * FROM [{t[0]}] LIMIT 5")
        cols = [d[0] for d in cursor.description]
        print(f"  Columns: {cols}")
        for row in cursor.fetchall():
            for c, v in zip(cols, row):
                if v is not None:
                    print(f"    {c}: {str(v)[:300]}")
            print("  ---")

conn.close()
