import sqlite3

from datetime import datetime
import pandas as pd

# Connect to the database
conn = sqlite3.connect('messages.db')

df = pd.read_sql_query("SELECT content, author, timestamp FROM messages ORDER BY timestamp DESC", conn)


print("Dumping all messages from the database:")
# Print each row
for index, row in df.iterrows():
    dt_object = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S.%f%z")
    date_string = dt_object.strftime("%Y-%m-%d")
    
    print(date_string)
    print(row['author'])
    print(row['content'])
    # print(f"Content: {row[0]}\nAuthor: {row[1]}\nTimestamp: {row[2]}\n---")

# Close the connection
conn.close()
