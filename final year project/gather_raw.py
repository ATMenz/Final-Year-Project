import zstandard
import os
import json
import csv
from datetime import datetime
import logging.handlers

# Configuration
input_file = r"zst_input\worldnews_comments.zst"
base_output_path = r"worldnews\worldnews_raw_economics1"  
start_year = "2018"  
end_year = "2022" 
keywords = ['economy', 'inflation', 'recession', 'GDP', 'unemployment', 'markets', 'stocks', 'bonds', 'interest rates',
             'exchange rate', 'trade', 'investment', 'savings', 'debt', 'deficit', 'taxation', 'budget', 'financial market', 'real estate', 
             'commodities', 'agriculture', 'manufacturing', 'services sector', 'tech sector', 'energy market'
             ]

years_to_process = list(range(int(start_year), int(end_year) + 1))  

log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024*1024*16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip()
            buffer = lines[-1]
        reader.close()

def process_file(input_file, output_base_path, year, field, values, exact_match):
    log.info(f"Starting processing for year {year}.")
    line_count = 0
    matched_lines = 0
    output_file_name = os.path.join(output_base_path, f"{year}.csv")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    handle = open(output_file_name, 'a', encoding='UTF-8', newline='')
    writer = csv.writer(handle)
    writer.writerow(["score", "created_utc", "author", "body"]) 
    for line in read_lines_zst(input_file):
        line_count += 1
        if line_count % 10000 == 0:
            log.info(f"Processed {line_count} lines so far...")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))
            if created.year != year:
                continue
            if not values or (field in obj and any(val.lower() in obj[field].lower() for val in values)):
                writer.writerow([obj.get("score"), created.strftime("%Y-%m-%d"), obj.get("author"), obj.get("body")])
                matched_lines += 1
        except Exception as e:
            log.error(f"Failed to process line: {e}")
    handle.close()
    log.info(f"Completed processing for year {year}. Total lines processed: {line_count}. Total matched lines: {matched_lines}.")

if __name__ == "__main__":
    field = "body"
    
    exact_match = False

    for year in years_to_process:
        year_output_path = os.path.join(base_output_path, str(year))
        os.makedirs(year_output_path, exist_ok=True) 
        process_file(input_file, year_output_path, year, field, keywords, exact_match)