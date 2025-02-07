import csv

input_filename = 'dataset/validation.csv'
output_filename = 'dataset/validation_small.csv'

# Try different encodings until you find the right one
with open(input_filename, 'r', newline='', encoding='utf-8') as input_file:  # Most common encoding
    reader = csv.reader(input_file)
    rows = list(reader)
    total_rows = len(rows)
    
    # Calculate rows to keep (10% of total)
    rows_to_keep = int(total_rows * 0.3)
    new_rows = rows[:rows_to_keep]

with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(new_rows)

print(f"Kept first {len(new_rows)} rows (10%) and deleted the last {total_rows - len(new_rows)} rows.")