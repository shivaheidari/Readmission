import csv


input_file = "../Data/NOTEEVENTS.csv"
output_file = "../Data/NOTEEVENTS_CLEAN.csv"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    header = [col if col != 'STORETIME' else 'CHARTTIME' for col in header]
    writer.writerow(header)

    for row in reader:
        writer.writerow(row)