import csv

#Small script to cut down the super large CSV files. Just cuts down to the first 2000 rows.
#Make sure the folder you are transferring to is empty
def CutDown(oldFilepath, newFilepath):
    with open('Validation\\EHAM_LIMC.csv', newline='') as f:
        reader = csv.reader(f)
        #Change the 2000 to change the number of rows that the final version gets
        for row in range(0,2000):
            with open('Validation\\Testing_Data.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(reader)
