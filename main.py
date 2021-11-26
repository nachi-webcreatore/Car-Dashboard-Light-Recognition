# Python program to read an excel file
# import openpyxl module 
import openpyxl
import numpy as np

# Give the location of the file 
path = "1637216720-area.xlsx"

# To open the workbook 
# workbook object is created 
wb_obj = openpyxl.load_workbook(path)

# Get workbook active sheet object 
# from the active attribute 
sheet_obj = wb_obj.active

# Cell objects also have a row, column, 
# and coordinate attributes that provide 
# location information for the cell. 

# Note: The first row or 
# column integer is 1, not 0. 
x = np.ndarray
# Cell object is created by using 
# sheet object's cell() method.
for n in range(100):
    cell_obj = sheet_obj.cell(row=2 + n, column=2)
    print(cell_obj.value)
    np.insert(cell_obj.value.split(","), n, x)
#print(x)
