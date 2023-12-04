import os

import openpyxl
import pandas as pd

print(os.getcwd())

excel_file_path = 'FalseAug_Cheb_Cornell_output.xlsx'

workbook = openpyxl.load_workbook(excel_file_path)

# Access the desired sheet
sheet1 = workbook['Epoch']
sheet2 = workbook['Split']

# Adjust the column width for specific columns
sheet1.column_dimensions['Epoch_Output'].width = 100
sheet2.column_dimensions['Split_Output'].width = 100

# Save the changes
workbook.save(excel_file_path)

# Close the workbook
workbook.close()


# writer = pd.ExcelWriter(excel_file_path, mode="a", engine="openpyxl")
#
# # with pd.ExcelWriter(excel_file_path, mode="a", engine="openpyxl") as writer:
# worksheet1 = writer.sheets['Epoch']
#
# # Adjust the column width for Column1 in Sheet1
# worksheet1.column_dimensions['Epoch_Output'].width = 100
#
# # Access the worksheet for Sheet2
# worksheet2 = writer.sheets['Split']
#
# # Adjust the column width for ColumnA in Sheet2
# worksheet2.column_dimensions['Split_Output'].width = 100