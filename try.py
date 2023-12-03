# import pandas as pd
#
# def print_and_save_output(split, test_accSHA, test_bacc, test_f1, device):
#     out_str = 'split: {}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(split, test_accSHA * 100, test_bacc * 100, test_f1 * 100)
#     Eout_str = 'split: {}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(split, test_accSHA * 666666100, test_bacc * 100, test_f1 * 100)
#
#     # Print to console
#     print(out_str)
#     print('Device:', device)
#
#     # Create a DataFrame
#     df = pd.DataFrame({'Epoch_Output': [Eout_str],'Output': [out_str], 'Device': [device]})
#
#     # Write the DataFrame to an Excel file
#     df.to_excel('output.xlsx', index=False)
#
# # Example usage
# split = 1
# test_accSHA = 0.85
# test_bacc = 0.90
# test_f1 = 0.87
# device = 'cuda:0'
#
# print_and_save_output(split, test_accSHA, test_bacc, test_f1, device)
import pandas as pd

# Assuming excel_file_path is defined elsewhere
excel_file_path = 'your_excel_file.xlsx'

# Dummy values for demonstration
epoch = 1
test_accSHA = 0.85
test_bacc = 0.75
test_f1 = 0.80
epoch_time = 10.5
split = 1

# Construct the first DataFrame for the Epoch output
epoch_output_str = 'Epoch:{}, time:{:2f}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(epoch, epoch_time, test_accSHA * 100, test_bacc * 100, test_f1 * 100)
epoch_df = pd.DataFrame({'Epoch_Output': [epoch_output_str]})

# Try to read the existing Excel file
try:
    existing_data = pd.read_excel(excel_file_path, sheet_name='Sheet1', engine='openpyxl')
    combined_data = pd.concat([epoch_df, existing_data], ignore_index=True)
except FileNotFoundError:
    combined_data = epoch_df

# Write the combined data to the Excel file with the first sheet
combined_data.to_excel(excel_file_path, sheet_name='Sheet1', index=False, engine='openpyxl')

# Construct the second DataFrame for the Split output
split_output_str = 'split: {}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(split, test_accSHA * 100, test_bacc * 100, test_f1 * 100)
split_df = pd.DataFrame({'Split_Output': [split_output_str]})

# Try to read the existing Excel file again
try:
    existing_data = pd.read_excel(excel_file_path, sheet_name='Sheet2', engine='openpyxl')
    combined_data = pd.concat([split_df, existing_data], ignore_index=True)
except FileNotFoundError:
    combined_data = split_df

# Write the combined data to the Excel file with the second sheet
combined_data.to_excel(excel_file_path, sheet_name='Sheet2', index=False, engine='openpyxl')
