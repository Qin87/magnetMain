import pandas as pd

def print_and_save_output(split, test_accSHA, test_bacc, test_f1, device):
    out_str = 'split: {}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(split, test_accSHA * 100, test_bacc * 100, test_f1 * 100)
    Eout_str = 'split: {}, test_Acc: {:.2f}, test_bacc: {:.2f}, test_f1: {:.2f}'.format(split, test_accSHA * 666666100, test_bacc * 100, test_f1 * 100)

    # Print to console
    print(out_str)
    print('Device:', device)

    # Create a DataFrame
    df = pd.DataFrame({'Epoch_Output': [Eout_str],'Output': [out_str], 'Device': [device]})

    # Write the DataFrame to an Excel file
    df.to_excel('output.xlsx', index=False)

# Example usage
split = 1
test_accSHA = 0.85
test_bacc = 0.90
test_f1 = 0.87
device = 'cuda:0'

print_and_save_output(split, test_accSHA, test_bacc, test_f1, device)
