import pandas as pd
import csv

#should be training data, script will ensure all columns in this file are present in the output
ref_filename = "data/ncats_mixture_simplex_fixed.txt"
ref_db = pd.read_csv(ref_filename, sep = '\t')

#all test descriptor files (structures were batched make sirms generation faster on cleber's end)
#filenames = ["other_lab_predictions/top30s_concatenated_simplex.txt"]
filenames = ["backup_two_combos/backup_two_combos_simplex.txt"]

#I still don't the right way to grow a dataframe from nothing
full_df = 0

for adj_filename in filenames:
    print(adj_filename)

    adj_db = pd.read_csv(adj_filename, sep = '\t')
    adj_num_obs = adj_db.shape[0]
    adj_column_set = set(adj_db.columns)

    #initialize output dataframe with all zeros and all columns from reference file
    output_df = pd.DataFrame(0, index = adj_db.index, columns = ref_db.columns)

    #check each test file column and add it if it's in the reference set
    #if it's not, it will already be all zeros from initialization
    for i in range(len(ref_db.columns)):
        column_name = ref_db.columns[i]
        if column_name in adj_column_set:
            output_df[column_name] = adj_db[column_name]

    #just awkward way of growing the dataframe with every iteration
    if type(full_df) == int:
        full_df = output_df
    else:
        full_df = pd.concat((full_df, output_df))

full_df.to_csv("backup_two_combos/adjusted_backup_two_combos_simplex.txt", sep = '\t', quoting=csv.QUOTE_NONNUMERIC, index = False)

