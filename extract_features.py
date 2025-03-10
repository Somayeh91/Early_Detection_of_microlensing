import multiprocessing
import time
import warnings
import sys
import pandas as pd
from tqdm import tqdm
from functions_for_extracting_features import *

def process(index, df):
    try:
        # Extract time and magnification
        time, magnification = get_time_magnification(df, index)
        # plot_magnification(time, magnification, df, index)

        # Filter the data
        result = filter_days(time, magnification, index, df, days=5, thresh_mag=1.01)
        if result is None:
            return index, None  # Skip this index

        time_filtered, magnification_filtered = result
        # plot_magnification(time_filtered, magnification_filtered, df, index)

        # Make a copy of the row to avoid modifying the original DataFrame
        row_copy = df.iloc[index:index+1].copy()
        
        # Perform PSPL fit
        row_copy = pspl_fit(time_filtered, magnification_filtered, index, row_copy)
        # plot_pspl(time_filtered, magnification_filtered, index,row_copy)
        
        # Compute residuals and their statistics
        row_copy, residual = residuals(time_filtered, magnification_filtered, index, row_copy)
        row_copy = residuals_bin(time_filtered, magnification_filtered, index, residual, row_copy)
        
        # Fit and analyze with Chebyshev polynomials
        row_copy, cheby_y = fit_Cheby(time_filtered, magnification_filtered, index, row_copy, degree=50)
        # plot_cheby(time_filtered, magnification_filtered, index, cheby_y, row_copy)
        
        # Compute skewness and von Neumann ratio
        row_copy = calculate_skewness(magnification_filtered, index, row_copy)
        row_copy = calculate_von_neumann(magnification_filtered, index, row_copy)
        
        return index, row_copy
    
    except Exception as e:
        print(f"Error processing index {index}: {str(e)}")
        return index, None




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    multiprocessing.set_start_method('spawn')

    # Load the data
    df = extract_lightcurve_data("./")
    if df.empty:
        print("No data found. Check input directories.")
        sys.exit(1)

    # Define expected columns in the correct order
    expected_columns = [
        'Class', 'Source', 'File', 'Band', 'baseline_magnitude', 'zeroth_magnitude',
        't0', 'tE', 'u0', 'blending', 'chi2',
        'residual_mean', 'residual_median', 'residual_std',
        'residual_bin_mean', 'residual_bin_median', 'residual_bin_std',
        'Cheby_a0', 'Cheby_a2', 'Cheby_a4', 'Cheby_a6', 'Cheby_a8', 'Cheby_a10',
        'Cheby_cj_sqr', 'pos_log10_Cheby_cj_sqr_minus_one', 'log10_Cheby_cj_sqr_minus_one', 
        'delta_A_chebyshev_sqr', 'skewness', 'von_neumann'
    ]
    
    # Create an empty DataFrame with the expected columns
    output_df = pd.DataFrame(columns=expected_columns)
    
    # Write the header to the output file
    output_df.to_csv("processed_features.txt", index=False)
    
    num_cores = multiprocessing.cpu_count()
    print('Number of cores available:', num_cores)
    print('Total number of simulations:', len(df))

    start_time = time.time()
    indices = list(range(len(df)))
    
    with tqdm(total=len(indices), desc="Processing batches") as pbar:
        for i in range(0, len(indices), num_cores):
            batch_indices = indices[i:i+num_cores]
            batch_results = []
            
            with multiprocessing.Pool(processes=len(batch_indices)) as pool:
                batch_results = pool.starmap(process, [(idx, df) for idx in batch_indices])
            
            # Collect valid results for this batch
            batch_df = pd.DataFrame()
            for idx, result_row in batch_results:
                if result_row is not None:
                    batch_df = pd.concat([batch_df, result_row], ignore_index=True)
            
            # Append this batch's results to the output file
            if not batch_df.empty:
                # Ensure columns are in the correct order
                batch_df = batch_df.reindex(columns=expected_columns)
                batch_df.to_csv("processed_features.txt", mode='a', header=False, index=False)
            
            pbar.update(len(batch_indices))
    
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print("Processing complete. Results saved to processed_results.csv")