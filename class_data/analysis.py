import os
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt  # For plotting boxplots
from scipy.stats import ttest_rel

def display_max_min_values(d_prime_short, d_prime_long, proportion_correct_short, proportion_correct_long):
    print("\nSummary of Maximum and Minimum Values:")
    
    # d' values
    print(f"  Max d' (Short Condition): {max(d_prime_short) if d_prime_short else 'No data'}")
    print(f"  Min d' (Short Condition): {min(d_prime_short) if d_prime_short else 'No data'}")
    print(f"  Max d' (Long Condition): {max(d_prime_long) if d_prime_long else 'No data'}")
    print(f"  Min d' (Long Condition): {min(d_prime_long) if d_prime_long else 'No data'}")
    
    # Proportion correct values
    print(f"  Max Proportion Correct (Short Condition): {max(proportion_correct_short) if not proportion_correct_short.empty else 'No data'}")
    print(f"  Min Proportion Correct (Short Condition): {min(proportion_correct_short) if not proportion_correct_short.empty else 'No data'}")
    print(f"  Max Proportion Correct (Long Condition): {max(proportion_correct_long) if not proportion_correct_long.empty else 'No data'}")
    print(f"  Min Proportion Correct (Long Condition): {min(proportion_correct_long) if not proportion_correct_long.empty else 'No data'}")

def plot_proportion_correct(proportion_correct_short, proportion_correct_long):
    # Combine data for plotting
    proportion_correct_data = [proportion_correct_short, proportion_correct_long]

    # Plot Proportion Correct boxplot for participants
    plt.figure(figsize=(8, 5))
    plt.boxplot(proportion_correct_data, labels=['Short Condition', 'Long Condition'])
    plt.title("Participant-Level Proportion Correct by Condition")
    plt.ylabel("Proportion Correct")
    plt.ylim(0, 1)  # Proportions are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Function to classify trials as SN or NS based on the target location
def classify_trial(row):
    if row['left_image'].startswith('target'):
        return 'SN'  # Signal-Noise
    elif row['right_image'].startswith('target'):
        return 'NS'  # Noise-Signal
    else:
        return 'Unknown'  # Fallback in case of unexpected data
def plot_participant_level_dprime_lambda(d_prime_short, d_prime_long, lambda_short, lambda_long):
    # Combine data for plotting
    d_prime_data = [d_prime_short, d_prime_long]
    lambda_data = [lambda_short, lambda_long]

    # Plot D-prime boxplot for participants
    plt.figure(figsize=(8, 5))
    plt.boxplot(d_prime_data, labels=['Short Condition', 'Long Condition'])
    plt.title("Participant-Level D-prime Distribution by Condition")
    plt.ylabel("D-prime (FC)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot Lambda boxplot for participants
    plt.figure(figsize=(8, 5))
    plt.boxplot(lambda_data, labels=['Short Condition', 'Long Condition'])
    plt.title("Participant-Level Lambda Distribution by Condition")
    plt.ylabel("Lambda (FC)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Main function to process data
def calculate_hit_false_alarm_rates(folder_path='/Users/codylejang/Desktop/eyewitness/class_data'):
    # Find all CSV files in the specified folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize an empty list to store all datasets
    all_data = []

    # Assign participant IDs based on file index
    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            df['participant_id'] = f'participant_{i+1}'  # Add unique participant ID
            df['source_file'] = file  # Add source file name
            df['trial_type'] = df.apply(classify_trial, axis=1)

            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine all datasets into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
    else:
        print("No valid CSV files found.")
        return

    # Add a column to categorize "short" and "long" conditions based on encoding_duration
    combined_data['condition'] = combined_data['encoding_duration'].apply(
        lambda x: 'short' if x < 1 else 'long'
    )

    combined_data.to_csv('eyewitnesstotal.csv')

    # Group data by condition
    grouped = combined_data.groupby('condition')

    # Initialize results dictionary
    total_results = {}

    # Calculate aggregate metrics for each condition
    for condition, group in grouped:
        # Calculate proportions for SN and NS trials
        p_c_sn = group[group['trial_type'] == 'SN']['correct'].mean()  # Proportion correct for SN
        p_c_ns = group[group['trial_type'] == 'NS']['correct'].mean()  # Proportion correct for NS

        # Calculate overall proportion correct
        overall_p_c = group['correct'].mean()

        # Apply boundary corrections to avoid Z-scores of infinity
        if p_c_sn == 1:
            p_c_sn -= 0.5 / group[group['trial_type'] == 'SN'].shape[0]
        elif p_c_sn == 0:
            p_c_sn += 0.5 / group[group['trial_type'] == 'SN'].shape[0]

        if p_c_ns == 1:
            p_c_ns -= 0.5 / group[group['trial_type'] == 'NS'].shape[0]
        elif p_c_ns == 0:
            p_c_ns += 0.5 / group[group['trial_type'] == 'NS'].shape[0]

        # Calculate Z-scores
        Z_sn = st.norm.ppf(p_c_sn)
        Z_ns = st.norm.ppf(p_c_ns)

        # Calculate d', lambda, and log beta using the new formulas
        d_prime_fc = Z_sn + Z_ns  # New d'
        lambda_fc = 0.5 * (Z_ns - Z_sn)  # New lambda
        log_beta_fc = 0.5 * (Z_ns**2 - Z_sn**2)  # New log beta

        # Store results for the condition
        total_results[condition] = {
            'Overall Proportion Correct': overall_p_c,
            'P_C_SN': p_c_sn,
            'P_C_NS': p_c_ns,
            'D-prime (FC)': d_prime_fc,
            'Lambda (FC)': lambda_fc,
            'Log Beta (FC)': log_beta_fc
        }

    # Print aggregate results for each condition
    print("\nAggregate Condition-Level Stats:")
    for condition, stats in total_results.items():
        print(f"\n{condition.capitalize()} Condition:")
        print(f"  Overall Proportion Correct (Hit Rate): {stats['Overall Proportion Correct']}")
        print(f"  False Alarm: {1 - stats['Overall Proportion Correct']}")
        print(f"  D-prime (FC): {stats['D-prime (FC)']}")
        print(f"  Lambda (FC): {stats['Lambda (FC)']}")
        print(f"  Log Beta (FC): {stats['Log Beta (FC)']}")

    # Group data by participant and condition for participant-level stats
    grouped_participants = combined_data.groupby(['participant_id', 'condition'])

    # Store participant-level d', lambda, and log beta for t-tests
    d_prime_short = []
    d_prime_long = []
    lambda_short = []
    lambda_long = []

    # Calculate participant-level metrics
    for (participant, condition), group in grouped_participants:
        p_c_sn = group[group['trial_type'] == 'SN']['correct'].mean()
        p_c_ns = group[group['trial_type'] == 'NS']['correct'].mean()

        # Apply boundary corrections
        if p_c_sn == 1:
            p_c_sn -= 0.5 / group[group['trial_type'] == 'SN'].shape[0]
        elif p_c_sn == 0:
            p_c_sn += 0.5 / group[group['trial_type'] == 'SN'].shape[0]

        if p_c_ns == 1:
            p_c_ns -= 0.5 / group[group['trial_type'] == 'NS'].shape[0]
        elif p_c_ns == 0:
            p_c_ns += 0.5 / group[group['trial_type'] == 'NS'].shape[0]

        # Calculate Z-scores
        Z_sn = st.norm.ppf(p_c_sn)
        Z_ns = st.norm.ppf(p_c_ns)

        # Calculate d', lambda, and log beta
        d_prime_fc = Z_sn + Z_ns
        lambda_fc = 0.5 * (Z_ns - Z_sn)

        # Append d' and lambda to their respective lists for t-tests
        if condition == 'short':
            d_prime_short.append(d_prime_fc)
            lambda_short.append(lambda_fc)
        elif condition == 'long':
            d_prime_long.append(d_prime_fc)
            lambda_long.append(lambda_fc)

    # Perform paired t-tests for participant-level d' and lambda
    if len(d_prime_short) > 1 and len(d_prime_long) > 1:
        d_prime_ttest = ttest_rel(d_prime_short, d_prime_long)
        lambda_ttest = ttest_rel(lambda_short, lambda_long)

        # Print t-test results
        print("\nPaired T-Test Results:")
        print(f"  D-prime T-test: t-statistic = {d_prime_ttest.statistic}, p-value = {d_prime_ttest.pvalue}")
        print(f"  Lambda T-test: t-statistic = {lambda_ttest.statistic}, p-value = {lambda_ttest.pvalue}")
    else:
        print("Insufficient data for paired t-tests.")

    # Example calculation for proportion correct
    proportion_correct_short = combined_data[
        (combined_data['condition'] == 'short') & (combined_data['correct'] == 1)].groupby('participant_id').size() / combined_data[
        combined_data['condition'] == 'short'].groupby('participant_id').size()

    proportion_correct_long = combined_data[(combined_data['condition'] == 'long') & (combined_data['correct'] == 1)].groupby('participant_id').size() / combined_data[
    combined_data['condition'] == 'long'].groupby('participant_id').size()
    
    display_max_min_values(d_prime_short, d_prime_long, proportion_correct_short, proportion_correct_long)
    
    plot_participant_level_dprime_lambda(d_prime_short, d_prime_long, lambda_short, lambda_long)
    plot_proportion_correct(proportion_correct_short.tolist(), proportion_correct_long.tolist())
# Call the function
calculate_hit_false_alarm_rates()
