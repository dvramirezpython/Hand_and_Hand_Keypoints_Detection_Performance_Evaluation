import numpy as np
import scipy.stats as stats

def calculate_cohens_d_paired(group1, group2):
    """
    Calculates Cohen's d for paired (dependent) samples.
    Formula: mean(differences) / std_deviation(differences)
    """
    diff = np.array(group1) - np.array(group2)
    mean_diff = np.mean(diff)
    # Using ddof=1 for sample standard deviation
    std_diff = np.std(diff, ddof=1) 
    
    if std_diff == 0:
        return 0.0
    
    cohens_d = mean_diff / std_diff
    return cohens_d

def calculate_rank_biserial_paired(group1, group2):
    """
    Calculates the Rank-Biserial Correlation (r) for paired samples 
    matched with a Wilcoxon Signed-Rank Test.
    Formula: r = Wilcoxon_W_statistic / Total_Rank_Sum
    Effect size ranges from -1 to 1.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    diff = g1 - g2
    
    # Remove pairs with zero differences (Wilcoxon standard practice)
    diff = diff[diff != 0]
    n = len(diff)
    
    if n == 0:
        return 0.0
    
    # Get absolute ranks
    ranks = stats.rankdata(np.abs(diff))
    
    # Calculate sum of positive and negative ranks
    pos_ranks_sum = np.sum(ranks[diff > 0])
    neg_ranks_sum = np.sum(ranks[diff < 0])
    
    # Total sum of ranks
    total_rank_sum = np.sum(ranks)   # Mathematically equal to n * (n + 1) / 2
    
    # Rank-biserial correlation formula
    # (Difference between positive and negative rank sums divided by total rank sum)
    r = (pos_ranks_sum - neg_ranks_sum) / total_rank_sum
    return r

def run_statistical_analysis(regular_mAP, irregular_mAP):
    print("=== Statistical Analysis Results ===")
    print(f"Number of algorithms (n): {len(regular_mAP)}\n")
    
    # 1. Wilcoxon Signed-Rank Test
    # alternative='two-sided' checks if there is any significant difference.
    # alternative='greater' checks if regular_mAP is significantly larger than irregular_mAP.
    statistic, p_value = stats.wilcoxon(regular_mAP, irregular_mAP, alternative='two-sided')
    print(f"Wilcoxon Signed-Rank Test:")
    print(f"  - Statistic (W): {statistic}")
    print(f"  - p-value: {p_value:.5f}")
    if p_value < 0.05:
        print("  - Result: Statistically SIGNIFICANT difference (p < 0.05)")
    else:
        print("  - Result: NOT statistically significant (p >= 0.05)")
        
    print("\n=== Effect Sizes ===")
    
    # 2. Cohen's d (Parametric Effect Size)
    d_val = calculate_cohens_d_paired(regular_mAP, irregular_mAP)
    print(f"Cohen's d (Paired): {d_val:.4f}")
    print(f"  - Interpretation: {interpret_cohens_d(d_val)}")
    
    # 3. Rank-Biserial Correlation (Non-parametric Effect Size)
    r_val = calculate_rank_biserial_paired(regular_mAP, irregular_mAP)
    print(f"Rank-Biserial Correlation (r): {r_val:.4f}")
    print(f"  - Interpretation: {interpret_rank_biserial(r_val)}")

def interpret_cohens_d(d):
    abs_d = abs(d)
    if abs_d < 0.2: return "Negligible"
    elif abs_d < 0.5: return "Small effect"
    elif abs_d < 0.8: return "Medium effect"
    else: return "Large effect"

def interpret_rank_biserial(r):
    abs_r = abs(r)
    if abs_r < 0.1: return "Negligible"
    elif abs_r < 0.3: return "Small effect"
    elif abs_r < 0.5: return "Medium effect"
    else: return "Large effect"

if __name__ == "__main__":
    # Index 0 is Algo 1, Index 1 is Algo 2, etc.
    print('============= Statistical Analysis for mAP50 Scores =============')
    # Data for mAP50 scores for regular and irregular hands across 6 algorithms
    regular_hands_mAP = [0.9770, 0.9940, 0.9942, 0.9949, 0.9948, 0.9950]
    irregular_hands_mAP = [0.4153, 0.9855, 0.9942, 0.9939, 0.9946, 0.9828]
    run_statistical_analysis(regular_hands_mAP, irregular_hands_mAP)
    
    print('============= Statistical Analysis for mAP50-95 Scores =============')
    # Data for mAP50-95 scores for regular and irregular hands across 6 algorithms
    regular_hands_mAP = [0.8258, 0.9495, 0.9750, 0.9743, 0.9720, 0.9723]
    irregular_hands_mAP = [0.2033, 0.8693, 0.8852, 0.8859, 0.8839, 0.8771]
    run_statistical_analysis(regular_hands_mAP, irregular_hands_mAP)

    print('============= Statistical Analysis for Precision Scores =============')
    # Data for Precision scores for regular and irregular hands across 6 algorithms
    regular_hands_mAP = [0.9919, 0.9978, 0.9957, 0.9978, 0.9978, 0.9979]
    irregular_hands_mAP = [0.6447, 0.9884, 0.9891, 0.9880, 0.9918, 0.9801]
    run_statistical_analysis(regular_hands_mAP, irregular_hands_mAP)

    print('============= Statistical Analysis for Recall Scores =============')
    # Data for Recall scores for regular and irregular hands across 6 algorithms
    regular_hands_mAP = [0.9840, 0.9958, 0.9958, 0.9979, 0.9979, 0.9976]
    irregular_hands_mAP = [0.6460, 0.9840, 0.9860, 0.9878, 0.9900, 0.9833]
    run_statistical_analysis(regular_hands_mAP, irregular_hands_mAP)
    # run_bayesian_analysis(regular_hands_mAP, irregular_hands_mAP)