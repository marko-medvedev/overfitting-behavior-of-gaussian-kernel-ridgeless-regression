from flask_migrate import current

from gaussian_kernel_overfitting import *
import scipy.stats as stats
from plot_experiments import drop_na_ms

ds=[6,4,6]
noise_levels = [1,10,10000]
alphas = [0.01,1000,1]
#ms_sequences = [[100,200,300,400]+[500*i for i in range(1,5)], [100*i for i in range(1,10)], [500*i for i in range(1,10)]]
sequence = [100*i for i in range(1,10)]+[500*i for i in range(2,10)]+[1000*i for i in range(5,12)]
ms_sequences = [sequence for i in range(len(ds))]
max_ms = [np.max(np.array(ms)) for ms in ms_sequences]

# number_of_runs= 50

def split_into_two_sets(df, split = 1/4):
    length_of_m = len(df['m'].unique())
    number_of_experiments = int(len(df)/length_of_m) if length_of_m!=0 else 0
    if number_of_experiments == 0:
        print(f"Error, number of experiments is {number_of_experiments}")
    validation_set_size = int(number_of_experiments*split)*length_of_m
    validation_set = df[:validation_set_size] # check that this doesn't overlap
    test_set = df[validation_set_size:]
    return validation_set, test_set



def estimate_quantiles(validation_set):
    aggregated_df = validation_set.groupby("m")["test_error"].agg(
        mean_test_error="mean",
        std_test_error="std",
        min_test_error="min",
        max_test_error="max"
    ).reset_index()
    L_max = aggregated_df[["max_test_error","m"]] # does anything else make sense - if the spread is high then this doesn't make sense
    L_min = aggregated_df[["min_test_error","m"]]
    #print(f"L_max is {L_max}")
    return L_min, L_max # what else makes sense here except just reporting the max


def lower_tailed_test(n, p_star, alpha, T_1):
    t_1 = stats.binom.ppf(alpha, n, p_star)
    alpha_1 = stats.binom.cdf(t_1, n, p_star)
    p_value = stats.binom.cdf(T_1, n, p_star)
    #print(f"alpha is {alpha} t_1 is {t_1} T_1 is {T_1} p_value is {p_value} ")
    return t_1, alpha_1, p_value


def upper_tailed_test(n, p_star, alpha, T_2):
    t_2 = stats.binom.ppf(1-alpha, n, p_star)
    alpha_2 = 1-stats.binom.cdf(t_2, n, p_star)
    p_value = 1- stats.binom.cdf(T_2-1, n, p_star) if T_2-1>=0 else 1
    #print(f"alpha is {alpha} t_2 is {t_2} T_2 is {T_2} p_value is {p_value} ")
    return t_2, alpha_2, p_value


def perform_tests(test_set, L_min, L_max, p_star, alpha, number_of_runs):
    ms = test_set["m"].unique()
    T_1 = [] # number <= L_max
    T_2 = [] # number < L_min
    t_1 = [] # P(Y<= t_1) = alpha for Y~Binom(n,p_star)
    t_2 = [] # P(Y<= t_2) = 1-alpha for Y~Binom(n,p_star)
    alpha_1 = [] # P(Y<=t_1)
    alpha_2 = [] # 1-P(Y<=t_2)
    upper_p_value = []
    lower_p_value = []
    for m in ms:
        current_L_max = L_max[L_max["m"]==m]["max_test_error"].iloc[0]
        current_L_min = L_min[L_min["m"]==m]["min_test_error"].iloc[0]
        current_test_set = test_set[test_set["m"]==m]
        current_T_1 = len(current_test_set[current_test_set["test_error"]<=current_L_max])
        current_T_2 = len(current_test_set[current_test_set["test_error"]<current_L_min])
        T_1.append(current_T_1)
        T_2.append(current_T_2)
        current_t_1, current_alpha_1, lower_tailed_test_p_value = lower_tailed_test(number_of_runs, 1-(1-p_star)/2, alpha, current_T_1)
        current_t_2, current_alpha_2, upper_tailed_test_p_value = upper_tailed_test(number_of_runs, (1-p_star)/2, alpha, current_T_2)
        t_1.append(current_t_1)
        t_2.append(current_t_2)
        alpha_1.append(current_alpha_1)
        alpha_2.append(current_alpha_2)
        lower_p_value.append(lower_tailed_test_p_value)
        # print(f"lower tailed test p value {lower_tailed_test_p_value}")
        upper_p_value.append(upper_tailed_test_p_value)

    return {"m":ms.tolist(), "L_max":L_max["max_test_error"].values.tolist(), "L_min":L_min["min_test_error"].values.tolist(),  "T_1":T_1, "T_2":T_2, "t_1":t_1, "t_2":t_2, "alpha_1":alpha_1, "alpha_2":alpha_2, "number_of_runs":[number_of_runs for _ in ms], "lower_p_value":lower_p_value, "upper_p_value":upper_p_value}



for experiment_number in range(len(ds)):
    d=ds[experiment_number]
    noise_level = noise_levels[experiment_number]
    alpha = alphas[experiment_number]
    ms = ms_sequences[experiment_number]
    max_m = max_ms[experiment_number]


    file_path = 'results_' + str(noise_level) + "_" + str(d) + "_" + str(max_m) +"_"+str(alpha)+ "_aggregated" + '.pkl'
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    df = drop_na_ms(df)
    ms = df["m"].unique()
    #print(f"df is {df[0:10]}")
    split = 1/2
    number_of_runs = 50
    validation_set, test_set = split_into_two_sets(df, split = split)
    L_min, L_max = estimate_quantiles(validation_set)

    p_star = 0.8
    alpha = 0.05

    results = perform_tests(test_set ,L_min ,L_max , p_star = p_star, alpha= alpha, number_of_runs = number_of_runs)

    print(results)

    null_error = 100 + noise_level
    # plt.plot(ms, L_min, color="blue")
    L_max = L_max["max_test_error"].values.tolist()
    L_min = L_min["min_test_error"].values.tolist()
    plt.plot(ms, L_max, color="blue", label = "Max loss estimate")
    plt.plot(ms, L_min, color="blue", label = "Min loss estimate")

    plt.axhline(y=noise_level, color='red', linestyle='--', label="noise level")
    plt.axhline(y=null_error, color='orange', linestyle='--', label="null predictor error")

    # Labels and legend
    plt.xlabel("m")
    plt.ylabel("error")
    plt.legend()
    # plt.title(f"sigma^2 = {noise_level}, alpha = {alpha}, d= {d}")
    plt.show()

    last_10_indices = slice(-10, None)
    ms_last_10 = results["m"][last_10_indices]
    T_1_last_10 = results["T_1"][last_10_indices]
    T_2_last_10 = results["T_2"][last_10_indices]
    t_1_last_10 = results["t_1"][last_10_indices]
    t_2_last_10 = results["t_2"][last_10_indices]
    lower_p_value_last_10 = results["lower_p_value"][last_10_indices]
    upper_p_value_last_10 = results["upper_p_value"][last_10_indices]
    alpha_1_last_10 = results["alpha_1"][last_10_indices]
    alpha_2_last_10 = results["alpha_2"][last_10_indices]

    # X-axis positions for bar groups
    x = np.arange(len(ms_last_10))

    # Create a figure with 1 row and 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    width = 0.35  # width of the bars

    # T_1 vs t_1
    ax[0].bar(x - width / 2, T_1_last_10, width, label='T_1')
    ax[0].bar(x + width / 2, t_1_last_10, width, label='t_1')
    ax[0].set_title('Comparison of T_1 and t_1')
    ax[0].set_xlabel('Sample size m')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ms_last_10)
    ax[0].legend()

    # T_2 vs t_2
    ax[1].bar(x - width / 2, T_2_last_10, width, label='T_2')
    ax[1].bar(x + width / 2, t_2_last_10, width, label='t_2')
    ax[1].set_title('Comparison of T_2 and t_2')
    ax[1].set_xlabel('Sample size m')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ms_last_10)
    ax[1].legend()

    # Plotting p-value and alpha
    sum_p_values = [lp + up-1 for lp, up in zip(lower_p_value_last_10, upper_p_value_last_10)]
    sum_alpha_values = [a1 + a2 for a1, a2 in zip(alpha_1_last_10, alpha_2_last_10)]

    ax[2].bar(x - width / 2, sum_p_values, width, label='p-value')
    ax[2].bar(x + width / 2, sum_alpha_values, width, label='alpha')
    ax[2].set_title('Comparison of p-values and significance level')
    ax[2].set_xlabel('Sample size m')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(ms_last_10)
    ax[2].legend()

    plt.tight_layout()
    plt.show()
