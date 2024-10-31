from gaussian_kernel_overfitting import *

#number_of_experiments = 3
ds=[6,4,6]
noise_levels = [1,10,10000]
alphas = [0.01,1000,1]
#ms_sequences = [[100,200,300,400]+[500*i for i in range(1,5)], [100*i for i in range(1,10)], [500*i for i in range(1,10)]]
#sequence = [100*i for i in range(1,10)]+[500*i for i in range(2,10)]+[1000*i for i in range(5,50)]
#ms_sequences = [sequence for i in range(len(ds))]
sequence = [100*i for i in range(1,10)]+[500*i for i in range(2,10)]+[1000*i for i in range(5,12)]
ms_sequences = [sequence for i in range(len(ds))]
max_ms = [np.max(np.array(ms)) for ms in ms_sequences]

def drop_na_ms(df):
    ms = df["m"].unique()
    drop_m = []
    for m in ms:
        test_error = df[df["m"]==m]["test_error"]
        if not df[df["m"]==m]["test_error"].notna().all():
            #print(f"a bad m is {m}")
            drop_m.append(m)
    #print(f"we are dropping {drop_m}")
    df = df[df['m'].apply(lambda x: x not in drop_m)]
    left_ms = df["m"].unique()
    #print(f"we are left with {left_ms} and length {len(df)}")
    return df

def plot_mean_with_errors(ds,noise_levels,alphas, max_ms):
    for experiment_number in range(len(ds)):
        d=ds[experiment_number]
        noise_level = noise_levels[experiment_number]
        alpha = alphas[experiment_number]
        max_m = max_ms[experiment_number]


        file_path = 'results_' + str(noise_level) + "_" + str(d) + "_" + str(max_m) +"_"+str(alpha)+ "_aggregated" + '.pkl'
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            print(f"the length is {len(df)}")
            print(df)

        df = drop_na_ms(df)

        aggregated_df = df.groupby("m")["test_error"].agg(
            mean_test_error="mean",
            std_test_error="std",
            min_test_error="min",
            max_test_error="max"
        ).reset_index()

        ms = aggregated_df['m']
        mean_test_error = aggregated_df['mean_test_error']
        std_test_error = aggregated_df['std_test_error']
        null_error = 100 + noise_level
        print(ms)
        print(mean_test_error)
        print(std_test_error)
        # print(noise_level)
        # print(null_error)

        # Plot with error bars
        plt.errorbar(ms, mean_test_error, yerr=std_test_error, capsize=1)
        plt.plot(ms, mean_test_error, label="mean test error", color="lightblue")
        # Add horizontal lines
        plt.axhline(y=noise_level, color='red', linestyle='--', label="noise level")
        plt.axhline(y=null_error, color='orange', linestyle='--', label="null predictor error")

        # Labels and legend
        plt.xlabel("m")
        plt.ylabel("error")
        plt.legend()
        plt.title(f"sigma^2 = {noise_level}, alpha = {alpha}, d= {d}")
        plt.show()

def plot_all_runs(ds, noise_levels, alphas, max_ms):
    for experiment_number in range(len(ds)):
        d = ds[experiment_number]
        noise_level = noise_levels[experiment_number]
        alpha = alphas[experiment_number]
        max_m = max_ms[experiment_number]

        file_path = 'results_' + str(noise_level) + "_" + str(d) + "_" + str(max_m) + "_" + str(
            alpha) + "_aggregated" + '.pkl'
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            print(f"the length is {len(df)}")
            print(df)

        ms = df['m'].unique()
        length_of_m = len(ms)
        number_of_runs = int(len(df) / length_of_m) if length_of_m != 0 else 0
        if number_of_runs == 0:
            print(f"Error, number of experiments is {number_of_runs}")
        df['run'] = (df.index // length_of_m)  # create group labels based on length_of_m

        df = drop_na_ms(df)
        ms = df["m"].unique()
        dfs = [run for _, run in df.groupby('run')]





        aggregated_df = df.groupby("m")["test_error"].agg(
            mean_test_error="mean",
            min_test_error="min",
            max_test_error="max"
        ).reset_index()

        null_error = 100 + noise_level

        for current_df in dfs:
            print(f"length of ms is {len(ms)} length of current_df is {len(current_df)}")
            # print(f"current_df is {current_df}")
            plt.plot(ms, current_df['test_error'], color="lightblue")
        plt.plot(ms, aggregated_df['mean_test_error'], color="blue")
        #plt.plot(ms, aggregated_df['min_test_error'], color="blue")
        #plt.plot(ms, aggregated_df["max_test_error"], color="blue")
        plt.axhline(y=noise_level, color='red', linestyle='--', label="noise level")
        plt.axhline(y=null_error, color='orange', linestyle='--', label="null predictor error")

        # Labels and legend
        plt.xlabel("m")
        plt.ylabel("error")
        plt.legend()
        #plt.title(f"sigma^2 = {noise_level}, alpha = {alpha}, d= {d}")
        plt.show()

# plot_all_runs(ds, noise_levels, alphas, max_ms)