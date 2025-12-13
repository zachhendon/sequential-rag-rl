import wandb
import argparse
import os

def download_run(run_path, save_path):
    # Download the run data from wandb
    api = wandb.Api()
    run = api.run(run_path)

    # Create a dataframe with validation accuracy for each step and various configuration parameters
    run_df = run.history(keys=["val_accuracy", "val_examples_per"])
    run_df["generator_model"] = run.config["generator_model"]
    run_df["reward"] = run.config["reward"]
    run_df["int_reward_margin"] = run.config["int_reward_margin"]
    run_df["cr_coef"] = run.config["cr_coef"]

    # Save the dataframe to a csv file
    if not save_path.endswith(".csv"):
        save_path = save_path + ".csv"
    run_df.to_csv(
        save_path, 
        mode="a", 
        header=not os.path.exists(save_path), 
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str, required=True, help="The path to the wandb run to download. Can be found by clicking options -> copy -> run path on a specific run.")
    parser.add_argument("-s", "--save_path", type=str, default="train.csv", help="The csv file to save the downloaded run data. Default save path is 'results/train.csv")
    args = parser.parse_args()

    download_run(args.run_path, args.save_path)
